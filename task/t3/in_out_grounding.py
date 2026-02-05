import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient
import json

# Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

# Prompt templates
QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts hobby information from user questions about users.

## Instructions:
1. Analyze the user's question and identify hobbies mentioned
2. Extract specific hobby names mentioned in the query
3. If multiple hobbies are mentioned, include all of them
4. Only extract explicit hobby values - don't infer or assume hobbies not mentioned
5. Return hobbies in lowercase for consistency

## Examples:
- "Who loves hiking?" → hobbies: ["hiking"]
- "Find people who enjoy camping and rock climbing" → hobbies: ["camping", "rock climbing"]
- "I need people who love to go to mountains" → hobbies: ["mountain hiking", "rock climbing", "camping"]

## Response Format:
{format_instructions}
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user hobbies.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""

# Create AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small-1",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    dimensions=384
)

# Create AzureChatOpenAI client
llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)

# Create UserClient
user_client = UserClient()

# Pydantic models for output parsing
class HobbySearchRequest(BaseModel):
    hobby: str = Field(description="The hobby to search for")

class HobbySearchRequests(BaseModel):
    hobbies: list[HobbySearchRequest] = Field(
        default_factory=list,
        description="A list of hobby search requests"
    )

class HobbyGroup(BaseModel):
    hobby: str = Field(description="The hobby name")
    user_ids: list[str] = Field(description="List of user IDs with this hobby")

class HobbyGroups(BaseModel):
    hobby_groups: list[HobbyGroup] = Field(
        default_factory=list,
        description="List of hobby groups with associated user IDs"
    )


async def create_vectorstore():
    """Create and initialize the vectorstore with all users."""
    print("🔎 Loading all users...")

    # Get all users
    all_users = user_client.get_all_users()

    # Prepare array of Documents with user id and about_me fields only
    documents = [
        Document(id=user.get('id'), page_content=format_user_document(user))
        for user in all_users
    ]

    # Create Chroma vectorstore
    vectorstore = Chroma(
        collection_name="user_hobbies",
        embedding_function=embeddings
    )

    # Add documents to vectorstore
    await vectorstore.aadd_documents(documents)

    print("✅ Vectorstore is ready.")
    return vectorstore


async def update_vectorstore(vectorstore: Chroma):
    """Update the vectorstore by adding new users and removing deleted ones."""
    print("🔄 Updating vectorstore with changes in the User Service...")

    # Fetch updated user snapshots from the User Service
    active_users = user_client.get_all_users()
    active_user_ids = {str(user.get('id')): user for user in active_users}
    active_user_ids_set = set(active_user_ids.keys())

    # Fetch current IDs in the vectorstore
    existing_records = vectorstore.get()
    vectorstore_user_ids_set = set(str(user_id) for user_id in existing_records.get("ids", []))

    # Identify new users to add and removed users to delete
    new_user_ids = active_user_ids_set - vectorstore_user_ids_set
    deleted_user_ids = vectorstore_user_ids_set - active_user_ids_set

    new_documents = [
        Document(id=user_id, page_content=format_user_document(active_user_ids[user_id]))
        for user_id in new_user_ids
    ]

    # Remove users from vectorstore
    if deleted_user_ids:
        print(f"🗑️ Removing {len(deleted_user_ids)} users from vectorstore.")
        vectorstore.delete(list(deleted_user_ids))

    # Add new users to vectorstore
    if new_documents:
        print(f"➕ Adding {len(new_documents)} new users to vectorstore.")
        await vectorstore.aadd_documents(new_documents)

    if deleted_user_ids and new_user_ids:
        print("✅ Vectorstore updated successfully.")
    else:
        print("ℹ️ No changes detected from the User Service. Vectorstore remains unchanged.")


async def retrieve_context(query: str, vectorstore: Chroma, k: int = 10) -> list[dict[str, Any]]:
    """Retrieve relevant user documents based on query."""

    # Perform similarity search
    results = vectorstore.similarity_search_with_relevance_scores(query, k)

    # Extract user IDs from relevant documents
    user_ids = []
    for doc, relevance_score in results:
        if relevance_score > 0.1:  # Threshold for relevance
            user_id = doc.id
            if user_id:
                user_ids.append(user_id)
                print(f"Relevance Score: {relevance_score}, User ID: {user_id}")

    # Fetch full user information for retrieved IDs
    users = []
    for user_id in user_ids:
        user = await user_client.get_user(user_id)
        if user:
            users.append(user)

    return users


def format_user_document(user: dict[str, Any]) -> str:
    return f"User ID: {user['id']}\nAbout: {user['about_me']}"


async def main():
    # Create vectorstore
    vectorstore = await create_vectorstore()

    print("Query samples:")
    print(" - I need people who love to go to mountains")
    print(" - I need people who love to go surfing")

    while True:
        user_question = input("$ ").strip()
        if user_question.lower() in ['quit', 'exit']:
            break

        # 1. Update vectorstore before search
        await update_vectorstore(vectorstore)

        # 2. Use LLM to extract hobbies from the query
        parser = PydanticOutputParser(pydantic_object=HobbySearchRequests)

        messages = [
            SystemMessagePromptTemplate.from_template(template=QUERY_ANALYSIS_PROMPT),
            HumanMessage(user_question)
        ]

        prompt = ChatPromptTemplate.from_messages(messages).partial(format_instructions=parser.get_format_instructions())

        hobby_requests = (prompt | llm_client | parser).invoke({})

        # 3. If we found hobbies, search for users with those hobbies
        if hobby_requests.hobbies:
            # For each hobby, we'll search for users
            hobby_groups = []
            for hobby_request in hobby_requests.hobbies:
                hobby = hobby_request.hobby

                # Retrieve relevant users from vectorstore
                users = await retrieve_context(hobby, vectorstore)

                # Get user IDs for output grounding
                user_ids = [str(user.get('id')) for user in users]

                # Create hobby group
                hobby_group = HobbyGroup(hobby=hobby, user_ids=user_ids)
                hobby_groups.append(hobby_group)

            # 4. Output grounding: fetch full user info for all IDs
            output_groups = []
            for hobby_group in hobby_groups:
                full_users = []
                for user_id in hobby_group.user_ids:
                    user = await user_client.get_user(user_id)
                    if user:
                        full_users.append(user)

                # Create final output group with full user info
                output_groups.append({
                    hobby_group.hobby: full_users
                })

            # Print the final structured output
            print("\r\nOutput:")
            print(json.dumps(output_groups, indent=2))
        else:
            print("\r\nNo hobbies found in query.")


if __name__ == "__main__":
    asyncio.run(main())
