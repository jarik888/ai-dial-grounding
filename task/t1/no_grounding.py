import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

# 1. Create AzureChatOpenAI client
azure_chat_openai = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)
# 2. Create TokenTracker
token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...
    formatted_context = "\r\n".join(
        [
            f"User:\r\n  " + "\r\n  ".join(f"{key}: {value}" for key, value in user.items())
            for user in context
        ]
    )
    return formatted_context


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages array with system prompt and user message
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_message)
    ]

    # 2. Generate response
    response = await azure_chat_openai.ainvoke(messages, stop=None)

    # 3. Get usage
    token_usage = response.response_metadata["token_usage"]["total_tokens"]

    # 4. Add tokens to `token_tracker`
    token_tracker.add_tokens(token_usage)

    # 5. Print response content and `total_tokens`
    print("\r\nGenerated Response:\r\n", response.content)
    print("Total Tokens Used:", token_tracker.total_tokens)
    # 5. return response content
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\r\n--- Searching user database ---")

        # 1. Get all users (use UserClient)
        all_users = UserClient().get_all_users()

        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        user_batches = [
            all_users[i : i + 100] for i in range(0, len(all_users), 100)
        ]

        # 3. Prepare tasks for async run of response generation for users batches:
        tasks = []
        for batch in user_batches:
            context = join_context(batch)
            user_prompt = USER_PROMPT.format(context=context, query=user_question)
            tasks.append(generate_response(BATCH_SYSTEM_PROMPT, user_prompt))

        # 4. Run task asynchronously, use method `gather` form `asyncio`
        responses = await asyncio.gather(*tasks)

        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        filtered_results = [response for response in responses if response != "NO_MATCHES_FOUND"]

        # 5. If results after filtration are present
        if filtered_results:
        #       - combine filtered results with "\n\n" spliterator
            combined_results = "\r\n\r\n".join(filtered_results)

        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
            user_prompt = USER_PROMPT.format(context=combined_results, query=user_question)
            final_response = await generate_response(FINAL_SYSTEM_PROMPT, user_prompt)

            print("\r\n--- Final Response ---")
            print(final_response)

        # 6. Otherwise prin the info that `No users found matching`
        else:
            print("\r\nNo users found matching")

        # 7. In the end print info about usage
        print("\r\nToken Usage Summary:", token_tracker.get_summary())


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation