import asyncio
import random
from openai import RateLimitError


async def exponential_backoff(attempt):
    """
    Implements exponential backoff with a jitter.

    Args:
        attempt: The current retry attempt number.

    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    wait_time = (2**attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.

    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.

    Returns:
        The result of the coroutine if successful.

    Raises:
        The last encountered exception if all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            return await coroutine
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            await exponential_backoff(attempt)
    raise Exception("Max retries reached")
