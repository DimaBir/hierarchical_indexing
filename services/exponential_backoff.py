import asyncio
import random
from openai import RateLimitError


async def exponential_backoff(attempt):
    """
    Implements exponential backoff with jitter. The wait time increases exponentially
    with each retry attempt and includes a random jitter.

    Args:
        attempt (int): The current retry attempt number.

    Returns:
        None
    """
    wait_time = (2**attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine_factory, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.

    Args:
        coroutine_factory (Callable): A factory function that returns a coroutine to be executed.
        max_retries (int): The maximum number of retry attempts.

    Returns:
        Any: The result of the coroutine if successful.

    Raises:
        Exception: If all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            # Call the factory to get a new coroutine and attempt to execute it
            return await coroutine_factory()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            await exponential_backoff(attempt)
    raise Exception("Max retries reached")

