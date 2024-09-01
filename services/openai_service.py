from langchain_openai import ChatOpenAI
from services.exponential_backoff import retry_with_exponential_backoff


class OpenAIService:
    """
    The OpenAIService class provides methods to interact with OpenAI's API.
    It handles the creation of summary chains and summarization of documents.

    Attributes:
        llm (ChatOpenAI): An instance of the ChatOpenAI language model.
    """

    def __init__(self):
        """
        Initializes the OpenAIService with an instance of the ChatOpenAI language model.
        """
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

    def create_summary_chain(self):
        """
        Creates a summary chain using the specified language model.

        Returns:
            Callable: A summarization chain configured with the specified language model.
        """
        from langchain.chains.summarize.chain import load_summarize_chain

        return load_summarize_chain(self.llm, chain_type="map_reduce")

    async def summarize_document(self, summary_chain, doc):
        """
        Summarizes a document using the summary chain with rate limit handling.

        Args:
            summary_chain (Callable): The summarization chain to be used.
            doc (Document): The document to be summarized.

        Returns:
            dict: The output of the summarization process.
        """
        # lambda will create a new coroutine each time it's called
        return await retry_with_exponential_backoff(lambda: summary_chain.ainvoke([doc]))

