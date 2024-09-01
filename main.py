import asyncio
from dotenv import load_dotenv
from core.hierarchical_retriever import HierarchicalRetriever
from core.vector_store_manager import VectorStoreManager
from services.openai_service import OpenAIService


class Application:
    """
    The Application class is responsible for orchestrating the main logic of the hierarchical
    Q&A system. It loads or creates vector stores, handles user queries, and retrieves relevant
    information from the document.

    Attributes:
        path (str): The path to the PDF document.
        vector_store_manager (VectorStoreManager): Manages the loading and creation of vector stores.
        openai_service (OpenAIService): Provides methods to interact with OpenAI's API.
    """

    def __init__(self):
        """
        Initializes the Application by loading environment variables and setting up
        the path and vector store manager.
        """
        load_dotenv()
        self.path = "./data/intro_to_algo_book.pdf"
        self.vector_store_manager = VectorStoreManager()
        self.openai_service = OpenAIService()

    async def run(self):
        """
        The main function that runs the application. It loads or creates vector stores,
        defines a query, performs hierarchical retrieval, and presents the results.
        """
        # Load or create vector stores
        summary_store, detailed_store = (
            await self.vector_store_manager.get_or_create_vector_stores(self.path)
        )

        # Define a query
        query = "What is the QuickSelect algo?"

        # Perform hierarchical retrieval
        retriever = HierarchicalRetriever(summary_store, detailed_store)
        results = retriever.retrieve(query)

        # Present the results
        for chunk in results:
            print(f"Page: {chunk.metadata['page']}")
            print(f"Content: {chunk.page_content[:100]}...")
            print("---")


if __name__ == "__main__":
    app = Application()
    asyncio.run(app.run())
