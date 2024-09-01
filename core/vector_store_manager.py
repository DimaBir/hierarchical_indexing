import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from core.document_processor import DocumentProcessor


class VectorStoreManager:
    """
    The VectorStoreManager class is responsible for managing the creation and loading of vector stores.
    It abstracts the process of checking if vector stores exist and either loading them or creating new ones.

    Attributes:
        embeddings (OpenAIEmbeddings): An instance of OpenAIEmbeddings used for creating vector stores.
    """

    def __init__(self):
        """
        Initializes the VectorStoreManager with an instance of OpenAIEmbeddings.
        """
        self.embeddings = OpenAIEmbeddings()

    async def get_or_create_vector_stores(self, path):
        """
        Retrieves existing vector stores or creates new ones if they don't exist.

        Args:
            path (str): The path to the PDF document.

        Returns:
            Tuple[FAISS, FAISS]: A tuple containing the summary and detailed vector stores.
        """
        if self._vector_stores_exist():
            return self._load_vector_stores()
        else:
            return await self._create_vector_stores(path)

    def _vector_stores_exist(self):
        """
        Checks if the vector stores already exist.

        Returns:
            bool: True if both summary and detailed vector stores exist, False otherwise.
        """
        return os.path.exists("./vector_stores/summary_store") and os.path.exists(
            "./vector_stores/detailed_store"
        )

    def _load_vector_stores(self):
        """
        Loads the existing vector stores from the file system.

        Returns:
            Tuple[FAISS, FAISS]: A tuple containing the summary and detailed vector stores.
        """
        summary_store = FAISS.load_local(
            "./vector_stores/summary_store",
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        detailed_store = FAISS.load_local(
            "./vector_stores/detailed_store",
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return summary_store, detailed_store

    async def _create_vector_stores(self, path):
        """
        Creates new vector stores by processing the document.

        Args:
            path (str): The path to the PDF document.

        Returns:
            Tuple[FAISS, FAISS]: A tuple containing the summary and detailed vector stores.
        """
        processor = DocumentProcessor()

        summaries, detailed_chunks = await processor.load_and_process_document(path)

        summary_store = FAISS.from_documents(summaries, self.embeddings)
        detailed_store = FAISS.from_documents(detailed_chunks, self.embeddings)

        summary_store.save_local("./vector_stores/summary_store")
        detailed_store.save_local("./vector_stores/detailed_store")

        return summary_store, detailed_store
