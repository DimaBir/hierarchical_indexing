import asyncio
from tqdm.asyncio import tqdm_asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.openai_service import OpenAIService
from langchain.docstore.document import Document


class DocumentProcessor:
    """
    The DocumentProcessor class is responsible for processing documents, including loading,
    summarizing, and splitting them into chunks.

    Attributes:
        openai_service (OpenAIService): Provides methods to interact with OpenAI's API.
    """

    def __init__(self):
        """
        Initializes the DocumentProcessor with an instance of OpenAIService.
        """
        self.openai_service = OpenAIService()

    async def load_and_process_document(
        self, path, chunk_size=1000, chunk_overlap=200, is_string=False
    ):
        """
        Loads and processes a document by splitting it into chunks and creating summaries.

        Args:
            path (str): The path to the PDF document or the string content.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
            is_string (bool): Whether the input is a path to a PDF or a text string.

        Returns:
            Tuple[List[Document], List[Document]]: A tuple containing the document-level summaries
            and detailed chunks.
        """
        documents = self._load_documents(path, chunk_size, chunk_overlap, is_string)
        summaries = await self._create_document_summaries(documents, path)
        detailed_chunks = self._split_into_chunks(documents, chunk_size, chunk_overlap)
        return summaries, detailed_chunks

    @staticmethod
    def _load_documents(path, chunk_size, chunk_overlap, is_string):
        """
        Loads documents from a PDF file or a string.

        Args:
            path (str): The path to the PDF document or the string content.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
            is_string (bool): Whether the input is a path to a PDF or a text string.

        Returns:
            List[Document]: A list of loaded documents.
        """
        if not is_string:
            loader = PyPDFLoader(path)
            return loader.load()
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            return text_splitter.create_documents([path])

    async def _create_document_summaries(self, documents, path):
        """
        Creates summaries for a list of documents in batches to avoid hitting rate limits.

        Args:
            documents (List[Document]): A list of documents to be summarized.
            path (str): The path to the PDF document or the string content.

        Returns:
            List[Document]: A list of summarized documents.
        """
        summary_chain = self.openai_service.create_summary_chain()
        summaries = []

        # Process documents in smaller batches to avoid rate limits
        batch_size = 5  # Adjust this based on your rate limits and document size
        total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

        for i in tqdm_asyncio(range(0, len(documents), batch_size), desc="Summarizing documents", total=total_batches):
            batch = documents[i: i + batch_size]
            batch_summaries = await asyncio.gather(
                *[self._summarize_doc(doc, summary_chain, path) for doc in batch]
            )
            summaries.extend(batch_summaries)
            await asyncio.sleep(1)  # Short pause between batches to avoid rate limits

        return summaries

    async def _summarize_doc(self, doc, summary_chain, path):
        """
        Summarizes a single document with rate limit handling.

        Args:
            doc (Document): The document to be summarized.
            summary_chain (Callable): The summarization chain to be used.
            path (str): The path to the PDF document or the string content.

        Returns:
            Document: A summarized document object.
        """
        summary_output = await self.openai_service.summarize_document(
            summary_chain, doc
        )
        summary = Document(
            page_content=summary_output["output_text"],
            metadata={"source": path, "page": doc.metadata["page"], "summary": True},
        )
        return summary

    @staticmethod
    def _split_into_chunks(documents, chunk_size, chunk_overlap):
        """
        Splits documents into smaller chunks.

        Args:
            documents (List[Document]): A list of documents to be split into chunks.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.

        Returns:
            List[Document]: A list of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        return text_splitter.split_documents(documents)
