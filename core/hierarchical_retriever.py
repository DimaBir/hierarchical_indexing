class HierarchicalRetriever:
    """
    The HierarchicalRetriever class is responsible for performing hierarchical retrieval.
    It searches through the summary vector store first and then drills down into the detailed chunks.

    Attributes:
        summary_vectorstore (FAISS): The vector store containing document summaries.
        detailed_vectorstore (FAISS): The vector store containing detailed document chunks.
    """

    def __init__(self, summary_vectorstore, detailed_vectorstore):
        """
        Initializes the HierarchicalRetriever with summary and detailed vector stores.

        Args:
            summary_vectorstore (FAISS): The vector store containing document summaries.
            detailed_vectorstore (FAISS): The vector store containing detailed document chunks.
        """
        self.summary_vectorstore = summary_vectorstore
        self.detailed_vectorstore = detailed_vectorstore

    def retrieve(self, query, k_summaries=3, k_chunks=5):
        """
        Performs hierarchical retrieval using the query. It first searches the summary vector store
        and then drills down into the detailed vector store.

        Args:
            query (str): The search query.
            k_summaries (int): The number of top summaries to retrieve.
            k_chunks (int): The number of detailed chunks to retrieve per summary.

        Returns:
            List[Document]: A list of relevant detailed document chunks.
        """
        top_summaries = self.summary_vectorstore.similarity_search(query, k=k_summaries)

        relevant_chunks = []
        for summary in top_summaries:
            page_number = summary.metadata["page"]
            page_filter = lambda metadata: metadata["page"] == page_number
            page_chunks = self.detailed_vectorstore.similarity_search(
                query, k=k_chunks, filter=page_filter
            )
            relevant_chunks.extend(page_chunks)

        return relevant_chunks
