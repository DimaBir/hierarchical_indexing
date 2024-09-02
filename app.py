import asyncio
import streamlit as st
from core.vector_store_manager import VectorStoreManager
from core.hierarchical_retriever import HierarchicalRetriever


class QAApplication:
    """
    The QAApplication class manages the interaction between the user and the hierarchical Q&A system.
    It loads or creates vector stores and handles the retrieval of answers based on user queries.
    """

    def __init__(self):
        """
        Initializes the QAApplication by setting up the vector store manager and document path.
        """
        self.vector_store_manager = VectorStoreManager()
        self.path = "./data/intro_to_algo_book.pdf"
        self.summary_store = None
        self.detailed_store = None

    async def initialize_stores(self):
        """
        Initializes the vector stores by loading them from the file system or creating them if they don't exist.
        """
        self.summary_store, self.detailed_store = (
            await self.vector_store_manager.get_or_create_vector_stores(self.path)
        )

    def get_answer(self, query):
        """
        Retrieves an answer to the user's query using hierarchical retrieval.

        Args:
            query (str): The user's query.

        Returns:
            List[Document]: A list of relevant detailed document chunks.
        """
        retriever = HierarchicalRetriever(self.summary_store, self.detailed_store)
        results = retriever.retrieve(query)
        return results


def display_response(results, query):
    """
    Displays the response from the hierarchical retrieval system in a user-friendly format.

    Args:
        results (List[Document]): The list of retrieved document chunks.
        query (str): The user's query.
    """
    st.write(f"Results for: **{query}**")
    for chunk in results:
        st.write(f"**Page {chunk.metadata['page']}**")
        st.write(chunk.page_content[:300])
        st.write("---")


# Initialize the app
app = QAApplication()

# Streamlit UI
st.title("Hierarchical Q&A System")
st.write("Ask a question and retrieve relevant information from the document.")

# Initialize session state to store chat history
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

query = st.text_input(
    "Enter your query:", placeholder="e.g., How does the Quicksort algorithm work?"
)

if st.button("Get Answer") and query:
    with st.spinner("Retrieving the answer using hierarchical indexing..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(app.initialize_stores())

        results = app.get_answer(query)

        # Store the response and query in the session state
        st.session_state["user_prompt_history"].append(query)
        st.session_state["chat_answers_history"].append(results)
        st.session_state["chat_history"].append(("human", query))
        st.session_state["chat_history"].append(("ai", results))

        display_response(results, query)


if st.session_state["chat_answers_history"]:
    for i, (user_query, generated_response) in enumerate(
        zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
        )
    ):
        # Display user query
        st.write(f"**User:** {user_query}")

        # Display AI response
        display_response(generated_response, user_query)
