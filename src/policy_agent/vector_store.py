from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from policy_agent.config import (
    COLLECTION_NAME,
    DEFAULT_RETRIEVAL_K,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
)
from policy_agent.document_loader import load_and_split_documents


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns: HuggingFace embedding model used by Chroma.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store() -> Chroma:
    """
    Create and persist a Chroma vector store from policy documents.

    Returns:
        The created Chroma vector store.
    """
    chunks = load_and_split_documents()
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    print(f"Created vector store with {len(chunks)} chunks.")
    print(f"Vector store path: {VECTOR_STORE_DIR}")

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing Chroma vector store from disk.

    Returns:
        A Chroma vector store object.
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    return vector_store


def retrieve_documents(
    question: str,
    k: int = DEFAULT_RETRIEVAL_K,
    source_filter: str | None = None,
) -> list[Document]:
    """
    Retrieve relevant document chunks for a user question.

    Args:
        question: User's natural-language question.
        k: Number of chunks to retrieve.
        source_filter: Optional source filter, e.g. 'eu_ai_act'.

    Returns:
        Relevant document chunks.
    """
    vector_store = load_vector_store()

    if source_filter is not None:
        return vector_store.similarity_search(
            query=question,
            k=k,
            filter={"source_name": source_filter},
        )

    return vector_store.similarity_search(query=question, k=k)
