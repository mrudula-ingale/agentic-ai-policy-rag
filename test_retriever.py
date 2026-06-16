from langchain_core.documents import Document

from policy_agent.vector_store import retrieve_documents


def print_retrieved_documents(documents: list[Document]) -> None:
    """
    Print retrieved documents in a readable format.

    Args:
        documents: Retrieved document chunks.
    """
    print("\n--- Retrieved Documents ---")

    for index, document in enumerate(documents, start=1):
        source_name = document.metadata.get("source_name", "unknown")
        file_name = document.metadata.get("file_name", "unknown")
        page = document.metadata.get("page", "unknown")

        print(f"\nDocument {index}")
        print(f"Source: {source_name}")
        print(f"File: {file_name}")
        print(f"Page: {page}")
        print("Text preview:")
        print(document.page_content[:500])


def main() -> None:
    """
    Test whether retrieval works from the existing vector store.
    """
    question = "What are high-risk AI systems under the EU AI Act?"

    documents = retrieve_documents(
        question=question,
        k=3,
        source_filter="eu_ai_act",
    )

    print_retrieved_documents(documents)


if __name__ == "__main__":
    main()
