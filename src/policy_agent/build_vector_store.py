from policy_agent.vector_store import create_vector_store


def main() -> None:
    """
    Build the Chroma vector store from the policy PDFs.
    """
    create_vector_store()


if __name__ == "__main__":
    main()
