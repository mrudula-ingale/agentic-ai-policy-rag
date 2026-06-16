from policy_agent.graph import build_graph


def print_retrieved_sources(result: dict) -> None:
    """
    Print retrieved document previews.

    Args:
        result: Final LangGraph state returned by app.invoke().
    """
    print("\n--- Retrieved Evidence Preview ---")

    for index, document in enumerate(result["retrieved_documents"], start=1):
        print(f"\nDocument {index}")
        print(f"Source: {document['source_name']}")
        print(f"File: {document['file_name']}")
        print(f"Page: {document['page']}")
        print("Preview:")
        print(document["content"][:400])


def main() -> None:
    # Build and compile the LangGraph workflow.
    app = build_graph()

    question = input("Ask an AI policy question: ")

    # app.invoke(...) runs the graph once.
    # We provide the initial state here.
    # The graph will update question_type and response as it moves through nodes.
    result = app.invoke(
        {
            "question": question,
            "question_type": "unknown",
            "selected_sources": [],
            "retrieved_documents": [],
            "answer": "",
            "revised_answer": "",
            "validation_status": "not_checked",
            "validation_feedback": "",
            "llm_error": "",
            "response": "",
        }
    )
    print("\n--- RESULT ---")
    print(f"{result['response']}")
    print_retrieved_sources(result)


if __name__ == "__main__":
    main()
