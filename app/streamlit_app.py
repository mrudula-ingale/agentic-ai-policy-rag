import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from policy_agent.graph import build_graph


def run_agent(question: str) -> dict:
    """
    Run the LangGraph agent with the user's question
    """

    app = build_graph()

    return app.invoke(
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


def main() -> None:
    """
    Streamlit UI for the Agentic AI Policy Assistant.
    """
    st.set_page_config(
        page_title="Agentic AI Policy Assistant",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Agentic AI Policy Assistant")

    st.write("Ask questions about the EU AI Act. GDPR. and NIST AI RMF")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Can AI be used to screen job applicants?",
        height=120,
    )

    run_button = st.button("Run Agent")

    if run_button:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Running LangGraph workflow..."):
            result = run_agent(question)

        st.subheader("Final Answer")
        st.write(result["response"])

        st.subheader("Agent Workflow Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Question Type", result["question_type"])

        with col2:
            st.metric(
                "Selected Sources",
                len(result["selected_sources"]),
            )

        with col3:
            st.metric("Validation", result["validation_status"])

        st.write("**Selected Policy Sources:**")
        st.write(", ".join(result["selected_sources"]))

        with st.expander("Validator Feedback"):
            st.write(result["validation_feedback"])

        with st.expander("Retrieved Evidence"):
            for index, document in enumerate(result["retrieved_documents"], start=1):
                st.markdown(f"### Source {index}")
                st.write(f"**Document:** {document['file_name']}")
                st.write(f"**Page:** {document['page']}")
                st.write(f"**Source:** {document['source_name']}")
                st.text(document["content"][:1000])


if __name__ == "__main__":
    main()
