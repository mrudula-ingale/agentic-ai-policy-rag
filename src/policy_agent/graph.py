from langgraph.graph import END, START, StateGraph

from policy_agent.config import RETRIEVAL_K_PER_SOURCE
from policy_agent.llm import get_llm
from policy_agent.prompts import (
    build_answer_prompt,
    build_revision_prompt,
    build_validation_prompt,
)
from policy_agent.state import PolicyAgentState, RetrievedDocument, SourceName
from policy_agent.vector_store import retrieve_documents


# This is a LangGraph node.
#
# A node is just a normal Python function.
# It receives the current graph state as input.
# It returns a dictionary containing only the fields it wants to update.
def classify_question(state: PolicyAgentState) -> str:
    """
    Classify the user's question into a simple question type.
    Args:
        state: Current graph state.

    Returns:
        Partial state update containing question_type.
    """
    question = state["question"].lower()

    if any(word in question for word in ["what is", "define", "meaning"]):
        question_type = "definition"
    elif any(
        word in question
        for word in [
            "allowed",
            "legal",
            "comply",
            "requirement",
            "obligation",
            "use",
            "screen",
            "job",
            "applicant",
            "employment",
            "hiring",
            "recruitment",
        ]
    ):
        question_type = "compliance"
    elif any(
        word in question
        for word in [
            "risk",
            "danger",
            "high-risk",
            "prohibited",
            "ban",
            "bias",
            "fairness",
            "monitoring",
            "manage",
            "management",
            "governance",
            "validation",
        ]
    ):
        question_type = "risk"
    else:
        question_type = "unknown"

    # We do not return the full state.
    # We only return the part of the state that changed.
    return {"question_type": question_type}


def route_documents(state: PolicyAgentState) -> dict:
    """
    Select which policy documents are relevant for the question.

    This is a rule-based router.

    Why rule-based first?
    - Easy to understand
    - Easy to debug
    - Deterministic
    - Good baseline before adding LLM-based routing

    Later we can improve this with a hybrid router:
    rules first + LLM fallback.
    """
    question = state["question"].lower()
    selected_sources: list[SourceName] = []

    # EU AI Act is relevant for AI system classification,
    # prohibited systems, high-risk systems, biometric AI, and employment AI.
    eu_ai_act_keywords = [
        "high-risk",
        "prohibited",
        "ban",
        "biometric",
        "facial recognition",
        "job",
        "applicant",
        "employment",
        "hiring",
        "recruitment",
        "education",
        "law enforcement",
        "credit scoring",
    ]

    # GDPR is relevant whenever personal data, privacy,
    # consent, or data subject rights are involved.
    gdpr_keywords = [
        "personal data",
        "privacy",
        "consent",
        "data subject",
        "processing",
        "automated decision",
        "profiling",
        "employee data",
        "candidate data",
        "job",
        "applicant",
        "employment",
        "hiring",
        "recruitment",
        "biometric",
        "facial recognition",
    ]

    # NIST AI RMF is relevant for AI risk management,
    # governance, monitoring, evaluation, and lifecycle risks.
    nist_keywords = [
        "risk management",
        "governance",
        "monitoring",
        "evaluation",
        "trustworthy",
        "bias",
        "fairness",
        "transparency",
        "accountability",
        "lifecycle",
        "test",
        "validation",
        "monitor",
        "monitored",
        "deployment",
        "deployed",
        "evaluate",
        "evaluating",
        "ai risks",
        "risk assessment",
    ]

    if any(keyword in question for keyword in eu_ai_act_keywords):
        selected_sources.append("eu_ai_act")

    if any(keyword in question for keyword in gdpr_keywords):
        selected_sources.append("gdpr")

    if any(keyword in question for keyword in nist_keywords):
        selected_sources.append("nist_ai_rmf")

    # Fallback:
    # If no rule matches, search all documents.
    if not selected_sources:
        selected_sources = ["eu_ai_act", "gdpr", "nist_ai_rmf"]

    return {"selected_sources": selected_sources}


def retrieve_policy_documents(state: PolicyAgentState) -> dict:
    """
    Retrieve relevant chunks from the selected policy documents.

    Instead of searching all PDFs together, this node searches each routed
    source separately. This improves source coverage.

    Args:
        state: Current graph state.

    Returns:
        Partial state update containing retrieved_documents.
    """
    question = state["question"]

    selected_sources = state["selected_sources"]

    retrieved_documents: list[RetrievedDocument] = []

    for source_name in selected_sources:
        raw_documents = retrieve_documents(
            question=question,
            k=RETRIEVAL_K_PER_SOURCE,
            source_filter=source_name,
        )

        for document in raw_documents:
            metadata = document.metadata

            retrieved_documents.append(
                {
                    "source_name": metadata.get("source_name", "unknown"),
                    "file_name": metadata.get("file_name", "unknown"),
                    "page": metadata.get("page", "unknown"),
                    "content": document.page_content,
                }
            )

    return {"retrieved_documents": retrieved_documents}


def generate_grounded_answer(state: PolicyAgentState) -> dict:
    """
    Generate a grounded answer using Groq LLM and retrieved policy context.

    This node is the first LLM-powered node in the graph.
    """
    retrieved_documents = state["retrieved_documents"]

    if not retrieved_documents:
        return {
            "answer": "",
            "llm_error": "No retrieved documents available for answer generation.",
        }

    prompt = build_answer_prompt(
        question=state["question"],
        question_type=state["question_type"],
        selected_sources=state["selected_sources"],
        retrieved_documents=retrieved_documents,
    )

    try:
        llm = get_llm()

        # invoke() sends the prompt to the LLM and returns a message object.
        llm_response = llm.invoke(prompt)

        return {
            "answer": llm_response.content,
            "llm_error": "",
        }

    except Exception as error:
        return {
            "answer": "",
            "llm_error": str(error),
        }


def validate_answer_grounding(state: PolicyAgentState) -> dict:
    """
    Validate whether the generated answer is supported by retrieved context.

    This is an LLM-as-judge validator.
    It is not perfect, but it is useful for detecting many unsupported claims.
    """
    if state["llm_error"]:
        return {
            "validation_status": "failed",
            "validation_feedback": "Answer generation failed, so validation could not run.",
        }

    prompt = build_validation_prompt(
        question=state["question"],
        answer=state["answer"],
        retrieved_documents=state["retrieved_documents"],
    )

    try:
        llm = get_llm()
        llm_response = llm.invoke(prompt)
        validation_text = llm_response.content.strip()

        # Simple parser for the validator output.
        # We expect the validator to start with STATUS: passed or STATUS: failed.
        lower_text = validation_text.lower()

        if "status: passed" in lower_text:
            status = "passed"
        else:
            status = "failed"

        return {
            "validation_status": status,
            "validation_feedback": validation_text,
        }

    except Exception as error:
        return {
            "validation_status": "failed",
            "validation_feedback": f"Validation failed due to error: {error}",
        }


def should_revise_answer(state: PolicyAgentState) -> str:
    """
    Conditional edge function.

    LangGraph uses this return value to decide the next node.

    Returns:
        'revise' if validation failed.
        'final' if validation passed.
    """
    if state["validation_status"] == "passed":
        return "final"

    return "revise"


def revise_answer(state: PolicyAgentState) -> dict:
    """
    Revise the generated answer using validator feedback.
    """
    prompt = build_revision_prompt(
        question=state["question"],
        original_answer=state["answer"],
        validation_feedback=state["validation_feedback"],
        retrieved_documents=state["retrieved_documents"],
    )

    try:
        llm = get_llm()
        llm_response = llm.invoke(prompt)

        return {
            "revised_answer": llm_response.content,
            "llm_error": "",
        }

    except Exception as error:
        return {
            "revised_answer": "",
            "llm_error": str(error),
        }


def build_final_response(state: PolicyAgentState) -> dict:
    """
    Build the final response shown to the user.

    This node separates internal answer generation from final presentation.
    Later, validation results will also be added here.
    """
    selected_sources = ", ".join(state["selected_sources"])

    final_answer = state["revised_answer"] or state["answer"]

    source_summary = sorted(
        {
            f"{document['file_name']} page {document['page']}"
            for document in state["retrieved_documents"]
        }
    )
    source_lines = "\n".join(f"- {source}" for source in source_summary)

    response = f"{final_answer}\n\nRetrieved source pages:\n{source_lines}"

    return {"response": response}


def build_graph():
    """
    Build and compile the LangGraph workflow.
     Flow:
    classify → route → retrieve → answer → validate
    then:
    - if validation passed → final
    - if validation failed → revise → final
    """
    graph = StateGraph(PolicyAgentState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("route_documents", route_documents)
    graph.add_node("retrieve_policy_documents", retrieve_policy_documents)
    graph.add_node("generate_grounded_answer", generate_grounded_answer)
    graph.add_node("validate_answer_grounding", validate_answer_grounding)
    graph.add_node("revise_answer", revise_answer)
    graph.add_node("build_final_response", build_final_response)

    graph.add_edge(START, "classify_question")
    graph.add_edge("classify_question", "route_documents")
    graph.add_edge("route_documents", "retrieve_policy_documents")
    graph.add_edge("retrieve_policy_documents", "generate_grounded_answer")
    graph.add_edge("generate_grounded_answer", "validate_answer_grounding")

    # Conditional routing:
    # should_revise_answer returns either 'final' or 'revise'.
    graph.add_conditional_edges(
        "validate_answer_grounding",
        should_revise_answer,
        {
            "final": "build_final_response",
            "revise": "revise_answer",
        },
    )

    graph.add_edge("revise_answer", "build_final_response")
    graph.add_edge("build_final_response", END)

    return graph.compile()
