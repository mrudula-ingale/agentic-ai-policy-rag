# evaluation/evaluator.py

import pandas as pd

from policy_agent.graph import build_graph
from evaluation.config import QUESTIONS_FILE, RESULTS_FILE


def normalize_sources(source_string: str) -> set[str]:
    """
    Convert:

    "eu_ai_act,gdpr"

    into

    {"eu_ai_act", "gdpr"}
    """
    return {source.strip() for source in source_string.split(",")}


def evaluate_routing() -> None:
    """
    Evaluate source-routing accuracy.
    """
    df = pd.read_csv(QUESTIONS_FILE)

    app = build_graph()

    results = []

    for _, row in df.iterrows():
        question = row["question"]

        expected_sources = normalize_sources(row["expected_sources"])

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

        actual_sources = set(result["selected_sources"])

        routing_correct = expected_sources == actual_sources

        results.append(
            {
                "question": question,
                "expected_sources": ",".join(sorted(expected_sources)),
                "actual_sources": ",".join(sorted(actual_sources)),
                "routing_correct": routing_correct,
                "validation_status": result["validation_status"],
            }
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    routing_accuracy = results_df["routing_correct"].mean() * 100

    print("\n========== Evaluation ==========")
    print(f"Routing Accuracy: {routing_accuracy:.2f}%")

    print(f"Results saved to:\n{RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_routing()
