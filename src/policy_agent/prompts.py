from policy_agent.state import RetrievedDocument


def format_retrieved_context(documents: list[RetrievedDocument]) -> str:
    """
    Convert retrieved documents into a single context string for the LLM.

    Args:
        documents: Retrieved evidence chunks.

    Returns:
        Formatted context string with source labels.
    """
    context_blocks: list[str] = []

    for index, document in enumerate(documents, start=1):
        source_label = (
            f"[Source {index}: {document['file_name']}, page {document['page']}]"
        )

        context_blocks.append(f"{source_label}\n{document['content']}")

    return "\n\n---\n\n".join(context_blocks)


def build_answer_prompt(
    question: str,
    question_type: str,
    selected_sources: list[str],
    retrieved_documents: list[RetrievedDocument],
) -> str:
    """
    Build the prompt used by the answer generation node.

    This prompt forces the model to:
    - use only retrieved context
    - cite sources
    - state uncertainty
    - avoid legal-advice wording
    """
    context = format_retrieved_context(retrieved_documents)
    selected_sources_text = ", ".join(selected_sources)

    return f"""
You are an AI policy compliance assistant.

Answer the user's question using ONLY the retrieved policy context.

User question:
{question}

Question type:
{question_type}

Selected policy sources:
{selected_sources_text}

Retrieved policy context:
{context}

Rules:
1. Use only the retrieved context.
2. Do not invent legal rules, article numbers, obligations, or examples.
3. If the context is insufficient, say so clearly.
4. Cite claims using [Source 1], [Source 2], etc.
5. Avoid strong legal conclusions unless directly supported.
6. End with: "This is an informational summary, not legal advice."
7. Do not say a use case is "prohibited" unless the retrieved context explicitly states that this exact use case is prohibited.
8. If the context places a use case under employment, education, biometric, or safety categories, describe it as "regulated" or "potentially high-risk", not automatically prohibited.
9. Prefer cautious wording: "may be", "appears to", "based on the retrieved context".
10. Do not mention article numbers unless the retrieved context clearly shows the article heading and the claim comes from that same source. If the source is an annex/list/category without a visible article number, refer to it as "the retrieved EU AI Act context" instead of naming an article.
11. Never turn the phrase "AI systems intended to be used for..." into "AI systems are prohibited". A list of intended uses is category evidence, not prohibition evidence, unless the same source explicitly says those uses are prohibited.
12. For recruitment, hiring, job applicants, or worker management, say the retrieved EU AI Act context lists these systems under employment-related AI uses. Do not say they are banned.

Return:

Answer:
...

Relevant evidence:
- ...

Risk or compliance assessment:
...

Limitations:
...
""".strip()


def build_validation_prompt(
    question: str,
    answer: str,
    retrieved_documents: list[RetrievedDocument],
) -> str:
    """
    Build prompt for LLM-based groundedness validation.

    The validator checks whether the generated answer is supported
    by the retrieved evidence.
    """
    context = format_retrieved_context(retrieved_documents)

    return f"""
You are a strict RAG answer validator.

Your task is to check whether the answer is grounded in the retrieved context.

User question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Validation criteria:
1. Major claims must be supported by the retrieved context.
2. Source citations must refer to relevant retrieved sources.
3. The answer must not invent legal rules, article numbers, examples, or obligations.
4. If the answer makes unsupported claims, mark it as failed.
5. If the answer is mostly supported but needs softer wording, mark it as failed and explain what to revise.
6. Fail the answer if it says a use case is prohibited when the retrieved context only shows that it is high-risk, regulated, or listed in an annex/category.
7. Fail the answer if a citation does not directly support the sentence where it is used
8. Fail the answer if the answer makes broad claims about fairness, bias, transparency, or accountability without retrieved evidence.
9. If the answer needs softer wording, mark it as failed.
10. Fail the answer if it assigns the wrong article number to a claim, especially if Article 4 is used for a topic other than AI literacy.
11. Fail the answer if it treats "AI systems intended to be used for..." as a prohibition. That wording only describes intended-use categories unless the same source explicitly says the use is prohibited.
12. For job screening questions, pass only answers that distinguish between listed/regulated employment AI uses and explicitly prohibited AI practices.
Return exactly in this format:

STATUS: passed or failed

FEEDBACK:
short explanation of what is supported or unsupported
""".strip()


def build_revision_prompt(
    question: str,
    original_answer: str,
    validation_feedback: str,
    retrieved_documents: list[RetrievedDocument],
) -> str:
    """
    Build prompt to revise an answer after validation failure.
    """
    context = format_retrieved_context(retrieved_documents)

    return f"""
You are an AI policy compliance assistant.

The previous answer failed groundedness validation.

User question:
{question}

Retrieved context:
{context}

Original answer:
{original_answer}

Validator feedback:
{validation_feedback}

Revise the answer so that:
1. Every major claim is supported by the retrieved context.
2. Unsupported claims are removed or clearly marked as uncertainty.
3. Citations use [Source 1], [Source 2], etc.
4. The answer is cautious and does not overstate legal conclusions.
5. It ends with: "This is an informational summary, not legal advice."
6. Do not mention article numbers unless the retrieved context clearly shows the article heading and the claim comes from that same source.
7. If the source is an annex/list/category without a visible article number, refer to it as "the retrieved EU AI Act context" instead of naming an article.
8. Do not say a use case is "prohibited" unless the retrieved context explicitly states that this exact use case is prohibited.
9. Never turn the phrase "AI systems intended to be used for..." into "AI systems are prohibited". A list of intended uses is category evidence, not prohibition evidence.
10. For recruitment, hiring, job applicants, or worker management, say the retrieved EU AI Act context lists these systems under employment-related AI uses. Do not say they are banned.

Return:

Answer:
...

Relevant evidence:
- ...

Risk or compliance assessment:
...

Limitations:
...
""".strip()
