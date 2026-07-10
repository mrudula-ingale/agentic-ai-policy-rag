# Agentic AI Policy RAG Assistant

An Agentic Retrieval-Augmented Generation (RAG) system that answers questions about AI regulations using the **EU AI Act**, **GDPR**, and **NIST AI Risk Management Framework (AI RMF)**.

Unlike traditional RAG pipelines that follow a fixed retrieve-and-generate workflow, this project uses **LangGraph** to orchestrate multiple AI agents responsible for:

* Question classification
* Policy source routing
* Document retrieval
* Grounded answer generation
* Answer validation
* Automatic answer revision

The system retrieves evidence from policy documents, generates answers using an LLM, validates whether the response is supported by the retrieved evidence, and automatically revises unsupported answers.

---

# Live Demo

Try the deployed Streamlit app:

[Agentic AI Policy RAG Assistant](https://agentic-ai-policy-rag-2ywqecwxmuv55qikdtdcsh.streamlit.app/)

Note: The demo runs on free-tier hosting and free/limited LLM inference, so it may sleep when inactive or temporarily hit rate limits.

---

# Deployment

This project is deployed as a Streamlit web application connected to the GitHub repository. New commits pushed to the deployed branch trigger an automatic rebuild and app update on Streamlit Community Cloud.

Deployment stack:

* Streamlit Community Cloud for the hosted UI
* GitHub-based deployment workflow
* ChromaDB persisted vector store for retrieval
* Hugging Face sentence-transformer embeddings
* Groq-hosted LLM inference
* LangGraph for multi-step agent orchestration

Secrets such as `GROQ_API_KEY` are configured in Streamlit Cloud secrets and are not committed to the repository.

---

# Project Overview

AI regulation documents are long, complex, and difficult to search manually. This project builds an Agentic RAG assistant that can answer policy-related questions using retrieved evidence from regulatory documents.

This project demonstrates an **Agentic RAG architecture** that combines:

**Policy Documents → Vector Database → Multi-Agent Workflow → Grounded Answers**

The workflow routes questions to the most relevant policy sources, retrieves supporting evidence, generates answers, validates grounding, and revises responses when necessary.

The system follows this workflow:

User Question

     ↓

Question Classification

     ↓

Rule-Based Source Routing

     ↓

Document Retrieval

     ↓

Grounded Answer Generation

     ↓

Answer Validation

     ↓

If validation fails → Answer Revision

     ↓

Final Answer

---

# Features

## Agentic Workflow with LangGraph

The system is implemented as a LangGraph workflow with multiple nodes:

* Question classification
* Policy source routing
* Source-specific retrieval
* LLM answer generation
* Grounding validation
* Automatic answer revision
* Final response generation

Conditional graph execution enables answer revision when validation fails.

---

## Rule-Based Source-Aware Retrieval

The routing step selects relevant policy documents before retrieval.

* EU AI Act
* GDPR
* NIST AI RMF

Examples:

| Question Type                                 | Sources          |
| --------------------------------------------- | ---------------- |
| Employment AI/Job applicant screening         | EU AI Act + GDPR |
| Automated decision-making                     | GDPR             |
| Profiling & Privacy                           | GDPR             |
| Risk Management                               | NIST AI RMF      |
| High-Risk Systems                             | EU AI Act        |
| Biometric or facial recognition AI            | EU AI Act + GDPR |

This routing is currently rule-based and deterministic.

---

## Grounded Answer Generation

The assistant retrieves relevant document chunks from ChromaDB and passes them to the LLM as source context.

The answer prompt instructs the model to:

* Use only retrieved context
* Cite sources
* Avoid unsupported legal claims
* Avoid hallucinated article numbers
* Use cautious wording when evidence is incomplete

---

## Answer Validation and Revision

A validation agent checks whether the generated answer is supported by the retrieved evidence.

The validator checks for:

* Unsupported claims
* Incorrect citations
* Overconfident legal conclusions
* Hallucinated article numbers
* Claims not supported by retrieved sources

If validation fails, the workflow automatically routes to a revision step and regenerates a safer answer.

---

## Streamlit User Interface

The project includes a Streamlit UI for interactive use.

The UI shows:

* Final answer
* Question type
* Selected policy sources
* Validation status
* Validator feedback
* Retrieved evidence chunks

---

## Evaluation Framework

IThe project includes an evaluation pipeline for routing performance.

Current result:

Routing Accuracy: 90.00%

Evaluation includes:

* Benchmark policy questions
* Expected source labels
* Actual selected sources
* Routing correctness
* Validation status tracking

---

# Why This Project Is Interesting

This project extends a basic RAG pipeline into an agentic workflow.

Traditional RAG:

Question → Retriever → LLM → Answer

This project:

Question

   ↓

Classifier

   ↓

Source Router

   ↓

Retriever

   ↓

Answer Generator

   ↓

Validator

   ↓

Revision if needed

   ↓

Final Answer

It demonstrates:

* LangGraph workflow orchestration
* Stateful multi-step execution
* Rule-based source-aware routing
* Retrieval-Augmented Generation
* LLM-based groundedness validation
* Automatic answer revision
* Evaluation of routing performance

---

# Tech Stack

## Agentic AI and RAG

* LangGraph
* LangChain

## Retrieval

* ChromaDB
* Hugging Face Embeddings
* Sentence Transformers (MiniLM)

## LLM

* Groq
* Llama 3.1 8B Instant

## Data Processing

* PyPDFLoader
* Recursive Character Text Splitter

## UI and Evaluation

* Streamlit
* Pandas

## Environment Management

* uv

---

# Project Structure

```text
agentic-ai-policy-rag/
│
├── app/
│   └── streamlit_app.py
|
├── data/
│   ├── raw/ai_policy/
│   │   ├── eu_ai_act.pdf
│   │   ├── gdpr.pdf
│   │   └── nist_ai_rmf.pdf
│   │
│   └── vector_store/
│
├── evaluation/
│   ├── evaluation_questions.csv
│   ├── evaluation_results.csv
│   ├── evaluator.py
│   └── config.py
│
├── src/policy_agent/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── prompts.py
│   ├── graph.py
│   └── build_vector_store.py
│
├── .streamlit/
│   └── config.toml
|
├── main.py
├── .env.example
├── pyproject.toml
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd agentic-ai-policy-rag
```

Install dependencies:

```bash
uv sync
```

Activate environment:

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

---

# Build the Vector Store

Place policy PDFs in:

```text
data/raw/ai_policy/
```

Build ChromaDB:

```powershell
$env:PYTHONPATH="src"
uv run python src/policy_agent/build_vector_store.py
```

---

# Run the Terminal App

```powershell
$env:PYTHONPATH="src"
uv run python main.py
```

Example:

```text
Ask an AI policy question:
Can AI be used to screen job applicants?
```

---

# Run the Streamlit App

```powershell
$env:PYTHONPATH="src;."
uv run streamlit run app/streamlit_app.py
```

Open:

http://localhost:8501

Example questions:

* Can AI be used to screen job applicants?
* What rights do individuals have regarding automated decision-making?
* How should organizations manage AI bias?
* Is facial recognition allowed in public spaces?

---

# Run Evaluation

Evaluate routing performance:

```powershell
$env:PYTHONPATH="src;."
uv run python evaluation/evaluator.py
```

Example output:

```text
========== Evaluation ==========

Routing Accuracy: 90.00%

Results saved to:
evaluation/evaluation_results.csv
```

---

# Example Question

Question:

```text
Can AI be used to screen job applicants?
```

Expected workflow:

1. Classified as compliance question
2. Routed to EU AI Act and GDPR
3. Relevant policy evidence retrieved
4. LLM generates grounded answer
5. Validator checks grounding
6. Final answer returned

---

# Limitations

* Source routing is rule-based, not LLM-based.
* The validator uses an LLM-as-judge approach, which is useful but not perfect.
* Retrieval quality depends on chunk size, embeddings, and PDF text extraction quality
* The benchmark dataset is currently small.
* This project provides informational summaries only, not legal advice.

---

# Future Improvements

Potential extensions:

* LLM-based routing agent or hybrid routing
* Query rewriting before retrieval
* Hybrid search (semantic + keyword)
* Re-ranking models
* Larger evaluation benchmark
* Conversation memory
* Multi-turn policy discussions
* Docker deployment
* Retrieval quality evaluation
* Answer quality scoring

---

# Key Learning Outcomes

This project was built to explore concepts beyond traditional RAG, including:

* LangGraph fundamentals
* Agentic AI workflows
* RAG system design
* Source-aware retrieval
* State management in LangGraph
* LLM answer grounding
* Answer validation and revision
* Evaluation of AI system behavior

---

# Author

**Mrudula Ankush Ingale**

M.Sc. Computer Science

Interests:

* Artificial Intelligence
* Machine Learning
* Agentic AI
* Retrieval-Augmented Generation
* Explainable AI
* Data Science
