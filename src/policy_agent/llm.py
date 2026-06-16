from dotenv import load_dotenv
from langchain_groq import ChatGroq

from policy_agent.config import GROQ_MODEL_NAME, LLM_TEMPERATURE


def get_llm() -> ChatGroq:
    """
    Create and return the Groq chat model.

    load_dotenv():
    - Reads variables from the .env file.
    - This allows ChatGroq to find GROQ_API_KEY.

    ChatGroq:
    - LangChain wrapper for Groq chat models.
    - We set temperature=0 for more consistent factual answers.
    """
    load_dotenv()

    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )
