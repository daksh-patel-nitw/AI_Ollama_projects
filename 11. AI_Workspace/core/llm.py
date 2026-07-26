from langchain_community.chat_models import ChatOllama

from core.config import (
    LLM_NAME,
    TEMPERATURE
)

def get_llm():

    return ChatOllama(
        model=LLM_NAME,
        temperature=TEMPERATURE
    )