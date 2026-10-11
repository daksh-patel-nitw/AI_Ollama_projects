import streamlit as st

from core.chat_store import clear_history, save_message
from core.config import WEB_SEARCH_DB
from core.llm import describe_llm_error, get_llm, stream_agent_text
from core.prompts import SEARCH_SYSTEM_PROMPT
from core.search import web_search
from core.session import charge_session, get_session_id
from utils.ui import page_header, require_provider

st.set_page_config(page_title="Web Search Agent", page_icon="🌐")
require_provider()

CHAT_DB_NAME = "web_search_chat"
session_id = get_session_id()


def _clear_chat():
    clear_history(WEB_SEARCH_DB, CHAT_DB_NAME, session_id=session_id)
    st.session_state.search_history = []


page_header(
    "🌐 Web Search Agent",
    "Ask about anything current — answers are grounded in live DuckDuckGo "
    "results.",
    on_clear=_clear_chat,
)

if "search_history" not in st.session_state:
    st.session_state.search_history = []

for turn in st.session_state.search_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

query = st.chat_input("Ask about anything happening right now...")

if query and charge_session(len(query.encode())):
    st.session_state.search_history.append({"role": "user", "content": query})
    save_message(WEB_SEARCH_DB, CHAT_DB_NAME, "user", query, session_id=session_id)
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            from langchain.agents import create_agent

            agent = create_agent(
                get_llm(), tools=[web_search], system_prompt=SEARCH_SYSTEM_PROMPT
            )
            with st.spinner("Searching and thinking..."):
                answer = st.write_stream(stream_agent_text(agent, query)) or "(no answer)"
        except Exception as e:
            answer = describe_llm_error(e)
            st.markdown(answer)

    charge_session(len(answer.encode()))
    st.session_state.search_history.append({"role": "assistant", "content": answer})
    save_message(WEB_SEARCH_DB, CHAT_DB_NAME, "assistant", answer, session_id=session_id)

