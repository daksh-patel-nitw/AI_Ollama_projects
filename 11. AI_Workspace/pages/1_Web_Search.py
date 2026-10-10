import streamlit as st

from core.llm import get_llm
from core.prompts import SEARCH_SYSTEM_PROMPT
from core.search import web_search
from utils.ui import page_header, require_provider

st.set_page_config(page_title="Web Search Agent", page_icon="🌐")
require_provider()

page_header(
    "🌐 Web Search Agent",
    "Ask about anything current — answers are grounded in live DuckDuckGo "
    "results (free, no API key).",
)

if "search_history" not in st.session_state:
    st.session_state.search_history = []

for role, text in st.session_state.search_history:
    with st.chat_message(role):
        st.markdown(text)

query = st.chat_input("Ask about anything happening right now...")

if query:
    st.session_state.search_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                from langgraph.prebuilt import create_react_agent

                agent = create_react_agent(
                    get_llm(), tools=[web_search], prompt=SEARCH_SYSTEM_PROMPT
                )
                result = agent.invoke({"messages": [{"role": "user", "content": query}]})
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"Error: {e}"
            st.markdown(answer)

    st.session_state.search_history.append(("assistant", answer))
