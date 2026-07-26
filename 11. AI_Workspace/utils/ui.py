import streamlit as st


def page_header(title, description):
    st.title(title)
    st.caption(description)
    st.divider()