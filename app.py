from __future__ import annotations

import logging
import os

import retriver
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="AI Research Assistant", layout="wide")

if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

st.title("AI Research Assistant 🤖")

load_dotenv()
ENV_GROQ_API = os.getenv("GROQ_API", "")
SECRETS_GROQ_API = st.secrets.get("GROQ_API", "")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")


@st.cache_resource
def load_embedding() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm(api_key: str, model_name: str) -> ChatGroq:
    return ChatGroq(model=model_name, api_key=api_key)


with st.sidebar:
    st.subheader("Settings")
    max_pages = st.slider("Search pages", min_value=1, max_value=5, value=1)
    user_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        placeholder="gsk_...",
        help="You can provide the key here or via GROQ_API env / Streamlit secrets.",
    )
    if user_api_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = user_api_key

active_api_key = st.session_state.groq_api_key or ENV_GROQ_API or SECRETS_GROQ_API
if not active_api_key:
    st.info(
        "Add your Groq API key in the sidebar, or set GROQ_API in environment/Streamlit secrets."
    )
    st.stop()

with st.form(key="search_form"):
    query = st.text_input("Search Topic")
    search_button = st.form_submit_button(label="Search")

llm = load_llm(active_api_key, DEFAULT_MODEL)
embedding = load_embedding()

if search_button and query:
    st.session_state.last_query = query
    with st.spinner("Searching for papers..."):
        try:
            st.session_state.search_results = retriver.retrive_paper(query, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Paper search failed")
            st.error(f"Search failed: {exc}")
            st.session_state.search_results = []
    st.session_state.selected_paper = None
    st.session_state.vector_store = None
    st.session_state.chat_history = []

if st.session_state.search_results:
    st.subheader(
        f"Found {len(st.session_state.search_results)} papers for '{st.session_state.last_query}'"
    )
    for i, paper in enumerate(st.session_state.search_results):
        st.write(f"{paper['title']}** - by {paper['authors']}")
        with st.expander("View abstract"):
            st.write(paper["abstract"])
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Select this paper", key=f"select_{i}"):
                st.session_state.selected_paper = paper
                with st.spinner("Preparing papers..."):
                    try:
                        st.session_state.vector_store = retriver.create_vectorstore(
                            paper["link"], embedding
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("Vector store creation failed")
                        st.error(f"Failed to load PDF: {exc}")
                        st.session_state.selected_paper = None
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            st.link_button("Open PDF", paper["link"])

if st.session_state.selected_paper and st.session_state.vector_store:
    st.header(f"Chat with: {st.session_state.selected_paper['title']}")

    for msg in st.session_state.chat_history:
        role = "user" if msg["is_user"] else "assistant"
        with st.chat_message(role):
            st.markdown(msg["message"])
    user_question = st.chat_input("Ask a question about this paper...")
    if user_question:
        st.session_state.chat_history.append({"is_user": True, "message": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = retriver.get_answer(
                        llm,
                        st.session_state.vector_store,
                        user_question,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("QA generation failed")
                    response = f"I couldn't process that question right now: {exc}"
                st.markdown(response)
        st.session_state.chat_history.append({"is_user": False, "message": response})
