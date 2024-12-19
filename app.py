import streamlit as st
import os
import dotenv
import uuid
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    load_doc_to_db, 
    stream_llm_response,
    stream_llm_rag_response, 
)

dotenv.load_dotenv()

MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini"]

st.set_page_config(
    page_title="Hierarchical Document Chat", 
    page_icon="📚", 
    layout="centered"
)

st.html("""<h2 style="text-align: center;">📚🔍 <i> Hierarchical Document-Based Chat </i> 🤖💬</h2>""")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# Sidebar for document uploads and category selection
with st.sidebar:
    st.header("Upload Documents by Category")
    category = st.selectbox(
        "Select Document Category", 
        ["A (High Priority)", "B (Medium Priority)"], 
        help="Choose the category of the document before uploading."
    )
    st.file_uploader(
        "📄 Upload Document", 
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        on_change=lambda: load_doc_to_db(category),
        key="rag_docs",
    )

    st.selectbox(
        "🤖 Select a Model", 
        options=MODELS,
        key="model",
    )

    openai_api_key = st.text_input(
        "🔑 Enter your OpenAI API Key", 
        value=os.getenv("OPENAI_API_KEY", ""), 
        type="password",
        key="openai_api_key",
    )

    if not openai_api_key:
        st.warning("Please provide your OpenAI API key to continue.")
        st.stop()

    st.toggle("Use RAG", value=True, key="use_rag")
    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

# Initialize OpenAI LLM
llm_stream = ChatOpenAI(
    api_key=openai_api_key,
    model_name=st.session_state.model.split("/")[-1],
    temperature=0.3,
    streaming=True,
)

# Main chat loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
