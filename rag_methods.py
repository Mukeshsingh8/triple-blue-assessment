import os
from time import time
import uuid
import streamlit as st
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage

DB_DOCS_LIMIT = 10

# Function to stream the response of the LLM
def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})

# --- Indexing Phase ---
def load_doc_to_db(category: str):
    """
    Loads documents into the vector database and tags them with their respective category.
    """
    print(f"Starting to load documents into category: {category}")
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            os.makedirs("source_files", exist_ok=True)
            file_path = f"./source_files/{doc_file.name}"
            with open(file_path, "wb") as file:
                file.write(doc_file.read())

            try:
                if doc_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                elif doc_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif doc_file.type in ["text/plain", "text/markdown"]:
                    loader = TextLoader(file_path)
                else:
                    print(f"Unsupported file type: {doc_file.type}")
                    continue

                docs.extend(loader.load())
                print(f"Document {doc_file.name} loaded successfully!")
            except Exception as e:
                print(f"Error loading document {doc_file.name}: {e}")
            finally:
                os.remove(file_path)

        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
            document_chunks = text_splitter.split_documents(docs)

            # Add category, source, and unique ID metadata to each chunk
            for chunk in document_chunks:
                chunk.metadata["category"] = category
                chunk.metadata["source"] = doc_file.name
                chunk.metadata["uuid"] = str(uuid.uuid4())  # Add a unique ID
                print(f"Chunk Metadata Assigned: {chunk.metadata}")

            print(f"Document chunks prepared with category: {category}")
            initialize_vector_db(document_chunks)


def initialize_vector_db(docs):
    """
    Initializes the vector database and stores the documents with embeddings.
    """
    print(f"Initializing vector database with {len(docs)} documents...")
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)

    if "vector_db" not in st.session_state:
        # Create a new vector database if not already initialized
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
        )
        st.session_state.vector_db = vector_db
    else:
        # Add new documents to the existing database without overwriting
        st.session_state.vector_db.add_documents(docs)
        print(f"Appended {len(docs)} documents to the existing vector database.")

    print("Vector database initialized successfully!")


# --- Retrieval Augmented Generation (RAG) Phase ---
def get_prioritized_documents(query, retriever):
    """
    Retrieves documents with prioritization for Category A.
    Falls back to Category B only if Category A has no relevant documents.
    """
    print(f"Querying documents for: {query}")
    print("Applying metadata filter for Category A: {'category': 'A (High Priority)'}")

    # Retrieve documents from Category A
    category_a_results = retriever.invoke(query, metadata_filter={"category": "A (High Priority)"})
    if category_a_results:
        print(f"Category A documents retrieved: {len(category_a_results)}")
        for doc in category_a_results:
            print(f"Category A Document Metadata: {doc.metadata}")
        return category_a_results

    print("No documents found in Category A. Applying metadata filter for Category B: {'category': 'B (Medium Priority)'}")
    # Retrieve documents from Category B
    category_b_results = retriever.invoke(query, metadata_filter={"category": "B (Medium Priority)"})
    print(f"Category B documents retrieved: {len(category_b_results)}")
    for doc in category_b_results:
        print(f"Category B Document Metadata: {doc.metadata}")
    return category_b_results



def stream_llm_rag_response(llm_stream, messages):
    """
    Streams LLM response while prioritizing Category A documents.
    """
    query = messages[-1].content
    retriever = st.session_state.vector_db.as_retriever()

    # Retrieve documents from both categories
    category_a_results = retriever.invoke(query, metadata_filter={"category": "A (High Priority)"})
    category_b_results = retriever.invoke(query, metadata_filter={"category": "B (Medium Priority)"})

    # Combine and prioritize documents
    ranked_docs = category_a_results if category_a_results else category_b_results

    # Log the source of documents with proper categorization
    category_a_count = sum(1 for doc in ranked_docs if doc.metadata.get("category") == "A (High Priority)")
    category_b_count = sum(1 for doc in ranked_docs if doc.metadata.get("category") == "B (Medium Priority)")

    if category_a_count > 0:
        print(f"Using {category_a_count} chunks from Category A:")
        for doc in ranked_docs:
            if doc.metadata.get("category") == "A (High Priority)":
                print(f"- Metadata: {doc.metadata}")

    if category_b_count > 0:
        print(f"Using {category_b_count} chunks from Category B:")
        for doc in ranked_docs:
            if doc.metadata.get("category") == "B (Medium Priority)":
                print(f"- Metadata: {doc.metadata}")

    # Combine content for context
    context = "\n".join([doc.page_content for doc in ranked_docs])
    print(f"Final context prepared with {len(ranked_docs)} documents.")

    # LLM prompt with prioritized context
    prompt = (
        f"Use the following context to answer the question. Always prioritize Category A over Category B.\n\n"
        f"{context}\n\nQuestion: {query}"
    )

    response_message = ""
    for chunk in llm_stream.stream([HumanMessage(content=prompt)]):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


