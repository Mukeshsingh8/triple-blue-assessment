# Hierarchical Document Chat System

This project is a **Document-Based Chat System** powered by OpenAI's GPT-4, where users can upload documents and ask questions related to the content. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve and prioritize documents, enhancing the quality of AI-generated responses based on the uploaded content.

The application is built with **Streamlit** for the frontend, **LangChain** for document handling and LLM interaction, and **Chroma** for document indexing and retrieval.

## Key Features
- Upload documents (PDF, DOCX, TXT) and categorize them into High and Medium Priority.
- Ask questions related to the uploaded documents.
- Retrieve prioritized documents and generate AI-based responses using OpenAI's GPT-4 model.
- **RAG**: Retrieval-Augmented Generation for contextual AI responses.

## Technologies Used
- **Python 3.11**: Python programming language.
- **Streamlit**: A framework for building interactive web applications.
- **LangChain**: A library for LLMs, document loading, and text splitting.
- **OpenAI GPT-4**: Language model for answering questions.
- **Chroma**: A vector database used for storing document embeddings.
- **dotenv**: To manage environment variables.

## Setup Instructions

### Prerequisites
Ensure that you have Python 3.11 installed and have access to the following:
- OpenAI API key (to interact with GPT-4).
- Python virtual environment set up.

