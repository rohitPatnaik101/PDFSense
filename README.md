# PDFSense – Makes Sense of Your PDFs with AI

PDFSense is an AI-powered chatbot that processes uploaded PDF files and answers questions about their content using a Retrieval-Augmented Generation (RAG) pipeline. It leverages the Together API for efficient language model inference, making it lightweight and suitable for systems without GPUs.

## Features
- **PDF Processing**: Extracts text from uploaded PDFs (supports large files, 1000+ pages) using PyMuPDF with page-by-page processing.
- **Dynamic Chunking**: Splits PDF content into manageable chunks (1024 characters with 200-character overlap) for efficient retrieval using LangChain.
- **Vector Storage**: Uses FAISS with disk-based indexing for persistent storage of document embeddings, enabling reuse across sessions.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with a powerful language model to provide context-aware answers.
- **Together API Integration**: Offloads language model inference to Together AI’s servers, supporting models like `mistralai/Mixtral-8x7B-Instruct-v0.1` without local resource demands.
- **Streamlit UI**: Provides a simple, interactive web interface for uploading PDFs and querying their content.
- **Error Handling**: Includes basic error handling for file uploads and processing to improve user experience.

## Project Structure