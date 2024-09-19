## Description: 
This project demonstrates a powerful Retrieval-Augmented Generation (RAG) system designed to create interactive resumes through a Q&A conversation interface. Using LangChain and Chroma, the tool integrates past chat history, allows retrieval of relevant data from documents, and provides users with a seamless experience for generating dynamic content from their resume PDFs.

## Features:

**Retrieval-Augmented Generation (RAG):** Combines information retrieval and generative models to answer user queries using structured PDF documents.
**Chat History Management:** Maintains conversation context, allowing the system to remember past questions and responses.
**Streamlit Interface:** Presents an easy-to-use web application interface for real-time interactions with the resume data.
**Chroma Integration:** Enables efficient document search and retrieval using embeddings stored in a vector database.
**Chain-based Execution:** Uses LangChain's capabilities to create customizable chains for retrieval and generation processes.

## Technologies:

* Python
* LangChain
* Chroma
* Streamlit
* Generative AI models

## Usage:

* Install dependencies using the provided requirements.txt.
* Launch the Streamlit app with streamlit run InteractiveResume.py.
* Upload your resume PDF and start an interactive Q&A session to generate customized resumes or answers based on your document content.
