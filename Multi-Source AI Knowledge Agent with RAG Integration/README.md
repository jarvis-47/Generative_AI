## Description: 
This project demonstrates a multi-source chatbot leveraging Retrieval-Augmented Generation (RAG) principles to provide intelligent responses based on document embeddings. The chatbot utilizes LangChain, Chroma, and HuggingFaceEmbeddings to retrieve and interact with various document sources like Wikipedia and Arxiv.

## Features:

**Multi-Source Retrieval:** Integrates with Wikipedia and Arxiv to pull relevant information.
**Retrieval-Augmented Generation (RAG):** Combines information retrieval with generative models for highly relevant and precise responses.
**Chroma Vector Store:** Efficiently stores document embeddings for fast retrieval during conversations.
**Document Splitting:** Uses RecursiveCharacterTextSplitter to ensure optimized processing of large documents.
**HuggingFaceEmbeddings:** Utilized for generating document embeddings and integrating with HuggingFace models.
**Flexible Prompting:** Customizes conversation with ChatPromptTemplate to create contextually aware chatbot interactions.

## Technologies:

* Python
* LangChain
* Chroma
* HuggingFace Embeddings
* WikipediaAPIWrapper, ArxivAPIWrapper
* Generative AI models

## Usage:

* Install required libraries using requirements.txt.
* Run the chatbot locally by executing python multiAIRagChatbot.py.
* Engage with the chatbot to retrieve information from multiple sources, powered by RAG principles.
