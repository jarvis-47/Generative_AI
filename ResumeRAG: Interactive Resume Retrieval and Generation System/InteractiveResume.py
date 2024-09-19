## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
import nltk
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
pinecone_api_key = os.getenv("PINCONE_API_KEY")
api_key = os.getenv("GROQ_API_KEY")


# Download NLTK punkt resources
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt_tab")


# Download the NLTK resources
download_nltk_resources()

## Initialize the Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

## Input the Groq API Key
# api_key = st.text_input("Enter your Groq API key:", type="password")

# if not api_key:
#     st.warning("Please enter the GRoq API Key")
#     st.stop()  # Stops execution until the key is provided


## Initialize Pinecone
@st.cache_resource
def initialize_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


## Initialize the Pinecone Index
index_name = "hybrid-search-langchain-pinecone"
index = initialize_pinecone(pinecone_api_key, index_name)

## Initialize the Langchain ChatGroq
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

## initialize the BM25 encoder
bm25_encoder = BM25Encoder().default()

## chat interface

session_id = st.text_input("Session ID", value="default_session")
## statefully manage chat history


uploaded_files = st.file_uploader(
    "Choose A PDf file", type="pdf", accept_multiple_files=True
)
## Process uploaded  PDF's
if "store" not in st.session_state:
    st.session_state.store = {}
    st.session_state.uploaded_files = {}

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.session_state.uploaded_files[file_name] = uploaded_file
        # Process the file and save it in state
        temppdf = f"./temp_{file_name}.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Check embeddings
    document_embeddings = [
        embeddings.embed_query(text_chunk.page_content) for text_chunk in splits
    ]

    # Create unique IDs for each document
    data_to_upsert = [
        (str(i), embedding, {"context": text_chunk.page_content})
        for i, (embedding, text_chunk) in enumerate(zip(document_embeddings, splits))
    ]

    # Upsert the document embeddings to Pinecone
    index.upsert(vectors=data_to_upsert)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
    )

    contextualize_q_system_prompt = """
    You are an expert assistant tasked with ensuring the user's query is precise and unambiguous. Based on the provided conversation history, reformulate the latest user question to be self-contained and explicit. Use specific details from the user's resume and the job description to clarify any vague or context-dependent terms. Your goal is to make the question as detailed as possible to ensure accurate retrieval of relevant information.

    Important:
    1. Do NOT answer the question or provide additional information.
    2. Only rephrase or clarify the query to ensure it's contextually rich, clear, and can stand alone without prior conversation context.
    3. Use details from the resume and job description to ensure accuracy and alignment with the user's career goals.

    Return the reformulated question or the original if no changes are needed.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ## Answer question

    # Answer question
    system_prompt = (
        "You are an expert assistant for answering user questions accurately and concisely. "
        "You will use the following retrieved context from the user's resume and the provided job description to formulate your response. "
        "Ensure your response is highly relevant to the job description and the user's career background."
        "\n\n"
        "Important Instructions:\n"
        "1. **Use the resume context** to ensure the response aligns with the user's professional experience and skills.\n"
        "2. **Use the job description** to make sure the response highlights qualifications or requirements that are directly relevant to the job.\n"
        "3. **Cross-check multiple pieces of retrieved context** to ensure the accuracy of your answer, and address any discrepancies between the resume and the job description.\n"
        "4. If the provided context is insufficient to fully answer the question, clearly state what information is missing or ambiguous.\n"
        "5. **Keep your response as concise as possible**, including only the most relevant details while ensuring the answer is accurate and complete."
        "\n\n"
        "Resume Context:\n"
        "{context}"
        "\n\n"
        "Job Description:\n"
        "{job_description}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.write(
        "PDF's uploaded successfully. You can now start chatting with the assistant."
    )

    info = None

    user_input = st.text_area("Your question:")
    add_info = st.checkbox("Add more information (optional)")

    if user_input:
        if add_info:
            info = st.text_input("Add more information:")
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input, "job_description": info},
            config={
                "configurable": {"session_id": session_id}
            },  # constructs a key "abc123" in `store`.
        )

        st.write("Assistant:", response["answer"])
