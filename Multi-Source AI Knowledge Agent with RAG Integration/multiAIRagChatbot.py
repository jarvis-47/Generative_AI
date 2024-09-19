from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
import streamlit as st
import requests
from requests.exceptions import Timeout, RequestException
from pprint import pprint
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize Streamlit app
st.title("Material Science Query Assistant")
st.write("Ask a question related to computational material science!")

# User input for query
question = st.text_input("How can I help you today?:")

# Define the Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the documents (material science journals/blogs)
urls = [
    "https://www.journals.elsevier.com/computational-materials-science",
    "https://www.nature.com/npjcompumats/",
    "https://www.materialscloud.org/",
    "https://onlinelibrary.wiley.com/journal/1096987x",
    "https://materialsproject.org/",
    "https://www.mrs.org/bulletin",
    "https://iopscience.iop.org/journal/1749-4699",
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=750, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

# Create a ChromaDB Vector Store to store the document embeddings
vector_store = Chroma(
    embedding_function=embeddings, collection_name="material_science_docs"
)

# Add the documents to the vector store
vector_store.add_documents(doc_splits)

# Retrieving from ChromaDB
retriever = vector_store.as_retriever()

### Router


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search", "arxiv_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia, arxiv or a vectorstore.",
    )


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """
You are an expert assistant trained to intelligently route user queries to the most relevant knowledge source: either a computational material science-specific vectorstore, arxiv_search, or wiki_search.

- The **vectorstore** contains detailed, curated documents from top computational material science journals and blogs. These documents cover advanced topics such as computational modeling, materials simulation, nanomaterials, energy storage, advanced manufacturing, materials informatics, and machine learning in materials science. Use the vectorstore **only** for highly specific, research-driven queries or when the user explicitly requests technical or advanced information.
  
- **Arxiv** provides peer-reviewed research papers for queries seeking the latest research, technical depth, or cutting-edge insights. Use Arxiv for in-depth exploration of a specific material science concept or for detailed research papers.

- **Wikipedia** provides general, introductory, or broad information. Use Wikipedia for queries asking for definitions, general explanations, or broad overviews of material science topics. Examples include basic definitions (e.g., "What is nanomaterials?") or general background information.

Your job is to analyze the user's query and route it to the most appropriate source based on the following guidelines:
1. If the query seeks a **definition**, a **general explanation**, or a **broad introduction** to a topic, route the query to **wiki_search**.
2. If the query asks for **recent research**, **cutting-edge developments**, or a **detailed exploration** of a specific topic, route it to **arxiv_search**.
3. If the query requires **advanced technical details**, **specific examples**, or **deep research**, prioritize the **vectorstore**. Use this for technical queries that go beyond general explanations.

Be precise in your routing. Ensure your response is one of the following keywords: **vectorstore**, **wiki_search**, or **arxiv_search**.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=5000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=5000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


## Graph


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    question = state["question"]
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


def wiki_search(state):
    """
    Perform a Wikipedia search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]

    st.write(f"Performing Wikipedia search for: {question}")

    # Wiki search
    docs = wiki.invoke({"query": question})

    wiki_results = docs
    wiki_results = [Document(page_content=wiki_results)]

    return {"documents": wiki_results, "question": question}


def arxiv_search(state):
    """
    arxiv search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended arxiv results
    """

    question = state["question"]

    # Arxiv search
    docs = arxiv.invoke({"query": question})
    st.write(f"Retrieving documents for questionA: {question}")
    arxiv_results = docs
    arxiv_results = [Document(page_content=arxiv_results)]

    return {"documents": arxiv_results, "question": question}


def generate_llm_response(documents, question):
    prompt = ChatPromptTemplate.from_template(
        """
    You are an expert assistant. Based on the following context retrieved from relevant documents, generate a detailed, well-structured response to the user query. Use the context as your primary source of information, and leverage your reasoning abilities to ensure the response is coherent, accurate, and insightful. If any crucial details are missing in the context, use your knowledge to fill in the gaps where necessary.

    Context:
    {context}

    User Query:
    {input}
    """
    )

    llm_chain = create_stuff_documents_chain(llm, prompt)
    llm_response = llm_chain.invoke({"input": question, "context": documents})

    return llm_response


def route_question(query):
    """Route the question to either vectorstore, Arxiv, or Wikipedia."""
    try:
        source = question_router.invoke({"question": query})
    except Exception as e:
        st.error(f"Error routing the query: {e}")
        return None

    try:
        if source.datasource == "wiki_search":
            st.write("Routing to Wikipedia...")
            return "wiki_search"
        elif source.datasource == "vectorstore":
            st.write("Routing to Vectorstore...")
            return "vectorstore"
        elif source.datasource == "arxiv_search":
            st.write("Routing to Arxiv...")
            return "arxiv_search"
        else:
            st.error("Invalid routing detected.")
            return None  # Handle case where no valid route is found
    except Timeout:
        st.error(
            f"Timeout occurred while querying the {source.datasource}. Please try again."
        )
    except RequestException as e:
        st.error(f"Network error while querying {source.datasource}: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # Node for Wikipedia search
workflow.add_node("retrieve", retrieve)  # Node for retrieving from vectorstore
workflow.add_node("arxiv_search", arxiv_search)  # Node for Arxiv search

# Build graph and define conditional edges
workflow.add_conditional_edges(
    START,
    route_question,  # This should route to valid nodes
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
        "arxiv_search": "arxiv_search",
    },
)

workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
workflow.add_edge("arxiv_search", END)

# Compile the graph into the app
app = workflow.compile()

# Streamlit interactive part: Capturing response using app.stream()
# Streamlit interactive part: Capturing response using app.stream()
if question:
    st.write("Processing query...")

    inputs = {"question": question}

    final_response = None

    # Process the stream, but only capture the final output
    for output in app.stream(inputs):
        for key, value in output.items():
            if "documents" in value and len(value["documents"]) > 0:
                st.write(f"Node: {key}")

                # Process the documents with LLM
                documents_content = " ".join(
                    [doc.page_content for doc in value["documents"]]
                )
                final_response = generate_llm_response(value["documents"], question)

    # Display only the final response in Streamlit
    if final_response:
        # If final_response is the output from the LLM (string), display it directly
        st.write(final_response)
    else:
        st.write("No documents found for this query.")
