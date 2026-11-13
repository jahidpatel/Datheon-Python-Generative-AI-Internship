
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "document_qa"


# Initialize LLM
def initialize_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    llm = ChatOpenAI(
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        streaming=True
    )

    return llm


# Initialize embedding model
def initialize_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# Load PDF and TXT documents
def load_documents():

    documents = []

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError("Documents folder not found!")

    for file_path in DOCUMENTS_DIR.iterdir():

        if file_path.suffix.lower() == ".pdf":

            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif file_path.suffix.lower() == ".txt":

            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No PDF or TXT documents found!")

    print(f"Loaded {len(documents)} document pages/files.")

    return documents


# Split documents into chunks
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} document chunks.")

    return chunks


# Create or load vector database
def initialize_vectorstore(embeddings):

    db_file = CHROMA_DIR / "chroma.sqlite3"

    if db_file.exists():

        print("Loading existing Chroma database...")

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )

        if vectorstore._collection.count() > 0:
            print("Existing vector database loaded.")
            return vectorstore

        print("Existing database is empty. Rebuilding...")

    print("Creating new vector database...")

    documents = load_documents()
    chunks = split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector database created successfully!")

    return vectorstore


# Initialize complete RAG system
def initialize_rag():

    print("Initializing RAG system...")

    embeddings = initialize_embeddings()

    vectorstore = initialize_vectorstore(embeddings)

    llm = initialize_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    print("RAG system is ready!")

    return llm, retriever


# Generate streaming AI response
def generate_response(question, chat_history, llm, retriever):

    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Build conversation messages
    messages = [
        SystemMessage(
            content=f"""
You are a helpful AI assistant with document question-answering
capabilities.

Answer the user's question using the provided document context.

If the answer is not available in the documents, clearly say
that the information is not available in the provided documents.

Do not invent facts.

DOCUMENT CONTEXT:
{context}
"""
        )
    ]

    # Include recent conversation history
    for message in chat_history[-6:]:

        if message["role"] == "user":

            messages.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":

            messages.append(
                AIMessage(content=message["content"])
            )

    # Add current question
    messages.append(
        HumanMessage(content=question)
    )

    # Stream response token by token
    for chunk in llm.stream(messages):

        if isinstance(chunk.content, str):
            if chunk.content:
                yield chunk.content
