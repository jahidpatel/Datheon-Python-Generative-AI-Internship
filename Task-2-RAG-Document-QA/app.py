
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

DOCUMENTS_DIR = Path("documents")
DB_DIR = "./chroma_db"
COLLECTION_NAME = "document_qa"

DOCUMENTS_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD AI MODELS
# ==================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing from .env"
        )

    return ChatOpenAI(
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2
    )


# ==================================================
# 3. DOCUMENT LOADING
# ==================================================

def load_documents():

    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        try:

            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif extension == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            else:
                continue

            loaded_docs = loader.load()
            documents.extend(loaded_docs)

        except Exception as error:

            st.warning(
                f"Could not read {file_path.name}: {error}"
            )

    return documents


# ==================================================
# 4. VECTOR DATABASE MANAGEMENT
# ==================================================

def create_vector_database(documents):

    if not documents:
        raise ValueError(
            "No readable PDF or TXT documents found."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text could be extracted."
        )

    embeddings = load_embedding_model()

    # Delete the old collection before rebuilding
    try:

        old_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        old_db.delete_collection()
        del old_db

    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vector_store, len(chunks)


def load_existing_vector_database():

    if not Path(DB_DIR).exists():
        return None

    try:

        embeddings = load_embedding_model()

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        if vector_store._collection.count() > 0:
            return vector_store

    except Exception:
        return None

    return None


def remove_vector_database():

    try:

        embeddings = load_embedding_model()

        old_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        old_db.delete_collection()
        del old_db

    except Exception:
        pass


# ==================================================
# 5. CONVERSATIONAL MEMORY
# ==================================================

def create_standalone_question(question, history):

    if not history:
        return question

    recent_history = history[-8:]

    conversation = ""

    for message in recent_history:

        role = message["role"].upper()
        content = message["content"]

        conversation += (
            f"{role}: {content}\n"
        )

    prompt = f"""
You are a question rewriting assistant.

Your job is to understand follow-up questions
using the previous conversation.

Rewrite the latest question into a standalone
question that can be understood without the
conversation history.

Rules:
1. Preserve the original meaning.
2. Resolve references such as it, they, this,
   that project, and those technologies.
3. Do not answer the question.
4. If it is already standalone, keep it unchanged.
5. Return only the rewritten question.

CONVERSATION:
{conversation}

LATEST QUESTION:
{question}

STANDALONE QUESTION:
"""

    response = load_llm().invoke(prompt)

    rewritten = response.content.strip()

    return rewritten if rewritten else question


# ==================================================
# 6. QUESTION ANSWERING WITH RAG
# ==================================================

def answer_question(question, vector_store, history):

    # Convert follow-up into a standalone question
    search_question = create_standalone_question(
        question,
        history
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(search_question)

    if not relevant_docs:

        return (
            "I couldn't find relevant information "
            "in the uploaded documents.",
            []
        )

    context_parts = []
    sources = []

    for doc in relevant_docs:

        filename = Path(
            doc.metadata.get("source", "Unknown")
        ).name

        page = doc.metadata.get("page")

        if isinstance(page, int):
            page_number = page + 1
            source_name = (
                f"{filename}, Page {page_number}"
            )
        else:
            page_number = None
            source_name = filename

        context_parts.append(
            f"Source: {source_name}\n"
            f"Content: {doc.page_content}"
        )

        sources.append({
            "filename": filename,
            "page": page_number,
            "content": doc.page_content
        })

    context = "\n\n".join(context_parts)

    # Include recent conversation for context
    history_text = ""

    for message in history[-8:]:

        history_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are an intelligent Document Question
Answering Assistant.

Answer using ONLY the supplied document context.

RULES:
1. Do not invent facts.
2. If information is missing, say:
   "I couldn't find that information in the
   uploaded documents."
3. Give clear and simple answers.
4. Use conversation history to understand
   follow-up questions.
5. Do not treat conversation history as
   evidence about the documents.
6. Use the document context as the factual source.

CONVERSATION HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = load_llm().invoke(prompt)

    return response.content, sources


# ==================================================
# 7. DOCUMENT SUMMARIZATION
# ==================================================

def summarize_documents(documents):

    if not documents:
        raise ValueError(
            "No documents are available to summarize."
        )

    # Group text by document
    grouped_documents = {}

    for doc in documents:

        filename = Path(
            doc.metadata.get("source", "Unknown")
        ).name

        if filename not in grouped_documents:
            grouped_documents[filename] = []

        grouped_documents[filename].append(
            doc.page_content
        )

    summaries = []

    for filename, contents in grouped_documents.items():

        document_text = "\n\n".join(contents)

        # Limit the context sent to the LLM
        if len(document_text) > 30000:
            document_text = (
                document_text[:30000]
                + "\n[Text truncated for summarization]"
            )

        prompt = f"""
You are a professional document summarization assistant.

Summarize the following document.

Include:
1. Main topic
2. Important points
3. Key findings or objectives
4. Important technologies, names or terms
5. Overall conclusion

Use simple and clear language.
Do not invent information.

DOCUMENT NAME:
{filename}

DOCUMENT CONTENT:
{document_text}

SUMMARY:
"""

        response = load_llm().invoke(prompt)

        summaries.append(
            f"## {filename}\n\n{response.content}"
        )

    return "\n\n---\n\n".join(summaries)


# ==================================================
# 8. SESSION STATE
# ==================================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = (
        load_existing_vector_database()
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:

    st.session_state.indexed = (
        st.session_state.vector_store is not None
    )

if "summary" not in st.session_state:
    st.session_state.summary = None


# ==================================================
# 9. SIDEBAR - DOCUMENT MANAGEMENT
# ==================================================

with st.sidebar:

    st.title("📚 Document Manager")

    st.caption(
        "Upload, organize and process your documents."
    )

    # Upload documents
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "Save uploaded documents",
            use_container_width=True
        ):

            saved_count = 0

            for uploaded_file in uploaded_files:

                safe_name = Path(
                    uploaded_file.name
                ).name

                destination = (
                    DOCUMENTS_DIR / safe_name
                )

                destination.write_bytes(
                    uploaded_file.getvalue()
                )

                saved_count += 1

            st.success(
                f"{saved_count} document(s) saved."
            )

            st.info(
                "Now click Process Documents."
            )

    st.divider()

    # List documents
    st.subheader("Your documents")

    document_files = sorted([
        file for file in DOCUMENTS_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".pdf", ".txt"]
    ])

    if document_files:

        for file in document_files:
            st.write(f"📄 {file.name}")

        # Document removal
        selected_files = st.multiselect(
            "Select documents to remove",
            options=[file.name for file in document_files]
        )

        if st.button(
            "🗑️ Remove selected documents",
            use_container_width=True,
            disabled=not selected_files
        ):

            for filename in selected_files:

                file_path = DOCUMENTS_DIR / filename

                if file_path.exists():
                    file_path.unlink()

            # Invalidate old database
            st.session_state.vector_store = None
            st.session_state.indexed = False
            st.session_state.messages = []
            st.session_state.summary = None

            remove_vector_database()

            st.success(
                "Selected documents removed."
            )

            st.rerun()

    else:

        st.caption("No documents uploaded.")

    st.divider()

    # Process documents
    if st.button(
        "⚙️ Process Documents",
        use_container_width=True
    ):

        with st.spinner(
            "Processing documents and creating embeddings..."
        ):

            try:

                docs = load_documents()

                if not docs:

                    st.warning(
                        "Upload at least one readable document."
                    )

                else:

                    vector_store, chunk_count = (
                        create_vector_database(docs)
                    )

                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.indexed = True
                    st.session_state.messages = []
                    st.session_state.summary = None

                    st.success(
                        "Documents processed successfully!"
                    )

                    st.write(
                        f"Loaded pages/documents: {len(docs)}"
                    )

                    st.write(
                        f"Chunks indexed: {chunk_count}"
                    )

            except Exception as error:

                st.error(
                    f"Processing error: {error}"
                )

    if st.session_state.indexed:
        st.success("Vector database ready.")
    else:
        st.warning("Documents need processing.")

    st.divider()

    # Summarization
    st.subheader("Document Summarizer")

    if st.button(
        "📝 Summarize Documents",
        use_container_width=True,
        disabled=not st.session_state.indexed
    ):

        with st.spinner(
            "Generating document summaries..."
        ):

            try:

                docs = load_documents()

                summary = summarize_documents(docs)

                st.session_state.summary = summary

                st.success(
                    "Summary generated successfully!"
                )

            except Exception as error:

                st.error(
                    f"Summarization error: {error}"
                )

    st.divider()

    # Download chat history
    st.subheader("Chat Export")

    if st.session_state.messages:

        export_text = (
            "DOCUMENT Q&A ASSISTANT\n"
            f"Exported: {datetime.now().strftime('%d-%m-%Y %H:%M')}\n"
            + "=" * 50
            + "\n\n"
        )

        for message in st.session_state.messages:

            role = message["role"].upper()

            export_text += (
                f"{role}:\n"
                f"{message['content']}\n\n"
            )

            if message["role"] == "assistant":

                for source in message.get("sources", []):

                    if source["page"] is not None:

                        export_text += (
                            f"Source: {source['filename']}, "
                            f"Page {source['page']}\n"
                        )

                    else:

                        export_text += (
                            f"Source: {source['filename']}\n"
                        )

                export_text += "\n"

        st.download_button(
            "⬇️ Download Chat History",
            data=export_text,
            file_name="document_qa_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:

        st.caption("No conversation to export yet.")


# ==================================================
# 10. MAIN APPLICATION INTERFACE
# ==================================================

st.title("📚 Document Q&A Assistant")

st.caption(
    "Your intelligent AI assistant for understanding documents."
)

# Display summary if available
if st.session_state.summary:

    with st.expander(
        "📝 Document Summaries",
        expanded=True
    ):

        st.markdown(
            st.session_state.summary
        )

if not st.session_state.indexed:

    st.info(
        "Upload your PDF or TXT files from the sidebar "
        "and click Process Documents to begin."
    )

    st.markdown(
        """
        ### How it works

        1. Upload your documents.
        2. Text is extracted and split into chunks.
        3. Embeddings are generated and stored in ChromaDB.
        4. Ask questions in natural language.
        5. Nemotron generates answers using relevant content.
        """
    )

else:

    st.success(
        "Your documents are ready. Start asking questions!"
    )

    # Display conversation history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander("📎 View Sources"):

                        for source in sources:

                            if source["page"] is not None:

                                st.markdown(
                                    f"**{source['filename']} — "
                                    f"Page {source['page']}**"
                                )

                            else:

                                st.markdown(
                                    f"**{source['filename']}**"
                                )

                            st.caption(source["content"])

    # Chat input
    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question:

        # Save current question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # History before the current question
        history = st.session_state.messages[:-1]

        with st.chat_message("assistant"):

            with st.spinner(
                "Understanding your question and searching documents..."
            ):

                try:

                    answer, sources = answer_question(
                        question,
                        st.session_state.vector_store,
                        history
                    )

                    st.markdown(answer)

                    if sources:

                        with st.expander("📎 View Sources"):

                            for source in sources:

                                if source["page"] is not None:

                                    st.markdown(
                                        f"**{source['filename']} — "
                                        f"Page {source['page']}**"
                                    )

                                else:

                                    st.markdown(
                                        f"**{source['filename']}**"
                                    )

                                st.caption(source["content"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as error:

                    error_message = (
                        f"Something went wrong: {error}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })

    # Clear conversation
    if st.button("🗑️ Clear Chat History"):

        st.session_state.messages = []
        st.rerun()
