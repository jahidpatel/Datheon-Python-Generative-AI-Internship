
import streamlit as st
import uuid
import hashlib
import tempfile
import os
from pathlib import Path
from itertools import chain

from rag_pipeline import initialize_rag, generate_response

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Datheon AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# SESSION STATE
# ==========================================

if "conversations" not in st.session_state:

    first_id = str(uuid.uuid4())

    st.session_state.conversations = {
        first_id: {
            "title": "New conversation",
            "messages": [],
            "upload_notice": ""
        }
    }

    st.session_state.active_chat = first_id


if "show_history" not in st.session_state:
    st.session_state.show_history = True

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "llm" not in st.session_state:
    st.session_state.llm = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()


# ==========================================
# CURRENT CONVERSATION
# ==========================================

active_chat = st.session_state.active_chat

current_chat = st.session_state.conversations[active_chat]

messages = current_chat["messages"]


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 50% -20%,
            #35205D 0%,
            #211638 32%,
            #11101C 68%,
            #090A12 100%);
    color: #F2EEFA;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    max-width: 950px;
    padding-top: 1.5rem;
    padding-bottom: 7rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #171225 0%,
        #100F1A 100%
    );
    border-right: 1px solid #302544;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] * {
    color: #E9E2F5;
}

/* LOGO */

.logo {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #FFFFFF;
    padding: 5px 0 22px;
}

.logo span {
    color: #B99AFF;
}

/* HEADER */

.top-heading {
    color: #DCD0F4;
    font-size: 13px;
    letter-spacing: 1px;
    font-weight: 600;
    text-align: right;
}

/* WELCOME SCREEN */

.welcome {
    text-align: center;
    padding-top: 16vh;
    padding-bottom: 30px;
}

.welcome-orb {
    width: 85px;
    height: 85px;
    margin: 0 auto 30px;
    border-radius: 50%;
    background: radial-gradient(
        circle at 35% 30%,
        #F0DCFF 0%,
        #C39BFF 18%,
        #8757E7 42%,
        #422477 70%,
        #17102B 100%
    );
    box-shadow:
        0 0 25px #A56CFF88,
        0 0 70px #8C4DE966,
        inset -10px -10px 25px #25113F;
    animation: breathe 2.5s ease-in-out infinite;
}

@keyframes breathe {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 25px #A56CFF88,
                    0 0 60px #8C4DE966;
    }
    50% {
        transform: scale(1.08);
        box-shadow: 0 0 40px #C69CFFAA,
                    0 0 90px #8C4DE988;
    }
}

/* THINKING ANIMATION */

.thinking-container {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 0;
}

.thinking-orb {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: radial-gradient(
        circle at 35% 30%,
        #F5E6FF,
        #B38AFF 30%,
        #7542C9 65%,
        #281443
    );
    box-shadow: 0 0 18px #A77AFF99;
    animation: think 1.2s ease-in-out infinite;
}

@keyframes think {
    0%, 100% {
        transform: scale(0.85);
        opacity: 0.7;
    }
    50% {
        transform: scale(1.15);
        opacity: 1;
        box-shadow: 0 0 35px #C39AFF;
    }
}

.thinking-text {
    color: #C8B5E7;
    font-size: 13px;
    animation: fade 1.5s ease-in-out infinite;
}

@keyframes fade {
    0%, 100% { opacity: 0.45; }
    50% { opacity: 1; }
}

/* WELCOME TEXT */

.welcome h1 {
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 650;
    letter-spacing: -1.5px;
    color: #F6F0FF;
    margin-bottom: 12px;
}

.welcome p {
    color: #B4A8C9;
    font-size: 14px;
}

/* SIDEBAR BUTTONS */

[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid transparent;
    color: #DAD0EA;
    border-radius: 10px;
    text-align: left;
    transition: 0.2s;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #2B2140;
    border-color: #57416F;
    color: white;
}

/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    background: rgba(38, 30, 53, 0.68);
    border: 1px solid rgba(174, 133, 221, 0.13);
    border-radius: 17px;
    padding: 16px 18px;
    margin-bottom: 15px;
}

/* CHAT INPUT */

[data-testid="stChatInput"] {
    background: #211A30;
    border: 1px solid #675080;
    border-radius: 18px;
    box-shadow: 0 8px 35px #00000055;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #C19BFF;
    box-shadow: 0 0 0 1px #A77BEF55;
}

[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
}

/* BUTTONS */

.stButton > button {
    background: #261D36;
    border: 1px solid #59436F;
    color: #F2EAFE;
    border-radius: 11px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #39274F;
    border-color: #B18AEF;
    color: white;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #211A30;
    border: 1px solid #59436F;
    border-radius: 12px;
    padding: 10px;
}

/* FOOTER */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# NEW CHAT FUNCTION
# ==========================================

def create_new_chat():

    new_id = str(uuid.uuid4())

    st.session_state.conversations[new_id] = {
        "title": "New conversation",
        "messages": [],
        "upload_notice": ""
    }

    st.session_state.active_chat = new_id

    st.rerun()


# ==========================================
# INITIALIZE RAG
# ==========================================

if not st.session_state.rag_initialized:

    with st.spinner("Preparing AI..."):

        try:

            llm, retriever = initialize_rag()

            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.rag_initialized = True

        except Exception as e:

            st.error(f"AI initialization failed: {e}")
            st.stop()


# ==========================================
# DOCUMENT UPLOAD PROCESSING
# ==========================================

def process_uploaded_files(files):

    if not files:
        return []

    vectorstore = st.session_state.retriever.vectorstore

    processed_files = []

    for uploaded_file in files:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash in st.session_state.uploaded_hashes:
            continue

        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in [".pdf", ".txt"]:
            continue

        temp_path = None

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:

            if suffix == ".pdf":

                documents = PyPDFLoader(temp_path).load()

            else:

                documents = TextLoader(
                    temp_path,
                    encoding="utf-8"
                ).load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(documents)

            if chunks:

                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name

                vectorstore.add_documents(chunks)

                st.session_state.uploaded_hashes.add(file_hash)

                processed_files.append(uploaded_file.name)

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return processed_files


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="logo"><span>✦</span> Datheon</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "＋   New chat",
        use_container_width=True
    ):

        create_new_chat()

    st.markdown("---")

    st.toggle(
        "Show chat history",
        key="show_history"
    )

    if st.session_state.show_history:

        st.caption("YOUR CONVERSATIONS")

        conversation_items = list(
            st.session_state.conversations.items()
        )

        for chat_id, chat in reversed(conversation_items):

            if st.button(
                f"◦  {chat['title'][:32]}",
                key=f"history_{chat_id}",
                use_container_width=True
            ):

                st.session_state.active_chat = chat_id
                st.rerun()

    st.markdown("---")

    st.caption("KNOWLEDGE SOURCES")

    st.caption("📘 Ocean World Guide")
    st.caption("📄 Urban Gardening")

    st.markdown("---")

    st.caption("Datheon AI · Powered by LangChain")


# ==========================================
# UPDATE ACTIVE CONVERSATION
# ==========================================

active_chat = st.session_state.active_chat

current_chat = st.session_state.conversations[active_chat]

messages = current_chat["messages"]


# ==========================================
# MAIN HEADER
# ==========================================

header_left, header_right = st.columns([4, 1])

with header_left:

    st.markdown(
        '<div style="font-size:14px;color:#CBB9E7;'
        'font-weight:600;">✦ Datheon AI</div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        '<div class="top-heading">● ONLINE</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ==========================================
# WELCOME SCREEN
# ==========================================

if not messages:

    st.markdown("""
    <div class="welcome">
        <div class="welcome-orb"></div>
        <h1>How can I help you today?</h1>
        <p>Ask a question, explore your documents, or start a conversation.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("attachments"):

            for attachment in message["attachments"]:
                st.caption(f"📎 {attachment}")


# ==========================================
# ATTACHMENT MENU
# ==========================================

with st.popover("＋  Attach"):

    st.caption("UPLOAD DOCUMENTS OR IMAGES")

    uploaded_files = st.file_uploader(
        "Select files",
        type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"chat_upload_{active_chat}"
    )

    if uploaded_files:

        for file in uploaded_files:

            suffix = Path(file.name).suffix.lower()

            if suffix in [".png", ".jpg", ".jpeg", ".webp"]:

                st.image(
                    file,
                    caption=file.name,
                    width=180
                )

                st.info(
                    "Image preview available. "
                    "Image question-answering is not enabled yet."
                )

        if st.button(
            "Add documents to knowledge base",
            key=f"process_upload_{active_chat}"
        ):

            with st.spinner("Processing documents..."):

                try:

                    processed = process_uploaded_files(
                        uploaded_files
                    )

                    if processed:

                        current_chat["upload_notice"] = (
                            "Added: " + ", ".join(processed)
                        )

                    else:

                        current_chat["upload_notice"] = (
                            "No new documents were added. "
                            "Files may already exist or be unsupported."
                        )

                    st.rerun()

                except Exception as e:

                    st.error(f"Upload failed: {e}")


# ==========================================
# CHAT-SPECIFIC UPLOAD NOTIFICATION
# ==========================================

if current_chat["upload_notice"]:

    st.success(current_chat["upload_notice"])


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Message Datheon AI..."
)


# ==========================================
# AI RESPONSE GENERATION
# ==========================================

if question:

    active_chat = st.session_state.active_chat

    current_chat = st.session_state.conversations[active_chat]

    current_messages = current_chat["messages"]

    current_messages.append({
        "role": "user",
        "content": question
    })

    # Automatically generate conversation title
    if len(current_messages) == 1:

        current_chat["title"] = question[:32]

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        animation = st.empty()

        animation.markdown("""
        <div class="thinking-container">
            <div class="thinking-orb"></div>
            <div class="thinking-text">
                Datheon is thinking...
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:

            generator = generate_response(
                question=question,
                chat_history=current_messages[:-1],
                llm=st.session_state.llm,
                retriever=st.session_state.retriever
            )

            # Keep animation visible until first token arrives
            first_token = next(generator, None)

            animation.empty()

            if first_token is None:

                response = (
                    "I couldn't generate a response. Please try again."
                )

                st.markdown(response)

            else:

                response = st.write_stream(
                    chain([first_token], generator)
                )

            current_messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:

            animation.empty()

            st.error(f"Something went wrong: {e}")

    st.rerun()
