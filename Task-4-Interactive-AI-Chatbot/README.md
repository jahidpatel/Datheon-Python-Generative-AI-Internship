
# Datheon AI – Interactive Document Chatbot

## Project Overview

Datheon AI is an interactive AI-powered chatbot built using Python, Streamlit, and LangChain. It allows users to upload PDF and TXT documents and interact with their content through natural language conversations.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware responses using NVIDIA Nemotron through the OpenRouter API.

## Key Features

- **Interactive Chat Interface:** Modern chatbot interface with a violet-themed design.
- **Document Upload:** Supports PDF and TXT file uploads.
- **RAG-based Question Answering:** Retrieves relevant document information before generating answers.
- **Conversation Memory:** Maintains previous messages to understand follow-up questions.
- **Chat History Management:** Supports creating new conversations and switching between previous chats.
- **Streaming Responses:** Displays AI-generated responses progressively.
- **AI Thinking Animation:** Provides a visual indicator while generating responses.
- **Context-aware Responses:** Answers questions based on retrieved document content.
- **Hallucination Control:** Instructs the AI to acknowledge when information is unavailable in the provided documents.
- **Image Preview:** Supports image uploads and previews within the chat interface.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive chatbot interface |
| LangChain | LLM integration and RAG workflow |
| OpenRouter API | Access to the language model |
| NVIDIA Nemotron 3.5 Lightning | AI response generation |
| HuggingFace | Text embedding model |
| Sentence Transformers | Document embeddings |
| ChromaDB | Vector database and similarity search |
| PyPDF | PDF document extraction |
| python-dotenv | Environment variable management |

## RAG Architecture

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Loading:** Loads PDF and TXT documents.
2. **Text Splitting:** Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation:** Converts text chunks into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Storage:** Stores embeddings in ChromaDB.
5. **Similarity Search:** Retrieves the most relevant document chunks based on the user's question.
6. **Context Preparation:** Combines retrieved information with recent conversation history.
7. **Response Generation:** Sends the prepared context to NVIDIA Nemotron through OpenRouter.
8. **Streaming Output:** Displays the generated response progressively in the chatbot interface.

## Project Structure

```text
Task-4-Interactive-AI-Chatbot/
│
├── documents/
│   ├── the_ocean_world_five_page_guide.pdf
│   └── urban_gardening_and_sustainable_cities.txt
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

clone the repository
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project directory and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Obtain an API key from:

https://openrouter.ai/

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Configuration

**Language Model:**
- Provider: OpenRouter
- Model: `nvidia/nemotron-3.5-lightning:free`

**Embedding Model:**
- `sentence-transformers/all-MiniLM-L6-v2`

**Vector Database:**
- ChromaDB

**Text Chunking:**
- Chunk Size: 500
- Chunk Overlap: 100

**Retrieval:**
- Top 4 relevant document chunks.

## Testing

The application was tested for:

- PDF and TXT document uploading.
- Document extraction and embedding generation.
- Semantic document retrieval.
- Context-aware question answering.
- Follow-up questions using conversation history.
- Streaming AI responses.
- New conversation creation and chat history management.
- Handling questions outside the available document context.

## Future Enhancements

- Vision-enabled image understanding.
- Conversation-specific document storage and retrieval.
- Persistent chat history across application restarts.
- Support for additional document formats.
- User authentication and multi-user support.
- Document source citations in generated answers.

## Conclusion

Datheon AI demonstrates the integration of Large Language Models, Retrieval-Augmented Generation, vector embeddings, and conversational memory into an interactive chatbot application.

The project provides a practical approach to building document-based AI assistants using Python, LangChain, and Streamlit.

---

**Developed as part of the Datheon – Python + Generative AI & LangChain Internship.**
