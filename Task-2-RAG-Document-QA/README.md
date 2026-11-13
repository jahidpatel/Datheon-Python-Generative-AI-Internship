# Document Q&A Assistant with RAG & Vector Embeddings

An AI-powered Document Question Answering application built using Python, LangChain, ChromaDB, Hugging Face Embeddings, and NVIDIA Nemotron 3.5 Lightning through OpenRouter.

The application allows users to upload PDF and TXT documents, ask questions in natural language, receive context-based answers, generate document summaries, and view relevant source references through an interactive Streamlit web interface.

## Project Overview

Reading and searching through lengthy documents manually can be time-consuming. This project addresses that problem by developing an intelligent assistant that retrieves relevant information from uploaded documents and generates accurate, context-aware answers using a Large Language Model (LLM).

The application follows the Retrieval-Augmented Generation (RAG) approach, combining semantic search with LLM-based response generation.

## Key Features

- **Multiple Document Upload:** Upload multiple PDF and TXT files.
- **Document Processing:** Extracts and splits document content into smaller text chunks.
- **Text Embeddings:** Converts document chunks into vector representations using Sentence Transformers.
- **Vector Database:** Stores and retrieves embeddings using ChromaDB.
- **Semantic Search:** Finds relevant document content based on the meaning of user questions.
- **AI-Powered Answers:** Uses NVIDIA Nemotron 3.5 Lightning through OpenRouter to generate responses.
- **Conversational Memory:** Understands follow-up questions using recent conversation history.
- **Document Summarization:** Generates summaries of uploaded documents.
- **Document Management:** Allows users to remove documents and rebuild the vector database.
- **Source References:** Displays relevant document names and PDF page numbers.
- **Chat Export:** Allows users to download their conversation history.
- **Interactive Web Interface:** Provides a user-friendly interface using Streamlit.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | LLM and RAG application framework |
| NVIDIA Nemotron 3.5 Lightning | Large Language Model |
| OpenRouter | LLM API integration |
| Hugging Face Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| PyPDF | PDF document extraction |
| Python-dotenv | Environment variable management |

## System Architecture

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

1. **Document Upload:** Users upload PDF or TXT documents through the Streamlit interface.
2. **Text Extraction:** LangChain document loaders extract text from uploaded files.
3. **Text Chunking:** RecursiveCharacterTextSplitter divides documents into smaller chunks with overlapping content.
4. **Embedding Generation:** Sentence Transformers converts text chunks into numerical vector representations.
5. **Vector Storage:** ChromaDB stores the generated embeddings for efficient similarity-based retrieval.
6. **Question Processing:** The user's question is converted into a standalone query when it depends on previous conversation turns.
7. **Semantic Retrieval:** ChromaDB retrieves the most relevant document chunks.
8. **Response Generation:** The retrieved context is provided to NVIDIA Nemotron through OpenRouter.
9. **Answer Presentation:** The generated response and source references are displayed in Streamlit.

## Project Structure

```text
Task-2-RAG-Document-QA/
│
├── documents/
│   └── the_ocean_world_five_page_guide.pdf   (sample pdf file)
|   └── urban_gardening_and_sustainable_cities.txt   (sample txt file)
│
├── chroma_db/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

Note: `.venv/`, `.env`, and `chroma_db/` are excluded from GitHub. The `chroma_db/` directory is generated automatically when documents are processed.

## Installation and Setup

### 1. Clone the Repository

clone the repository
```

### 2. Navigate to the Project Directory

```bash
cd Task-2-RAG-Document-QA
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the API Key

Create a `.env` file in the project root directory.

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Do not upload your `.env` file or expose your API key publicly.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## How to Use

1. Start the Streamlit application.
2. Upload one or more PDF or TXT documents using the sidebar.
3. Save the uploaded documents.
4. Click **Process Documents** to extract text, generate embeddings, and create the vector database.
5. Ask questions about the uploaded documents using the chat interface.
6. View the retrieved source references to identify where the information came from.
7. Use the Document Summarizer to generate document summaries.
8. Remove documents when required and process the remaining documents again.
9. Download your conversation history as a text file.

## Models Used

**Large Language Model:**
- Provider: NVIDIA through OpenRouter
- Model: `nvidia/nemotron-3.5-lightning:free`

**Embedding Model:**
- Provider: Hugging Face / Sentence Transformers
- Model: `sentence-transformers/all-MiniLM-L6-v2`

## Environment Variables

| Variable | Description |
|---|---|
| OPENROUTER_API_KEY | API key used to access the NVIDIA Nemotron model through OpenRouter |

## Future Enhancements

- Support for Word documents and additional file formats.
- OCR support for scanned PDF documents.
- Improved conversational memory across sessions.
- Advanced document filtering and search.
- More detailed source citations and retrieval scoring.
- Cloud deployment and multi-user support.
- Improved summarization for very large documents.

## Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Retrieval-Augmented Generation (RAG).
- Working with Large Language Models and API integration.
- Implementing text embeddings and semantic search.
- Building vector databases using ChromaDB.
- Processing PDF and text documents using LangChain.
- Developing interactive AI applications using Streamlit.
- Managing Python environments and project dependencies.

## Internship Information

**Internship:** Python + Generative AI & LangChain Internship  
**Organization:** Datheon AI System and Automation Studio  
**Project:** Task 2 – Document Q&A Assistant with RAG & Vector Embeddings

---

Developed as part of the Datheon Python + Generative AI & LangChain Internship.

