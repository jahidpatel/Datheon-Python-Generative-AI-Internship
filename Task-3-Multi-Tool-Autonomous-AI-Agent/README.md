# Multi-Tool Autonomous AI Agent

A Python-based autonomous AI agent built using **LangChain and Generative AI** that can intelligently select and execute multiple tools to answer user queries. The agent integrates web search, mathematical calculations, and local document reading through a single interactive interface.

## Project Overview

This project demonstrates how an AI agent can use external tools to perform tasks instead of relying only on the knowledge stored in an LLM.

The agent understands the user's query, determines which tool is required, executes the tool, and generates a response based on the returned results.

The project uses an LLM accessed through OpenRouter and the NVIDIA Nemotron model.

## Key Features

### 1. Web Search
- Searches the internet using DuckDuckGo.
- Retrieves relevant search results with titles, descriptions, and URLs.
- Helps answer questions requiring external information.

### 2. Python Calculator
- Performs mathematical calculations using a safe expression evaluator.
- Supports arithmetic operations such as addition, subtraction, multiplication, division, and exponentiation.
- Supports advanced mathematical functions such as square root, logarithms, trigonometric functions, and exponential calculations.
- Handles invalid expressions and mathematical errors.

### 3. Local PDF/TXT Document Reader
- Reads local PDF and TXT files from the `documents` directory.
- Extracts text from PDF documents using PyPDF.
- Reads text files directly.
- Lists available documents.
- Provides document-based summaries and answers.

### 4. Autonomous Tool Selection
- Uses LangChain to create an AI agent.
- Automatically selects the appropriate tool based on the user's query.
- Supports sequential execution of multiple tools.
- Combines information from different tools to generate a final response.

### 5. Interactive Command-Line Interface
- Provides a simple terminal-based interface.
- Accepts continuous user queries.
- Displays tool execution actions and observations.
- Supports clean application termination using `exit` or `quit`.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | AI agent development and tool integration |
| OpenRouter | LLM API access |
| NVIDIA Nemotron 3.5 Lightning | Language model |
| DuckDuckGo Search (`ddgs`) | Web search functionality |
| PyPDF | PDF text extraction |
| Python AST | Safe mathematical expression evaluation |
| python-dotenv | Environment variable management |

## Project Structure

```text
Task-3-Multi-Tool-Autonomous-AI-Agent/
│
├── documents/
│   ├── the_ocean_world_five_page_guide.pdf
│   └── urban_gardening_and_sustainable_cities.txt
│
├── tools.py
├── test_tools.py
├── agent.py
├── test_agent.py
├── main.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

**File Descriptions:**

- `tools.py` – Implements web search, calculator, and local document reader tools.
- `test_tools.py` – Tests individual tools independently.
- `agent.py` – Configures the LLM and creates the autonomous LangChain agent.
- `test_agent.py` – Tests the agent's tool selection and execution.
- `main.py` – Runs the interactive command-line application.
- `requirements.txt` – Contains the project dependencies.
- `documents/` – Stores sample PDF and TXT documents for testing.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Visual Studio Code (recommended)
- OpenRouter API key
- Internet connection for web search and LLM access

### Step 1: Clone the Repository

clone the repository
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key

Create a `.env` file in the project root directory.

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Obtain an API key from:

https://openrouter.ai/keys

**Note:** Never upload your actual `.env` file or API key to GitHub.

### Step 5: Run the Application

```bash
python main.py
```

The interactive AI agent will start in the terminal.

## Example Queries

The following queries can be used to test the agent.

**Calculator:**

```text
Calculate 25% of 8400.
```

**Advanced Calculation:**

```text
If I invest 50000 at 12% annual compound interest for 3 years, what will be the final amount?
```

**Web Search:**

```text
Search for the latest developments in Artificial Intelligence.
```

**TXT Document Reader:**

```text
Summarize urban_gardening_and_sustainable_cities.txt.
```

**PDF Document Reader:**

```text
Explain the main topics covered in the_ocean_world_five_page_guide.pdf.
```

**Document Discovery:**

```text
What documents are available in my documents folder?
```

**Multi-Tool Execution:**

```text
Search for the population of Bengaluru and calculate 15% of that population. Show the source and calculation.
```

To close the application:

```text
exit
```

## Testing

The project includes two testing scripts:

**1. Individual Tool Testing**

```bash
python test_tools.py
```

Tests the web search, calculator, and document reader independently.

**2. Autonomous Agent Testing**

```bash
python test_agent.py
```

Tests the agent's ability to understand queries, select tools, and execute them.

## Learning Outcomes

Through this project, I gained practical experience in:

- Building autonomous AI agents using LangChain.
- Integrating LLMs through OpenRouter APIs.
- Creating and connecting custom tools.
- Implementing safe mathematical expression evaluation.
- Extracting and processing PDF and TXT documents.
- Understanding autonomous tool selection and sequential tool execution.
- Handling API errors and tool execution failures.
- Building interactive AI applications using Python.

## Future Improvements

- Add more tools such as weather, email, and database operations.
- Develop a Streamlit-based graphical user interface.
- Implement conversational memory for context-aware interactions.
- Improve web search result validation and source grounding.
- Support additional document formats such as DOCX.
- Add more robust error handling and logging.

## Internship Project

**Internship:** Python + Generative AI & LangChain Internship  
**Organization:** Datheon AI System and Automation Studio  
**Task:** Task 3 – Multi-Tool Autonomous AI Agent

---

**Developed as part of the Datheon Python + Generative AI & LangChain Internship.**