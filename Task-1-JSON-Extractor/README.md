# Structured LLM Prompt Pipeline & JSON Extractor

A Python project built for the Datheon **Python + Generative AI & LangChain Internship**. It extracts structured information from resume and invoice text or text-based PDF files using an LLM, then validates the output with Pydantic and saves it as JSON.

## Features

- Analyze a sample resume or invoice.
- Extract text from local, text-based PDF files with `pypdf`.
- Call an LLM through the OpenRouter Chat Completions API.
- Use a document-specific schema for resumes and invoices.
- Validate output using Pydantic.
- Retry once when the model returns invalid or incomplete JSON.
- Save validated results in the `output/` directory.

### Resume fields

- `document_type`
- `person_name`
- `email`
- `phone`
- `skills`
- `work_experience_year`
- `internship_experience`

### Invoice fields

- `document_type`
- `company`
- `invoice_number`
- `invoice_date`
- `total_amount`
- `currency`
- `items`

## Requirements

- Python 3.10 or later
- An OpenRouter account and API key
- Internet access for the LLM request

The project is configured to use `nvidia/nemotron-3.5-lightning:free`. Free model availability, limits, and provider routing can change. This code does not guarantee continuous free availability. Check the model's current pricing and usage limits before running it; do not assume all OpenRouter models are free.

## Setup

### 1. Clone the repository

clone the repository


### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Replace the placeholder with your own key. **Never commit `.env`, publish your API key, or paste it into issues or screenshots.** The `.gitignore` file excludes `.env`.

### 5. Run the program

```powershell
python main.py
```

Choose one of the menu options:

1. Test with a sample resume.
2. Test with a sample invoice.
3. Upload a resume or invoice PDF by entering its path.

For option 3, enter the full path to a PDF, or a relative path such as `sample_invoice_green_farms.pdf` if the file is in the current project folder.

## Output

Validated JSON results are saved to the `output/` directory. For example:

- `output/sample_resume_extracted.json`
- `output/sample_invoice_extracted.json`

The `output/` directory is excluded from version control because it contains generated results.

## Limitations

- The PDF reader extracts selectable text. Scanned or image-only PDFs need OCR, which is not included.
- LLM responses can be incomplete, inaccurate, or unavailable due to provider limits. Review extracted values before using them.
- `work_experience_year` and `internship_experience` are model-extracted estimates/labels, not verified facts.
- Resume and invoice formats vary, so extraction quality may vary by document.
- Do not upload private resumes, invoices, API keys, or personal data to a public repository.

## Project status

The script has been tested locally with fictional sample resume and invoice PDFs and with a resume PDF. More varied testing is recommended before production use.

## License

Add a license if you want others to reuse this project. If you are unsure, check the internship submission requirements before choosing one.
