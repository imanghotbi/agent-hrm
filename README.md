# Agent HRM (Resume Intelligence Workflow)

A Chainlit + LangGraph application for HR teams that:
- OCRs PDF resumes
- Structures them into a consistent schema
- Scores candidates against hiring requirements
- Compares candidates side-by-side
- Answers ad-hoc questions against stored candidate data
- Generates job descriptions

The assistant communicates in Persian (Farsi) by default for user-facing flows.

## Architecture Overview
- **Chainlit**: UI and conversational runtime
- **LangGraph**: Orchestrates the workflow state machine
- **MongoDB**: Stores processed resumes and usage logs
- **MinIO**: Stores raw resume files and comparison uploads

Core modules:
- `app/workflow/`: main graph and node logic
- `app/services/`: OCR, LLM factory, analyzers, storage services
- `utils/`: prompts, helpers, schema extraction

## Requirements
- Python 3.10+ (recommended)
- [uv](https://github.com/astral-sh/uv)
- Docker + docker-compose (for MongoDB + MinIO)

## Setup

### 1) Install dependencies
```bash
uv pip install -r requirements.txt
```

### 2) Configure environment variables
Create a `.env` file in the project root with at least the following values:

```env
# LLM
API_KEY=your_llm_api_key
BASE_URL=your_llm_base_url
MODEL_NAME=deepseek/deepseek-v3.2
STRUCTURED_MODEL_NAME=deepseek/deepseek-v3.2

# Mongo
MONGO_ENDPOINT=localhost:27017
MONGO_DB_NAME=hrm
MONGO_DB_USAGE=usage
MONGO_COLLECTION=resumes
MONGO_USERNAME=root
MONGO_PASSWORD=example

# MinIO
MINIO_ACCESS_KEY=hrm_resume
MINIO_SECRET_KEY=your_minio_secret
MINIO_ENDPOINT=http://localhost:9000
MINIO_RESUME_BUCKET=resumes
MINIO_COMPARE_BUCKET=compare-resume

# Optional tuning
OCR_WORKERS=5
STRUCTURE_WORKERS=10
EVAL_WORKERS=10
STRUCTURE_MAX_RETRIES=3
```

Notes:
- `MODEL_NAME` is used for normal LLM calls.
- `STRUCTURED_MODEL_NAME` is used for structured-output calls (e.g., schema extraction).
- If you use a different provider, ensure `BASE_URL` and `API_KEY` match that provider.

### 3) Start dependencies
```bash
docker-compose up -d
```

## Run (Chainlit)
```bash
uv run chainlit run main.py --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## How It Works
1. **Router** determines whether you want resume review, JD writing, or comparison.
2. **OCR** extracts text from resumes.
3. **Structure** converts text to a strict JSON schema.
4. **Scoring** evaluates each candidate against your requirements.
5. **QA** enables database queries via natural language.

## Persian Output
Most prompts and UI responses are tuned for Persian (Farsi). If you need English output, update the prompts in `utils/prompt.py`.

## Troubleshooting
- If you see connection errors, verify MongoDB/MinIO are running and `.env` values are correct.
- If OCR is slow, reduce DPI or worker counts in `.env`.
- For structured-output errors, check your model supports JSON mode/structured outputs.
