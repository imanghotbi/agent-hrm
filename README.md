# Agent HRM (Hiring-Only Resume Review)

This project now runs a **hiring-only** workflow:
- Collect hiring requirements (including scoring weights)
- Read resumes from a **local folder**
- OCR + structure + evaluate candidates
- Save candidates in MongoDB
- Record total review duration from requirement finalization to process end

The workflow is CLI-based (no Chainlit/UI routing flow).

## Architecture Overview
- **CLI runtime (`main.py`)**: interactive terminal session
- **LangGraph**: orchestration and checkpointing
- **MongoDB**:
  - candidate storage
  - token usage logs
  - graph checkpoints for restart/recovery

## Requirements
- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- Docker + docker-compose (MongoDB)

## Setup

### 1) Install dependencies
```bash
uv pip install -r requirements.txt
```

### 2) Configure environment variables
Create `.env` from `.env-example` and fill values:

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

# Resume source
RESUME_SOURCE_DIR=./resumes
```

### 3) Start MongoDB
```bash
docker-compose up -d mongo
```

## Run
```bash
uv run python main.py --resume-dir ./resumes
```

Optional flags:
- `--new-session`: force a new session even if an unfinished one exists
- `--session-id <id>`: resume a specific session

## Full Usage Flow
1. Put all candidate resumes (`.pdf`) into your resume folder (for example `./resumes`).
2. Run:
```bash
uv run python main.py --resume-dir ./resumes
```
3. The assistant asks hiring requirement questions in terminal.
4. Answer until requirements are complete (including weights/scores).
5. Review starts automatically:
   - reads all PDFs from the folder
   - OCR -> structure -> evaluate
   - saves candidates to MongoDB
6. At completion, result summary is written to:
   - `runtime/review_result_<session_id>.json`

## Restart Safety
- LangGraph state is checkpointed in MongoDB (`MongoDBSaver`)
- Runtime metadata is persisted in `runtime/run_state.json`
- Final output is persisted as `runtime/review_result_<session_id>.json`

If a crash/error occurs, rerun `main.py` and it resumes from the latest checkpoint by default.

## Start Fresh / Clear State
### Fresh run without deleting history (recommended)
```bash
uv run python main.py --new-session --resume-dir ./resumes
```

### Clear local runtime state files
```bash
rm -f runtime/run_state.json
rm -f runtime/review_result_*.json
```

### Clear MongoDB checkpoints (hard reset)
Open Mongo shell:
```bash
docker exec -it mongo_server mongosh -u root -p example --authenticationDatabase admin
```

Then inspect checkpoint DB/collections:
```javascript
show dbs
use langgraph
show collections
```

Delete one session:
```javascript
db.checkpoints.deleteMany({"thread_id":"session_xxx"})
db.checkpoint_writes.deleteMany({"thread_id":"session_xxx"})
```

Delete all checkpoint entries:
```javascript
db.checkpoints.deleteMany({})
db.checkpoint_writes.deleteMany({})
```

## What Is Logged
- App/node operational logs: console + rotating file log (`logs/app.log`)
- LLM token usage logs: Mongo usage collection (`MONGO_DB_USAGE`)
- Runtime progress snapshot: `runtime/run_state.json`
- Final review artifact: `runtime/review_result_<session_id>.json`
- Evaluated candidates: Mongo resume collection (`MONGO_COLLECTION`)

Notes:
- Not every raw user line is stored as a dedicated audit record.
- Checkpoint payloads are stored in MongoDB, not as local JSON dumps.
