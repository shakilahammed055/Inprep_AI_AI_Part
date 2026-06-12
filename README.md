# Inprep AI — Backend API

A FastAPI-based REST API for the Inprep AI platform.

## Project Structure

```
.
├── app/
│   ├── main.py                     # FastAPI app entry point
│   ├── api/
│   │   └── v1/
│   │       ├── api.py              # API v1 router aggregator
│   │       └── endpoints/
│   │           └── items.py        # Items CRUD endpoints
│   └── schemas/
│       └── items.py                # Pydantic models for items
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables (not committed)
```

## Requirements

- Python 3.12+
- Dependencies listed in [requirements.txt](requirements.txt):
  - `fastapi`
  - `uvicorn`
  - `asyncpg`
  - `greenlet`
  - `pydantic_core`

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI credentials:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

## Running the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/v1/models` | List suggested OpenAI models |
| POST | `/api/v1/chat/completions` | OpenAI-compatible chat completion endpoint |
| POST | `/api/v1/resume/upload` | Upload a PDF resume and generate an intro |
| POST | `/api/v1/resume/manual-introduction` | Submit manual profile data and generate an intro |
| POST | `/api/v1/items/` | Create a new item |
| GET | `/api/v1/items/` | List all items |
| GET | `/api/v1/items/{item_id}` | Get a single item by ID |

Interactive docs (Swagger UI) are available at `http://localhost:8000/docs`.

## Item Schema

```json
{
  "name": "string",
  "description": "string (optional)",
  "price": 0.0
}
```
