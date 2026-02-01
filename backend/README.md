# Vocal Mind Backend

Backend API for the Vocal Mind project, providing RAG (Retrieval-Augmented Generation) and future speech processing capabilities.

## Architecture

This project uses a **Modular Monolith** architecture:

- Single deployable service
- Strict separation by domain (`app/rag`, `app/asr`, etc.)
- Shared core utilities in `app/core`

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama running locally (for embeddings)
- Pinecone account (for vector storage)
- Groq API key (for LLM)

### Setup

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Configure environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the server**:

   ```bash
   uv run uvicorn main:app --reload --port 8000
   ```

4. **Access the API**:
   - Swagger UI: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## API Endpoints

### Global

| Endpoint  | Method | Description         |
| --------- | ------ | ------------------- |
| `/health` | GET    | Global health check |

### RAG Module (`/rag`)

| Endpoint      | Method | Description                            |
| ------------- | ------ | -------------------------------------- |
| `/rag/health` | GET    | RAG-specific health (Pinecone, Ollama) |
| `/rag/query`  | POST   | Query the RAG system                   |
| `/rag/ingest` | POST   | Trigger document ingestion             |

## Project Structure

```
backend/
├── main.py              # FastAPI application entry point
├── app/
│   ├── core/
│   │   └── config.py    # Environment settings
│   └── rag/
│       ├── router.py    # API endpoints
│       ├── models.py    # Pydantic schemas
│       ├── engine.py    # Query logic
│       ├── pipeline.py  # Ingestion logic
│       └── ingestion/   # Data loaders
```

## Environment Variables

| Variable              | Description                                         |
| --------------------- | --------------------------------------------------- |
| `GROQ_API_KEY`        | Groq API key for LLM                                |
| `PINECONE_API_KEY`    | Pinecone API key                                    |
| `PINECONE_INDEX_NAME` | Pinecone index name                                 |
| `OLLAMA_BASE_URL`     | Ollama server URL (default: http://localhost:11434) |

## Development

```bash
# Run with auto-reload
uv run uvicorn main:app --reload --port 8000

# Lint code
uv run ruff check .

# Format code
uv run ruff format .
```
