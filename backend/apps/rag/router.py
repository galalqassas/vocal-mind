"""RAG API Router."""
import logging
from fastapi import APIRouter, HTTPException

from apps.core.config import settings
from apps.rag.models import QueryRequest, QueryResponse, SourceNode, IngestRequest, IngestResponse, HealthResponse
from apps.rag.pipeline import DocumentIngestionPipeline
from apps.rag.engine import RAGQueryEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

# Singletons
_pipeline: DocumentIngestionPipeline | None = None
_engine: RAGQueryEngine | None = None


def get_engine() -> RAGQueryEngine:
    global _pipeline, _engine
    if _engine is None:
        _pipeline = _pipeline or DocumentIngestionPipeline()
        _engine = RAGQueryEngine(_pipeline.get_index())
    return _engine


def get_pipeline() -> DocumentIngestionPipeline:
    global _pipeline
    _pipeline = _pipeline or DocumentIngestionPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        get_pipeline().pc.list_indexes()
        status = "connected"
    except Exception as e:
        status = f"error: {str(e)[:50]}"
    return HealthResponse(
        status="ok" if status == "connected" else "degraded",
        components={"pinecone": status, "ollama": settings.embedding.base_url},
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = get_engine().query(request.text)
        sources = [
            SourceNode(rank=i, score=n.score, text=n.text, metadata=n.metadata)
            for i, n in enumerate(getattr(response, "source_nodes", []), 1)
        ]
        timing = response.metadata or {}
        return QueryResponse(
            answer=str(response),
            sources=sources,
            timing={k: timing.get(k, 0) for k in ["retrieval_seconds", "synthesis_seconds", "total_seconds"]},
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    try:
        global _engine
        _engine = None  # Reset to pick up new documents
        
        pipeline = get_pipeline()
        docs = pipeline.load_documents()
        if not docs:
            return IngestResponse(status="no_documents", documents_processed=0, nodes_created=0)
        
        pipeline.run(force_reindex=request.force, documents=docs)
        return IngestResponse(status="completed", documents_processed=len(docs), nodes_created=len(docs))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
