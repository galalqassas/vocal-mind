"""Pydantic models for RAG API."""
from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    text: str = Field(..., description="Question to ask")


class SourceNode(BaseModel):
    rank: int
    score: float | None
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode]
    timing: dict[str, float]


class IngestRequest(BaseModel):
    force: bool = False


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    nodes_created: int


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
