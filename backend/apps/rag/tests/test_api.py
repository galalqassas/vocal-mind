from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from apps.rag.router import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    with patch("apps.rag.router.get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.pc.list_indexes.return_value = []
        response = client.get("/rag/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "pinecone" in data["components"]


def test_query_endpoint():
    """Test query endpoint with mocked engine."""
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="Test answer")
    mock_response.source_nodes = []
    mock_response.metadata = {"retrieval_seconds": 0.1, "synthesis_seconds": 0.2}
    
    with patch("apps.rag.router.get_engine") as mock_engine:
        mock_engine.return_value.query.return_value = mock_response
        response = client.post("/rag/query", json={"text": "test query"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert "sources" in data
        assert "timing" in data


def test_ingest_endpoint():
    """Test ingest endpoint with mocked pipeline."""
    with patch("apps.rag.router.get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.load_documents.return_value = [Mock()]
        mock_pipeline.return_value.run.return_value = Mock()
        
        response = client.post("/rag/ingest", json={"force": False})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["documents_processed"] > 0
