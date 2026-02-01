import pytest
import sys
from pathlib import Path

# Fix import path for 'app' which seems to be mapped to 'apps' in production
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_json_data():
    """Generic nested JSON for testing extraction."""
    return {
        "slack": [
            {
                "id": "msg_1",
                "Message": {"User": {"userId": "eid_123", "text": "Hello world"}},
                "ThreadReplies": [],
            }
        ],
        "documents": [
            {
                "id": "doc_1",
                "title": "Technical Spec",
                "content": "This is a test document.",
            }
        ],
    }


@pytest.fixture
def employee_meta():
    """Sample metadata for ID resolution."""
    return [
        {"id": "eid_123", "name": "Alice Smith"},
        {"id": "eid_456", "name": "Bob Jones"},
    ]
