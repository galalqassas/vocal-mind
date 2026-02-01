"""ASR Data Models."""
from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    # For now, we might accept a file upload or a path/url
    # If file upload, this model might not be used directly in the body
    language: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[dict] = []
