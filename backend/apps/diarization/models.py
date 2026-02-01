"""Diarization Data Models."""
from pydantic import BaseModel


class SpeakerSegment(BaseModel):
    speaker: str
    start: float
    end: float


class DiarizationResponse(BaseModel):
    segments: list[SpeakerSegment]
