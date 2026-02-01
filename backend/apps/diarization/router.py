"""Diarization API Router."""
from fastapi import APIRouter, UploadFile, File

from app.diarization.models import DiarizationResponse
from app.diarization.engine import DiarizationEngine

router = APIRouter(prefix="/diarization", tags=["Diarization"])
engine = DiarizationEngine()

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "Diarization"}

@router.post("/process", response_model=DiarizationResponse)
async def process(file: UploadFile = File(...)):
    # In a real app, save file to temp or process stream
    result = engine.process(file.filename)
    return DiarizationResponse(**result)
