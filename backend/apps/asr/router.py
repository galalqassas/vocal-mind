"""ASR API Router."""
from fastapi import APIRouter, UploadFile, File

from app.asr.models import TranscribeResponse
from app.asr.engine import ASREngine

router = APIRouter(prefix="/asr", tags=["ASR"])
engine = ASREngine()

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "ASR"}

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    # In a real app, save file to temp or process stream
    result = engine.transcribe(file.filename)
    return TranscribeResponse(**result)
