"""Emotion API Router."""
from fastapi import APIRouter, UploadFile, File

from app.emotion.models import EmotionResponse
from app.emotion.engine import EmotionEngine

router = APIRouter(prefix="/emotion", tags=["Emotion"])
engine = EmotionEngine()

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "Emotion"}

@router.post("/analyze", response_model=EmotionResponse)
async def analyze(file: UploadFile = File(...)):
    # In a real app, save file to temp or process stream
    result = engine.analyze(file.filename)
    return EmotionResponse(**result)
