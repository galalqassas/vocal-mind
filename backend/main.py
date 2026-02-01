"""
FastAPI Application Entry Point.

This is the main entry point for the backend API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.rag.router import router as rag_router
from app.asr.router import router as asr_router
from app.emotion.router import router as emotion_router
from app.diarization.router import router as diarization_router

app = FastAPI(
    title="Vocal Mind API",
    description="Backend API for Vocal Mind - RAG, ASR, and more.",
    version="0.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(rag_router)
app.include_router(asr_router)
app.include_router(emotion_router)
app.include_router(diarization_router)


@app.get("/health")
async def health():
    """Global health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
