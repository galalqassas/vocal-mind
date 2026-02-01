"""RAG Query Engine - Retrieval and synthesis with timing."""
import json
import time
from datetime import datetime
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq

from apps.core.config import settings


class RAGQueryEngine:
    """RAG query engine with retrieval, synthesis, and logging."""

    def __init__(self, index: VectorStoreIndex) -> None:
        self.index = index
        self.logs_dir = settings.BASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup LLM
        self.llm = Groq(
            model=settings.groq.model,
            api_key=settings.groq.api_key.get_secret_value(),
            temperature=settings.groq.temperature,
            max_tokens=settings.groq.max_tokens,
            context_window=settings.groq.context_window,
        )
        
        # Setup retriever and synthesizer
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=settings.similarity_top_k)
        self.synthesizer = get_response_synthesizer(llm=self.llm, response_mode=settings.response_mode)

    def query(self, question: str) -> Any:
        """Query the RAG system with timing."""
        t0 = time.perf_counter()
        
        # Retrieve
        nodes = self.retriever.retrieve(question)
        retrieval_time = time.perf_counter() - t0
        
        # Synthesize
        t1 = time.perf_counter()
        response = self.synthesizer.synthesize(question, nodes=nodes)
        synthesis_time = time.perf_counter() - t1
        total_time = time.perf_counter() - t0
        
        # Add timing to response metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            "retrieval_seconds": retrieval_time,
            "synthesis_seconds": synthesis_time,
            "total_seconds": total_time,
        })
        
        # Log query
        self._log(question, nodes, str(response), retrieval_time, synthesis_time, total_time)
        return response

    def _log(self, question: str, nodes: list, answer: str, ret_t: float, syn_t: float, tot_t: float) -> None:
        """Log query to JSON file."""
        log_file = self.logs_dir / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        chunks = [{"rank": i, "score": n.score, "text": n.text[:200], "metadata": n.metadata} for i, n in enumerate(nodes, 1)]
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "model": settings.groq.model,
                    "timing": {"retrieval": ret_t, "synthesis": syn_t, "total": tot_t},
                    "chunks": chunks,
                    "answer": answer,
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Silent fail for logging
