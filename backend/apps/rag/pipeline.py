"""Document ingestion pipeline."""
import logging
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from apps.core.config import settings
from apps.rag.ingestion import get_ingestion_strategy

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for processing and indexing documents."""

    def __init__(self) -> None:
        self.strategy = get_ingestion_strategy()
        
        # Initialize services
        self.embed_model = OllamaEmbedding(
            model_name=settings.embedding.model,
            base_url=settings.embedding.base_url,
        )
        self.pc = Pinecone(api_key=settings.pinecone.api_key.get_secret_value())
        
        # Transformations
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )

        # Cache for index existence check
        self._index_checked = False

    def _ensure_index(self) -> None:
        """Ensure Pinecone index exists (checked once)."""
        if self._index_checked:
            return

        try:
            indexes = [i.name for i in self.pc.list_indexes()]
            if settings.pinecone.index_name not in indexes:
                logger.info(f"Creating Pinecone index: {settings.pinecone.index_name}")
                self.pc.create_index(
                    name=settings.pinecone.index_name,
                    dimension=settings.pinecone.dimension,
                    metric=settings.pinecone.metric,
                    spec=ServerlessSpec(
                        cloud=settings.pinecone.cloud,
                        region=settings.pinecone.region,
                    ),
                )
            self._index_checked = True
        except Exception as e:
            if "ALREADY_EXISTS" not in str(e):
                logger.error(f"Failed to ensure index: {e}")
                raise e

    def get_index(self) -> VectorStoreIndex:
        """Get vector store index."""
        self._ensure_index()
        return VectorStoreIndex.from_vector_store(
            PineconeVectorStore(pinecone_index=self.pc.Index(settings.pinecone.index_name)),
            embed_model=self.embed_model,
        )

    def load_documents(self) -> list[Document]:
        """Load documents via strategy."""
        return self.strategy.load_documents()

    def run(self, force_reindex: bool = False, documents: list[Document] | None = None) -> VectorStoreIndex:
        """Run ingestion pipeline."""
        self._ensure_index()
        pinecone_index = self.pc.Index(settings.pinecone.index_name)
        
        # Load docs
        if documents is None:
            documents = self.load_documents()
        if not documents:
            logger.warning("No documents to ingest")
            return self.get_index()

        logger.info(f"Ingesting {len(documents)} documents...")

        # Run pipeline
        pipeline = IngestionPipeline(
            transformations=[self.splitter, self.embed_model],
            vector_store=PineconeVectorStore(pinecone_index=pinecone_index),
        )

        nodes = pipeline.run(documents=documents, show_progress=True)
        logger.info(f"Successfully ingested {len(nodes)} nodes")

        return VectorStoreIndex.from_vector_store(
            pipeline.vector_store, embed_model=self.embed_model
        )
