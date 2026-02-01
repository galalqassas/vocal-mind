from typing import List
from llama_index.core import Document, SimpleDirectoryReader
from apps.core.config import settings
from apps.rag.ingestion.base import BaseIngestionStrategy


class StandardIngestionStrategy(BaseIngestionStrategy):
    """Standard directory ingestion."""

    def load_documents(self) -> List[Document]:
        print(f"Loading documents from: {settings.DATA_DIR}")
        if not settings.DATA_DIR.exists():
            print(f"Directory {settings.DATA_DIR} does not exist.")
            return []

        reader = SimpleDirectoryReader(
            input_dir=str(settings.DATA_DIR),
            recursive=True,
        )
        return reader.load_data()
