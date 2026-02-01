"""HERB dataset specific ingestion strategy."""
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from llama_index.core import Document
from apps.core.config import settings
from apps.rag.ingestion.base import BaseIngestionStrategy

logger = logging.getLogger(__name__)

# Pre-compiled regex for performance
RE_EID = re.compile(r"eid_[a-f0-9]+")
RE_CUST = re.compile(r"CUST-\d+")


@lru_cache(maxsize=1)
def load_metadata_cache(metadata_dir: Path) -> tuple[Dict[str, dict], Dict[str, dict]]:
    """Load and cache metadata to avoid redundant disk I/O."""
    employees = {}
    customers = {}

    try:
        emp_file = metadata_dir / "employee.json"
        if emp_file.exists():
            with open(emp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                employees = {e["id"]: e for e in data} if isinstance(data, list) else data

        cust_file = metadata_dir / "customers_data.json"
        if cust_file.exists():
            with open(cust_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                customers = {c["id"]: c for c in data} if isinstance(data, list) else data
                
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        # Non-critical, continue with empty metadata

    return employees, customers


class HERBDataLoader:
    """Loader for HERB enterprise dataset structure."""

    def __init__(self, data_root: Path):
        self.products_dir = data_root / "products"
        self._employees, self._customers = load_metadata_cache(data_root / "metadata")

    def _resolve(self, mapping: Dict[str, dict], key: str, name_field: str = "name") -> str:
        """Resolve ID to name generically."""
        item = mapping.get(key, {})
        name = item.get(name_field)
        return f"{name} ({key})" if name else key

    def _enrich_text(self, text: str) -> str:
        """Replace IDs with names in text."""
        if not text:
            return ""
        text = RE_EID.sub(lambda m: self._resolve(self._employees, m.group(0)), text)
        text = RE_CUST.sub(lambda m: self._resolve(self._customers, m.group(0)), text)
        return text

    def _parse_item(self, func, item: dict) -> tuple[str, str] | None:
        try:
            return func(item)
        except Exception as e:
            logger.warning(f"Error parsing item {item.get('id', '?')}: {e}")
            return None

    def _parse_slack(self, msg: dict) -> tuple[str, str]:
        user_info = msg.get("Message", {}).get("User", {})
        sender = self._resolve(self._employees, user_info.get("userId", "unknown"))
        text = self._enrich_text(user_info.get("text", ""))
        
        replies = [
            f"  - {self._resolve(self._employees, r.get('User', {}).get('userId'))}: "
            f"{self._enrich_text(r.get('User', {}).get('text', ''))}"
            for r in msg.get("ThreadReplies", [])
        ]
        
        content = f"[Slack] {sender}: {text}"
        if replies:
            content += "\nReplies:\n" + "\n".join(replies)
            
        return content, f"Slack Thread: {sender} - {text[:30]}..."

    def _parse_meeting(self, meeting: dict) -> tuple[str, str]:
        if "transcript" in meeting:
            return (
                f"[Meeting Transcript]\n{self._enrich_text(meeting['transcript'])}",
                f"Meeting Transcript - {meeting.get('id', 'unknown')}"
            )
        
        messages = meeting.get("messages", [])
        if messages:
            chat = [
                f"{self._resolve(self._employees, m.get('sender'))}: {self._enrich_text(m.get('text'))}"
                for m in messages
            ]
            return (
                "[Meeting Chat]\n" + "\n".join(chat),
                f"Meeting Chat - {meeting.get('id', 'unknown')}"
            )
            
        return f"[Meeting Data] {json.dumps(meeting)}", "Meeting Data"

    def _parse_doc(self, doc: dict) -> tuple[str, str]:
        title = doc.get("title", "Untitled")
        return f"[Document: {title}]\n{self._enrich_text(doc.get('content', ''))}", title

    def _parse_pr(self, pr: dict) -> tuple[str, str]:
        author = self._resolve(self._employees, pr.get("author", "unknown"))
        desc = self._enrich_text(pr.get("description", ""))
        title = pr.get("title", "Untitled PR")
        
        comments = [
            f"  - {self._resolve(self._employees, c.get('author'))}: {self._enrich_text(c.get('body'))}"
            for c in pr.get("comments", [])
        ]
        
        content = f"[PR: {title}]\nAuthor: {author}\n{desc}\n"
        if comments:
            content += "Comments:\n" + "\n".join(comments)
            
        return content, f"PR: {title}"

    def load_product(self, product_file: Path) -> List[Document]:
        try:
            with open(product_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {product_file}: {e}")
            return []

        documents = []
        parsers = {
            "slack": (self._parse_slack, "slack"),
            "meeting_transcripts": (self._parse_meeting, "meeting"),
            "meeting_chats": (self._parse_meeting, "meeting"),
            "meetings": (self._parse_meeting, "meeting"),
            "documents": (self._parse_doc, "document"),
            "pull_requests": (self._parse_pr, "pull_request"),
        }

        product_name = product_file.stem
        for key, (func, doc_type) in parsers.items():
            for item in data.get(key, []):
                result = self._parse_item(func, item)
                if result:
                    content, title = result
                    documents.append(Document(
                        text=content,
                        metadata={
                            "source": "herb",
                            "product": product_name,
                            "type": doc_type,
                            "id": item.get("id", "unknown"),
                            "title": title,
                        }
                    ))
        return documents

    def load_all(self) -> List[Document]:
        if not self.products_dir.exists():
            logger.warning(f"HERB products directory {self.products_dir} not found.")
            return []

        all_docs = []
        for p_file in self.products_dir.glob("*.json"):
            docs = self.load_product(p_file)
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} docs from {p_file.name}")
            
        return all_docs


class HERBIngestionStrategy(BaseIngestionStrategy):
    """Ingests data from specific HERB dataset structure."""

    def load_documents(self) -> List[Document]:
        logger.info(f"Loading HERB dataset from: {settings.DATA_DIR}")
        return HERBDataLoader(settings.DATA_DIR).load_all()
