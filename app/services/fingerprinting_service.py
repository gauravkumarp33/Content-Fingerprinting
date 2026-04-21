from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List

import faiss
import numpy as np

from services.embedding_service import generate_embedding
from utils.media import extract_frames


@dataclass
class MediaRecord:
    media_id: int
    filename: str
    media_type: str


class MediaFingerprintingService:
    _instance: "MediaFingerprintingService | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.records: List[MediaRecord] = []
        self.index_lock = Lock()

    @classmethod
    def get_instance(cls) -> "MediaFingerprintingService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_media(self, path: str, filename: str) -> Dict:
        embedding = self._build_embedding(path)
        with self.index_lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(int(embedding.shape[1]))
            self.index.add(embedding)
            media_id = len(self.records) + 1
            record = MediaRecord(
                media_id=media_id,
                filename=filename,
                media_type=self._detect_media_type(filename),
            )
            self.records.append(record)

        return {
            "media_id": media_id,
            "filename": filename,
            "media_type": record.media_type,
            "indexed_items": len(self.records),
        }

    def search_media(self, path: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or not self.records:
            return []

        query_embedding = self._build_embedding(path)
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.records)))
        matches: List[Dict] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = self.records[int(index)]
            matches.append(
                {
                    "media_id": record.media_id,
                    "filename": record.filename,
                    "media_type": record.media_type,
                    "score": float(score),
                }
            )

        return matches

    def _build_embedding(self, path: str) -> np.ndarray:
        suffix = Path(path).suffix.lower()
        frames = extract_frames(path, suffix)
        frame_embeddings = np.stack([generate_embedding(frame) for frame in frames], axis=0)
        embedding = frame_embeddings.mean(axis=0, keepdims=True)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.astype(np.float32)

    @staticmethod
    def _detect_media_type(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return "image" if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else "video"
