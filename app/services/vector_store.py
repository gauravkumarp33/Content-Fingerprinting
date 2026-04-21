from __future__ import annotations

from typing import List

import faiss
import numpy as np


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatL2 | None = None
        self.ids: List[str] = []

    def add_embedding(self, id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vector.shape[1])
        self.index.add(vector)
        self.ids.append(id)

    def search(self, vector: np.ndarray, k: int):
        if self.index is None or not self.ids:
            return []

        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(vector, min(k, len(self.ids)))
        return [
            {"id": self.ids[index], "distance": float(distance)}
            for distance, index in zip(distances[0], indices[0])
            if index >= 0
        ]
