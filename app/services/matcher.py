from __future__ import annotations

import numpy as np

from .vector_store import VectorStore


def match_embedding(embedding: np.ndarray, store: VectorStore) -> dict:
    matches = store.search(embedding, k=5)
    results = []

    for match in matches:
        score = max(0.0, 1.0 - (match["distance"] / 2.0))
        if score > 0.9:
            label = "exact"
        elif score >= 0.75:
            label = "similar"
        else:
            label = "no_match"

        results.append(
            {
                "id": match["id"],
                "score": score,
                "distance": match["distance"],
                "label": label,
            }
        )

    if not results:
        return {"match_type": "no_match", "matches": []}

    return {"match_type": results[0]["label"], "matches": results}
