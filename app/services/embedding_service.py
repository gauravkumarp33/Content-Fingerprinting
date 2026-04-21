from __future__ import annotations

import hashlib
from threading import Lock
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class EmbeddingService:
    _instance: "EmbeddingService | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()
        self.cache: dict[str, np.ndarray] = {}
        self.cache_lock = Lock()

    def _cache_key(self, image: np.ndarray) -> str:
        return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def generate_embedding(self, image: np.ndarray) -> np.ndarray:
        return self.generate_embeddings([image])[0]

    def generate_embeddings(self, images: Sequence[np.ndarray]) -> np.ndarray:
        cached_embeddings: list[np.ndarray | None] = []
        uncached_images: list[Image.Image] = []
        uncached_keys: list[str] = []

        for image in images:
            key = self._cache_key(image)
            with self.cache_lock:
                cached_embedding = self.cache.get(key)
            if cached_embedding is not None:
                cached_embeddings.append(cached_embedding)
                continue
            cached_embeddings.append(None)
            uncached_keys.append(key)
            uncached_images.append(Image.fromarray(image[:, :, ::-1]))

        if uncached_images:
            inputs = self.processor(images=uncached_images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            features = features / features.norm(dim=-1, keepdim=True)
            uncached_embeddings = features.cpu().numpy().astype(np.float32)
            with self.cache_lock:
                for key, embedding in zip(uncached_keys, uncached_embeddings):
                    self.cache[key] = embedding

        embeddings: list[np.ndarray] = []
        uncached_index = 0
        for cached_embedding in cached_embeddings:
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                continue
            embeddings.append(self.cache[uncached_keys[uncached_index]])
            uncached_index += 1

        return np.stack(embeddings, axis=0)


def generate_embedding(image: np.ndarray) -> np.ndarray:
    return EmbeddingService.get_instance().generate_embedding(image)


def generate_embeddings(images: Sequence[np.ndarray]) -> np.ndarray:
    return EmbeddingService.get_instance().generate_embeddings(images)
