from __future__ import annotations

import os
import uuid

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.embedding_service import generate_embeddings
from ..services.matcher import match_embedding
from ..services.media_pipeline import load_media
from ..services.vector_store import VectorStore
from ..utils.media import save_upload_to_temp


router = APIRouter(tags=["media"])
vector_store = VectorStore()


def _build_media_embedding(path: str) -> tuple[np.ndarray, int]:
    frames = load_media(path)
    embeddings = generate_embeddings(frames)
    embedding = embeddings.mean(axis=0)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32), len(frames)


@router.post("/upload-media")
def upload_media(file: UploadFile = File(...)) -> dict:
    temp_path = None
    try:
        temp_path, _ = save_upload_to_temp(file)
        embedding, frame_count = _build_media_embedding(temp_path)
        media_id = str(uuid.uuid4())
        vector_store.add_embedding(media_id, embedding)
        return {"id": media_id, "filename": file.filename, "frames_processed": frame_count}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/check-media")
def check_media(file: UploadFile = File(...)) -> dict:
    temp_path = None
    try:
        temp_path, _ = save_upload_to_temp(file)
        embedding, _ = _build_media_embedding(temp_path)
        return match_embedding(embedding, vector_store)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
