import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.fingerprinting_service import MediaFingerprintingService
from utils.media import save_upload_to_temp


router = APIRouter(tags=["media"])
service = MediaFingerprintingService.get_instance()


@router.post("/media/index")
def index_media(file: UploadFile = File(...)) -> dict:
    temp_path = None
    try:
        temp_path, _ = save_upload_to_temp(file)
        return service.add_media(temp_path, file.filename or "uploaded_media")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/media/search")
def search_media(file: UploadFile = File(...)) -> dict:
    temp_path = None
    try:
        temp_path, _ = save_upload_to_temp(file)
        return {"matches": service.search_media(temp_path, top_k=5)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
