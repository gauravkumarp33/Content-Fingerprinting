from fastapi import FastAPI

from .routes.media_routes import router as media_router

app = FastAPI()
app.include_router(media_router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
