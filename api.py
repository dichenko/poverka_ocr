"""HTTP service for the local meter OCR pipeline."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
import logging
import os
from pathlib import PurePath
import secrets
import threading

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import main


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_BYTES", 15 * 1024 * 1024)
OCR_WORKERS = _positive_int("OCR_WORKERS", 1)
MAX_INFLIGHT_REQUESTS = _positive_int("MAX_INFLIGHT_REQUESTS", 100)
OCR_API_KEY = os.getenv("OCR_API_KEY", "")


class OCRService:
    """One warmed model set with a bounded FIFO executor queue."""

    def __init__(self) -> None:
        self._slots = threading.BoundedSemaphore(MAX_INFLIGHT_REQUESTS)
        self._executor = ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="ocr")
        self.engine = self.digit_recognizer = None
        self.initialization_error = self.digit_error = None
        try:
            installed_version = version("paddleocr")
        except PackageNotFoundError:
            installed_version = None
        self.engine_info = {"name": "PaddleOCR", "version": installed_version,
                            "model": main.RECOGNITION_MODEL,
                            "detection_model": main.DETECTION_MODEL}

    def start(self) -> None:
        try:
            self.engine = main.create_engine()
            try:
                self.digit_recognizer = main.create_digit_recognizer()
            except Exception as exc:  # General OCR can still serve responses.
                self.digit_error = str(exc)
        except Exception as exc:
            self.initialization_error = str(exc) or type(exc).__name__
            logging.exception("OCR model initialization failed")

    @property
    def ready(self) -> bool:
        return self.engine is not None and self.initialization_error is None

    def submit(self, content: bytes, filename: str):
        if not self._slots.acquire(blocking=False):
            return None
        future = self._executor.submit(
            main.process_image_bytes, content, filename, self.engine, self.engine_info,
            self.initialization_error, self.digit_recognizer, self.digit_error,
        )
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def stop(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


service = OCRService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.start()
    yield
    service.stop()


app = FastAPI(title="Meter OCR API", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    if not service.ready:
        return JSONResponse(status_code=503, content={"status": "not_ready",
                                                        "error": service.initialization_error})
    return {"status": "ok", "ocr_workers": OCR_WORKERS,
            "max_inflight_requests": MAX_INFLIGHT_REQUESTS}


@app.post("/ocr")
async def recognize(request: Request, file: UploadFile = File(...)):
    supplied_key = request.headers.get("X-API-Key", "")
    if not OCR_API_KEY or not secrets.compare_digest(supplied_key, OCR_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid OCR API key")
    filename = PurePath((file.filename or "upload").replace("\\", "/")).name
    if not filename.lower().endswith(tuple(main.EXTENSIONS)):
        raise HTTPException(status_code=415, detail="Supported formats: jpg, jpeg, png, webp, bmp")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_BYTES} bytes")
    if not service.ready:
        raise HTTPException(status_code=503, detail="OCR models are not ready")
    future = service.submit(content, filename)
    if future is None:
        raise HTTPException(status_code=429, detail="OCR queue is full; retry later", headers={"Retry-After": "5"})
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        # The inference continues safely in its worker; its slot is released by the callback.
        raise
    finally:
        await file.close()
