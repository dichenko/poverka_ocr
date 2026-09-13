"""Local PaddleOCR plus conservative mechanical meter reading extraction."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import warnings

import numpy as np
from PIL import Image, ImageOps
from meter_reading import empty_reading, recognize_reading

ROOT = Path(__file__).resolve().parent
DETECTION_MODEL = "PP-OCRv5_mobile_det"
RECOGNITION_MODEL = "cyrillic_PP-OCRv5_mobile_rec"
DIGIT_MODEL = "en_PP-OCRv5_mobile_rec"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_image_files(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file()
                   and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.casefold())


def load_image(path: Path) -> np.ndarray:
    """Return contiguous BGR at EXIF-corrected resolution, without resizing."""
    with Image.open(path) as source:
        corrected = ImageOps.exif_transpose(source)
        if corrected.mode in ("RGBA", "LA", "P"):
            rgba = corrected.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            corrected = Image.alpha_composite(background, rgba)
        rgb = np.asarray(corrected.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])


def create_engine():
    # Must be configured before importing PaddleX/PaddleOCR.
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(ROOT / "models")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", message="No ccache found.*", category=UserWarning)
    from paddleocr import PaddleOCR
    for name in ("paddleocr", "paddlex"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return PaddleOCR(
        device="cpu", enable_mkldnn=False,
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=64, text_det_limit_type="min",
        text_rec_score_thresh=0.0,
    )


def run_ocr(engine, image: np.ndarray):
    return list(engine.predict(image))


def create_digit_recognizer():
    from paddleocr import TextRecognition
    return TextRecognition(model_name=DIGIT_MODEL, device="cpu", enable_mkldnn=False)


def normalize_ocr_result(results) -> list[dict]:
    items = []
    for result in results:
        texts, scores, polygons = result["rec_texts"], result["rec_scores"], result["rec_polys"]
        if not len(texts) == len(scores) == len(polygons):
            raise ValueError("Несогласованные массивы результата PaddleOCR")
        for text, score, polygon in zip(texts, scores, polygons):
            points = np.asarray(polygon, dtype=float)
            confidence = float(score)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                raise ValueError("Некорректный четырёхугольник PaddleOCR")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Некорректный confidence PaddleOCR")
            coordinates = points.tolist()
            items.append({"text": str(text), "confidence": confidence,
                          "polygon": coordinates,
                          "bbox": {"x1": float(points[:, 0].min()),
                                   "y1": float(points[:, 1].min()),
                                   "x2": float(points[:, 0].max()),
                                   "y2": float(points[:, 1].max())}})
    # Group neighbouring centres into approximate rows, then sort each row by x.
    rows = []
    for item in sorted(items, key=lambda i: (i["bbox"]["y1"], i["bbox"]["x1"])):
        box = item["bbox"]
        centre = (box["y1"] + box["y2"]) / 2
        height = max(1.0, box["y2"] - box["y1"])
        if rows and abs(centre - rows[-1][0]) <= 0.5 * min(height, rows[-1][1]):
            rows[-1][2].append(item)
        else:
            rows.append((centre, height, [item]))
    return [item for _, _, row in rows for item in sorted(row, key=lambda i: i["bbox"]["x1"])]


def save_json(path: Path, data: dict) -> None:
    """Atomic replacement prevents a partial JSON from being skipped next time."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".ocr-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_image(path: Path, engine, engine_info: dict, initialization_error=None,
                  digit_recognizer=None, digit_error=None) -> dict:
    started = perf_counter()
    data = {"source_file": path.name, "status": "error", "image": None,
            "ocr_engine": dict(engine_info), "full_text": [], "items": [],
            "items_count": 0, "processing_ms": 0, "error": None,
            "meter_reading": empty_reading("unavailable")}
    phase = "ImageReadError"
    try:
        image = load_image(path)
        data["image"] = {"width": int(image.shape[1]), "height": int(image.shape[0])}
        phase = "OCRInitializationError" if initialization_error else "OCRError"
        if initialization_error:
            raise RuntimeError(initialization_error)
        items = normalize_ocr_result(run_ocr(engine, image))
        data.update(status="success", items=items, items_count=len(items),
                    full_text=[item["text"] for item in items])
        try:
            if digit_error:
                raise RuntimeError(digit_error)
            if digit_recognizer is not None:
                data["meter_reading"] = recognize_reading(image, digit_recognizer, items)
                data["meter_reading"]["model"] = DIGIT_MODEL
        except Exception as exc:
            data["meter_reading"] = empty_reading("error", str(exc)[:1500])
    except Exception as exc:
        data["error"] = {"type": phase, "message": str(exc)[:1500] or type(exc).__name__}
    data["processing_ms"] = round((perf_counter() - started) * 1000)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Локальный OCR фотографий на CPU")
    parser.add_argument("--input", type=Path, default=ROOT / "input", help="Папка фотографий")
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="Папка JSON")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписать существующие JSON")
    args = parser.parse_args(argv)
    try:
        for folder in (ROOT / "input", ROOT / "output", ROOT / "models", args.input, args.output):
            folder.mkdir(parents=True, exist_ok=True)
        files = get_image_files(args.input)
    except OSError as exc:
        print(f"Ошибка доступа к папкам: {exc}", file=sys.stderr)
        return 2
    print(f"Найдено изображений: {len(files)}", flush=True)
    # Reject ambiguous output names rather than silently overwriting another photo.
    counts = Counter(p.stem.casefold() for p in files)
    if any(count > 1 for count in counts.values()):
        print("Ошибка: есть изображения с одинаковым именем без расширения. "
              "Дайте им разные имена, чтобы каждому соответствовал отдельный JSON.", file=sys.stderr)
        return 2
    pending = [p for p in files if args.overwrite or not (args.output / f"{p.stem}.json").exists()]
    try:
        installed_version = version("paddleocr")
    except PackageNotFoundError:
        installed_version = None
    info = {"name": "PaddleOCR", "version": installed_version,
            "model": RECOGNITION_MODEL, "detection_model": DETECTION_MODEL}
    engine, initialization_error = None, None
    digit_recognizer, digit_error = None, None
    if pending:
        print("Загрузка PaddleOCR (первый запуск может скачать модели)...", flush=True)
        try:
            with (ROOT / "models" / "download.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log):
                    engine = create_engine()
                    try:
                        digit_recognizer = create_digit_recognizer()
                    except Exception as exc:
                        digit_error = str(exc)
            print("Загружена модель PaddleOCR", flush=True)
        except Exception as exc:
            initialization_error = str(exc) or type(exc).__name__
            print(f"Ошибка загрузки PaddleOCR: {initialization_error}", file=sys.stderr)
    success = errors = skipped = 0
    for index, path in enumerate(files, 1):
        target = args.output / f"{path.stem}.json"
        prefix = f"[{index}/{len(files)}] {path.name}"
        if target.exists() and not args.overwrite:
            skipped += 1
            print(f"{prefix} — пропущено", flush=True)
            continue
        data = process_image(path, engine, info, initialization_error, digit_recognizer, digit_error)
        try:
            save_json(target, data)
        except OSError as exc:
            errors += 1
            print(f"{prefix} — ERROR — не удалось записать JSON: {exc}", flush=True)
            continue
        if data["status"] == "success":
            success += 1
            print(f"{prefix} — OK — найдено строк: {data['items_count']} — "
                  f"{data['processing_ms'] / 1000:.2f} сек. — показания: "
                  f"{data['meter_reading']['value'] or data['meter_reading']['status']}", flush=True)
        else:
            errors += 1
            print(f"{prefix} — ERROR — {data['error']['type']}: {data['error']['message']}", flush=True)
    print(f"Готово: успешно {success}, ошибок {errors}, пропущено {skipped}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
