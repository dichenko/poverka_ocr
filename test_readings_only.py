"""Benchmark only the mechanical-meter reading against file-name labels.

Optional reference file names have the form ``integer_fraction.jpg``; for example
``136_729.jpg`` represents 136.729.  The script leaves the normal ``output``
folder untouched and writes a compact JSON report plus optional visual crops.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from main import (DIGIT_MODEL, DETECTION_MODEL, RECOGNITION_MODEL, ROOT,
                  create_digit_recognizer, create_engine, load_image,
                  normalize_ocr_result, run_ocr)
from meter_reading import recognize_reading, crop_cell

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def expected_from_name(path: Path) -> str:
    """Convert ``30_682.jpg`` to the canonical reading ``30.682``."""
    try:
        integer, fraction = path.stem.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError("ожидается имя вида 30_682.jpg") from exc
    if not integer.isdigit() or not fraction.isdigit() or len(fraction) != 3:
        raise ValueError("целая часть должна состоять из цифр, дробная — из трёх цифр")
    return f"{int(integer)}.{fraction}"


def _compare_part(expected: str, actual: str, right_align: bool) -> tuple[int, int]:
    """Return (unknown, wrong), tolerating harmless leading zero counter wheels."""
    unknown = wrong = 0
    if right_align:
        extra, actual = actual[:-len(expected)] if len(actual) > len(expected) else "", actual[-len(expected):]
        unknown += extra.count("X")
        wrong += sum(char not in "0X" for char in extra)
        actual = actual.rjust(len(expected), "X")
    else:
        extra, actual = actual[len(expected):], actual[:len(expected)]
        wrong += len(extra)
        actual = actual.ljust(len(expected), "X")
    for wanted, received in zip(expected, actual):
        if received == "X":
            unknown += 1
        elif received != wanted:
            wrong += 1
    return unknown, wrong


def compare_reading(expected: str, actual: str | None) -> dict:
    """Compare readings without penalising extra leading zero wheels."""
    if not actual or "." not in actual:
        return {"unknown_digits": len(expected.replace(".", "")), "wrong_digits": 0,
                "passed": False}
    expected_integer, expected_fraction = expected.split(".", 1)
    actual_integer, actual_fraction = actual.split(".", 1)
    expected_integer = expected_integer.lstrip("0") or "0"
    unknown_i, wrong_i = _compare_part(expected_integer, actual_integer, right_align=True)
    unknown_f, wrong_f = _compare_part(expected_fraction, actual_fraction, right_align=False)
    unknown, wrong = unknown_i + unknown_f, wrong_i + wrong_f
    return {"unknown_digits": unknown, "wrong_digits": wrong,
            "passed": wrong == 0 and unknown <= 2}


def save_debug_image(image, reading: dict, target: Path) -> None:
    canvas = image.copy()
    for index, cell in enumerate(reading.get("cells", []), 1):
        polygon = np.asarray(cell["polygon"], dtype="int32")
        cv2.polylines(canvas, [polygon], True, (0, 220, 0), 3)
        point = tuple(polygon[0])
        cv2.putText(canvas, f"{index}:{cell['digit']}", point, cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(target.parent / f"{target.stem}_cell-{index:02d}.png"),
                    crop_cell(image, cell["polygon"]))
    cv2.imwrite(str(target), canvas)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Тест только распознавания показаний счётчика")
    parser.add_argument("--input", type=Path, default=ROOT / "input", help="Папка эталонных фото")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "reading_test_output",
                        help="Папка JSON-отчёта и визуальных вырезок")
    parser.add_argument("--no-debug-images", action="store_true", help="Не сохранять изображения с контурами")
    parser.add_argument("--require-labels", action="store_true", help="Требовать эталон в имени каждого фото")
    parser.add_argument("--backend", choices=("grayscale", "legacy"), default="grayscale")
    parser.add_argument("--seed-python", type=Path, default=ROOT / ".venv" / "Scripts" / "python.exe",
                        help="Python с установленным PaddleOCR для поиска строки")
    parser.add_argument("--integer-digits", type=int, default=5)
    args = parser.parse_args(argv)
    if args.integer_digits < 1:
        parser.error("--integer-digits должен быть положительным")
    if not args.input.is_dir():
        parser.error(f"Нет папки: {args.input}")
    files = sorted((p for p in args.input.iterdir() if p.suffix.lower() in EXTENSIONS),
                   key=lambda path: path.name.casefold())
    if not files:
        parser.error(f"В папке нет изображений: {args.input}")
    labels = {}
    for file in files:
        try:
            labels[file] = expected_from_name(file)
        except ValueError as exc:
            if args.require_labels:
                parser.error(f"{file.name}: {exc}")
            labels[file] = None
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    total_started = perf_counter()
    print("Загрузка OCR-моделей…", flush=True)
    seeds = None
    if args.backend == "grayscale":
        try:
            from counter_reader import CounterReader
            reader = CounterReader(ROOT / "models" / "easyocr", args.integer_digits,
                                   download_models=True)
            digit_recognizer = create_digit_recognizer()
        except (ImportError, FileNotFoundError) as exc:
            parser.error(f"Не готово окружение распознавания: {exc}. См. README; запускайте test_readings.bat")
        if not args.seed_python.is_file():
            parser.error(f"Не найден Python для PaddleOCR: {args.seed_python}")
        proposals_file = args.debug_dir / "row_proposals.json"
        with (args.debug_dir / "localization.log").open("w", encoding="utf8") as log:
            completed = subprocess.run([str(args.seed_python.resolve()), "-X", "utf8",
                str(ROOT / "reading_seeds.py"), "--input", str(args.input.resolve()),
                "--output", str(proposals_file.resolve())], cwd=ROOT, stdout=log, stderr=log)
        if completed.returncode:
            parser.error(f"Ошибка поиска строки: см. {args.debug_dir / 'localization.log'}")
        seeds = json.loads(proposals_file.read_text(encoding="utf8"))
    else:
        with (args.debug_dir / "localization.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log):
                engine, digit_recognizer = create_engine(), create_digit_recognizer()
    setup_seconds = round(perf_counter() - total_started, 2)
    report = []
    for file in files:
        started = perf_counter()
        image = None
        try:
            image = load_image(file)
            if seeds is not None:
                if isinstance(seeds[file.name], dict) and "error" in seeds[file.name]:
                    raise ValueError(seeds[file.name]["error"])
                reading = reader.read(image, seeds[file.name], digit_recognizer)
            else:
                items = normalize_ocr_result(run_ocr(engine, image))
                reading = recognize_reading(image, digit_recognizer, items)
        except Exception as exc:
            from meter_reading import empty_reading
            reading = empty_reading()
            reading.update(status="error", error=str(exc))
        expected = labels[file]
        verdict = compare_reading(expected, reading["value"]) if expected else {
            "unknown_digits": reading["value"].count("X") if reading["value"] else None,
            "wrong_digits": None, "passed": None}
        elapsed_ms = round((perf_counter() - started) * 1000)
        row = {"file": file.name, "expected": expected, "actual": reading["value"],
               "recognition_status": reading["status"], "error": reading.get("error"),
               "processing_ms": elapsed_ms, **verdict}
        report.append(row)
        mark = "RUN" if expected is None else "OK" if verdict["passed"] else "FAIL"
        print(f"{mark:4} {file.name:16} expected={expected or '-':8} actual={reading['value'] or '-':10} "
              f"unknown={verdict['unknown_digits']} wrong={verdict['wrong_digits']} ({elapsed_ms} ms)", flush=True)
        if not args.no_debug_images and image is not None:
            save_debug_image(image, reading, args.debug_dir / f"{file.name}_debug.jpg")
        (args.debug_dir / f"{file.name}.json").write_text(json.dumps(reading, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"total": len(report), "passed": sum(row["passed"] is True for row in report),
               "failed": sum(row["passed"] is False for row in report),
               "unlabelled": sum(row["passed"] is None for row in report),
               "errors": sum(row["recognition_status"] == "error" for row in report),
               "backend": args.backend, "setup_and_localization_seconds": setup_seconds,
               "total_seconds": round(perf_counter() - total_started, 2),
               "criterion": "0 wrong digits and no more than 2 X digits", "results": report,
               "models": {"detection": DETECTION_MODEL, "recognition": RECOGNITION_MODEL,
                          "digits": "EasyOCR english_g2 + PaddleOCR digit fallback" if seeds is not None else DIGIT_MODEL}}
    (args.debug_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    labelled = summary['total'] - summary['unlabelled']
    print(f"Проверено по эталонам: {labelled}; OK: {summary['passed']}; FAIL: {summary['failed']}; "
          f"без эталона: {summary['unlabelled']}; ошибок: {summary['errors']}. "
          f"Всего {summary['total_seconds']} с. Отчёт: {args.debug_dir / 'report.json'}")
    return 0 if summary["failed"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
