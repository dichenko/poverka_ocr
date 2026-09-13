"""Validate existing smoke-test JSON; exercise skip/overwrite without network."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "verification" / "input"
OUTPUT = ROOT / "verification" / "output"


def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def run(overwrite=False):
    # Block outbound sockets in the actual inference process, even outside a sandbox.
    code = (
        "import socket,runpy; "
        "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled during verification')); "
        "runpy.run_path('main.py',run_name='__main__')"
    )
    command = [sys.executable, "-c", code, "--input", str(INPUT), "--output", str(OUTPUT)]
    if overwrite:
        command.append("--overwrite")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8",
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    print(result.stdout)
    if result.returncode not in (0, 1):
        raise AssertionError(result.stderr)
    return result


def main():
    before_input, before_output = hashes(INPUT), hashes(OUTPUT)
    times = {p.name: p.stat().st_mtime_ns for p in OUTPUT.glob("*.json")}
    skipped = run()
    assert skipped.returncode == 0
    assert before_output == hashes(OUTPUT)
    assert times == {p.name: p.stat().st_mtime_ns for p in OUTPUT.glob("*.json")}
    overwritten = run(True)
    assert overwritten.returncode == 1  # Deliberately broken JPEG.
    assert all(p.stat().st_mtime_ns != times[p.name] for p in OUTPUT.glob("*.json"))
    assert before_input == hashes(INPUT)
    data = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTPUT.glob("*.json"))]
    assert len(data) == 6
    for record in data:
        assert record["ocr_engine"]["version"] == "3.7.0"
        assert record["items_count"] == len(record["items"]) == len(record["full_text"])
        assert record["full_text"] == [i["text"] for i in record["items"]]
        if record["source_file"] == "broken.jpg":
            assert record["status"] == "error" and record["error"]["type"] == "ImageReadError"
            continue
        assert record["status"] == "success", record
        for item in record["items"]:
            assert 0 <= item["confidence"] <= 1
            assert len(item["polygon"]) == 4
            for x, y in item["polygon"]:
                assert 0 <= x <= record["image"]["width"]
                assert 0 <= y <= record["image"]["height"]
        if record["source_file"] == "blank.png":
            assert record["items_count"] == 0
    for p in INPUT.glob("*_photo_*.jpg"):
        original = ROOT.parent / "dataset" / p.name
        assert hashlib.sha256(original.read_bytes()).hexdigest() == before_input[p.name]
    print("PASS: 6 JSON valid; skip, overwrite, offline inference, original hashes verified.")
    print(next(r["full_text"] for r in data if r["source_file"] == "контроль.png"))


if __name__ == "__main__":
    main()
