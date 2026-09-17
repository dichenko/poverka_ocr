"""Offline contract checks; real model checks are recorded in verification/REPORT.md."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import main


class FakeEngine:
    def predict(self, image):
        yield {"rec_texts": ["№ A-123/45", "СЧЁТЧИК"],
               "rec_scores": np.array([0.8, 0.95], dtype=np.float32),
               "rec_polys": np.array([[[5, 30], [25, 30], [25, 40], [5, 40]],
                                       [[5, 5], [30, 5], [30, 15], [5, 15]]])}


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photo = self.root / "счётчик.png"
        Image.new("RGB", (80, 60), "red").save(self.photo)

    def tearDown(self):
        self.temp.cleanup()

    def test_exif_and_bgr(self):
        exif = Image.Exif()
        exif[274] = 6
        photo = self.root / "поворот.jpg"
        Image.new("RGB", (80, 60), "red").save(photo, exif=exif)
        image = main.load_image(photo)
        self.assertEqual(image.shape, (80, 60, 3))
        self.assertGreater(int(image[0, 0, 2]), 240)
        self.assertLess(int(image[0, 0, 0]), 10)

    def test_formats_and_nonrecursive_discovery(self):
        for extension in main.EXTENSIONS:
            Image.new("RGB", (40, 40), "white").save(self.root / ("test" + extension))
            self.assertEqual(main.load_image(self.root / ("test" + extension)).shape, (40, 40, 3))
        (self.root / "nested").mkdir()
        Image.new("RGB", (40, 40)).save(self.root / "nested" / "hidden.png")
        self.assertEqual(len(main.get_image_files(self.root)), 6)

    def test_result_and_json(self):
        data = main.process_image(self.photo, FakeEngine(), {"version": "test"})
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["full_text"], ["СЧЁТЧИК", "№ A-123/45"])
        self.assertEqual(data["items_count"], 2)
        self.assertEqual(data["items"][0]["bbox"], {"x1": 5, "y1": 5, "x2": 30, "y2": 15})
        output = self.root / "result.json"
        main.save_json(output, data)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), data)
        self.assertIn("СЧЁТЧИК", output.read_text(encoding="utf-8"))

    def test_in_memory_upload_has_the_same_contract(self):
        data = main.process_image_bytes(self.photo.read_bytes(), "upload.png", FakeEngine(),
                                        {"version": "test"})
        self.assertEqual(data["source_file"], "upload.png")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["image"], {"width": 80, "height": 60})

    def test_new_counter_reader_preserves_general_ocr_contract(self):
        class Reader:
            def read(self, image, seeds, digit_recognizer=None):
                self.seeds = seeds
                return {'value': '00123.456', 'status': 'recognized', 'cells': [],
                        'decimal_places': 3, 'digits_count': 8, 'error': None}
        reader = Reader()
        seeds = [{'kind': 'polygon', 'polygon': [[1, 1], [2, 1], [2, 2], [1, 2]],
                  'width': 1.0, 'height': 1.0}]
        with patch.object(main, 'locate_counter_rows', return_value=seeds) as locator:
            result = main.process_image(self.photo, FakeEngine(), {'version': 'test'},
                                        digit_recognizer=object(), counter_reader=reader)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['full_text'], ['СЧЁТЧИК', '№ A-123/45'])
        self.assertEqual(result['meter_reading']['value'], '00123.456')
        self.assertEqual(result['meter_reading']['model'], 'EasyOCR english_g2 + PaddleOCR digit fallback')
        locator.assert_called_once()
        self.assertEqual(reader.seeds, seeds)

    def test_compact_api_response_filters_noise_and_metadata(self):
        data = {
            "items": [
                {"text": "СЧЁТЧИК"}, {"text": "A"}, {"text": "Я"},
                {"text": "Q=1,5"}, {"text": "  00123.456  "}, {"text": "7"},
            ],
            "meter_reading": {"value": "00123.456", "status": "recognized", "cells": [1]},
            "error": {"message": "hidden"},
        }
        self.assertEqual(main.compact_api_response(data), {
            "ocr": ["СЧЁТЧИК", "00123.456", "7"],
            "meter_reading": "00123.456",
        })
        self.assertEqual(main.compact_api_response({"items": [], "meter_reading": {"value": None}}),
                         {"ocr": [], "meter_reading": None})

    def test_errors_and_empty(self):
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not a photo")
        data = main.process_image(broken, FakeEngine(), {})
        self.assertEqual(data["error"]["type"], "ImageReadError")
        self.assertIsNone(data["image"])
        with patch.object(FakeEngine, "predict", side_effect=RuntimeError("test failure")):
            self.assertEqual(main.process_image(self.photo, FakeEngine(), {})["error"]["type"], "OCRError")
        with patch.object(FakeEngine, "predict", return_value=[{"rec_texts": [], "rec_scores": [], "rec_polys": []}]):
            data = main.process_image(self.photo, FakeEngine(), {})
            self.assertEqual((data["status"], data["items_count"]), ("success", 0))

    def test_skip_overwrite_and_continue(self):
        output = self.root / "out"
        (self.root / "broken.jpg").write_bytes(b"broken")
        args = ["--input", str(self.root), "--output", str(output)]
        with patch.object(main, "ROOT", self.root), patch.object(main, "create_digit_recognizer", return_value=None), patch.object(main, "create_engine", return_value=FakeEngine()) as factory:
            self.assertEqual(main.main(args), 1)
            self.assertEqual(factory.call_count, 1)
            saved = (output / "счётчик.json").read_bytes()
            self.assertEqual(main.main(args), 0)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual((output / "счётчик.json").read_bytes(), saved)
            self.assertEqual(main.main(args + ["--overwrite"]), 1)
            self.assertEqual(factory.call_count, 2)

    def test_malformed_result_rejected(self):
        with self.assertRaises(ValueError):
            main.normalize_ocr_result([{"rec_texts": ["x"], "rec_scores": [], "rec_polys": []}])


if __name__ == "__main__":
    unittest.main()
