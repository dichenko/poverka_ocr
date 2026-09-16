"""Service wiring checks without loading OCR models or opening a socket."""
import unittest
from unittest.mock import patch

try:
    import api
except ModuleNotFoundError as exc:
    if exc.name != 'fastapi':
        raise
    api = None


@unittest.skipIf(api is None, 'FastAPI is installed by the Docker/production dependency set')
class APIServiceTests(unittest.TestCase):
    def test_warmed_counter_reader_is_passed_to_http_worker(self):
        engine, digit_model, counter_reader = object(), object(), object()
        service = api.OCRService()
        try:
            with patch.object(api.main, 'create_engine', return_value=engine), \
                 patch.object(api.main, 'create_digit_recognizer', return_value=digit_model), \
                 patch.object(api, 'CounterReader', return_value=counter_reader), \
                 patch.object(api.main, 'process_image_bytes', return_value={'status': 'success'}) as process:
                service.start()
                future = service.submit(b'image', 'meter.jpg')
                self.assertEqual(future.result(timeout=2), {'status': 'success'})
            self.assertIs(service.counter_reader, counter_reader)
            self.assertIsNone(service.counter_error)
            process.assert_called_once_with(
                b'image', 'meter.jpg', engine, service.engine_info, None, digit_model, None,
                counter_reader, None)
        finally:
            service.stop()

    def test_counter_initialization_failure_does_not_disable_text_ocr(self):
        service = api.OCRService()
        try:
            with patch.object(api.main, 'create_engine', return_value=object()), \
                 patch.object(api.main, 'create_digit_recognizer', return_value=object()), \
                 patch.object(api, 'CounterReader', side_effect=RuntimeError('model unavailable')):
                service.start()
            self.assertTrue(service.ready)
            self.assertIsNone(service.counter_reader)
            self.assertEqual(service.counter_error, 'model unavailable')
        finally:
            service.stop()


if __name__ == '__main__':
    unittest.main()
