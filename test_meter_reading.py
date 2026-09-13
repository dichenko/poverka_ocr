import unittest
from unittest.mock import patch
import cv2
import numpy as np
from meter_reading import recognize_reading, detect_cells


class ReadingTests(unittest.TestCase):
    def test_unknown_positions_and_decimal(self):
        image = np.full((100, 500, 3), 255, np.uint8)
        polygons = [np.array([[i*60,0],[i*60+50,0],[i*60+50,90],[i*60,90]], np.float32) for i in range(7)]
        class Recognizer:
            def __init__(self):
                self.index = 0
            def predict(self, images):
                text = ['0','1','2','3','4','5','?'][self.index]
                self.index += 1
                return [{'rec_text':text,'rec_score':.99} for _ in images]
        with patch('meter_reading.detect_cells', return_value=polygons):
            result = recognize_reading(image, Recognizer(), [])
        self.assertEqual(result['value'], '0123.45X')
        self.assertEqual(result['digits_count'], 7)
        self.assertEqual(result['status'], 'partial')
        self.assertIsNone(result['cells'][-1]['confidence'])

    def test_empty_is_not_fabricated_reading(self):
        result = recognize_reading(np.full((200,300,3),255,np.uint8), None, [])
        self.assertIsNone(result['value'])
        self.assertIsNone(result['digits_count'])
        self.assertEqual(result['status'], 'not_found')

    def test_conflicting_digits_are_unknown(self):
        polygons = [np.array([[i*60,0],[i*60+50,0],[i*60+50,90],[i*60,90]], np.float32) for i in range(7)]
        class ConflictingRecognizer:
            def predict(self, images):
                return [{'rec_text':str(i%2),'rec_score':.99} for i in range(len(images))]
        with patch('meter_reading.detect_cells', return_value=polygons):
            result = recognize_reading(np.full((100,500,3),255,np.uint8), ConflictingRecognizer(), [])
        self.assertEqual(result['value'], 'XXXX.XXX')

    def test_reading_failure_keeps_raw_ocr(self):
        import main
        from test_main import FakeEngine
        with patch.object(main, 'load_image', return_value=np.full((100,300,3),255,np.uint8)), patch.object(main, 'recognize_reading', side_effect=RuntimeError('test')):
            result = main.process_image(__import__('pathlib').Path('photo.png'), FakeEngine(), {}, digit_recognizer=object())
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['items_count'], 2)
        self.assertEqual(result['meter_reading']['status'], 'error')

    def test_automatic_count(self):
        for count in (7, 8, 9):
            image = np.full((400,1000,3),220,np.uint8)
            for i in range(count):
                x=70+i*65
                cv2.rectangle(image,(x,120),(x+53,215),(245,245,245),-1)
                cv2.rectangle(image,(x,120),(x+53,215),(80,80,80),2)
                cv2.putText(image,'8',(x+6,195),cv2.FONT_HERSHEY_SIMPLEX,2.1,
                            (0,0,190) if i>=count-3 else (15,15,15),3)
            item={'text':'88888','polygon':[[65,112],[70+count*65,112],[70+count*65,225],[65,225]]}
            self.assertEqual(len(detect_cells(image,[item])),count)


if __name__ == '__main__':
    unittest.main()
