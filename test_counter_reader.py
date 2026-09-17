"""Offline regression checks; no OCR model loading or network access."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from counter_reader import (CounterReader, combine_votes, locate_counter_rows,
                            paddle_vote, vote, vote_strength)
from counter_vision import angles, clean, grids, rotate
from experimental_counter_reader import direct_digit_windows
from test_readings_only import compare_reading, expected_from_name


class CounterTests(unittest.TestCase):
    def test_consensus_needs_repeated_support(self):
        self.assertEqual(vote([('7', .999), ('8', .6), ('5', .4)])[0], 'X')
        self.assertEqual(vote([('7', .999), ('7', .6), ('5', .4)])[0], '7')

    def test_conflicting_strong_predictions_are_unknown(self):
        self.assertEqual(vote([('7', .999), ('1', .99), ('7', .95)])[0], 'X')

    def test_paddle_digit_fallback_accepts_repeated_and_alias_glyphs(self):
        self.assertEqual(paddle_vote([('I', .82), ('I', .81), ('x', .1)])[0], '1')
        self.assertEqual(paddle_vote([('4,', .80), ('b', .14), ('4', .97)])[0], '4')
        self.assertEqual(paddle_vote([('7', .80), ('2', .75), ('x', .2)])[0], 'X')

    def test_confirmed_alternate_fills_strict_primary_unknown(self):
        # Regression: a visually clear 3 was found by Otsu three times at 1.0,
        # while adaptive preprocessing produced three lower-confidence 3s.
        self.assertEqual(combine_votes(('X', None), ('3', 1.0)), ('3', 1.0))
        self.assertEqual(combine_votes(('4', .99), ('3', 1.0)), ('4', .99))

    def test_per_cell_repeated_evidence_can_override_row_preference(self):
        # Regression: two neighbouring drums were visibly 4 and 6.  The row
        # preference had selected a 4, while the alternate variant read 6
        # twice with greater combined support.
        primary=[('4', .858), ('4', .941), ('8', .578)]
        alternate=[('8', .678), ('6', .900), ('6', .933)]
        self.assertGreater(vote_strength(alternate, '6'), vote_strength(primary, '4'))
        self.assertEqual(combine_votes(('4', .941), ('6', .933), primary, alternate), ('6', .933))

    def test_repeated_moderate_or_multichar_predictions(self):
        self.assertEqual(vote([('7', .8)] * 3)[0], '7')
        self.assertEqual(vote([('17', .999)] * 3)[0], 'X')

    def test_missing_leading_wheels_are_counted(self):
        verdict = compare_reading('263.145', 'X0263.14X')
        self.assertEqual(verdict['unknown_digits'], 2)
        self.assertTrue(verdict['passed'])
        self.assertFalse(compare_reading('263.145', 'XX263.1XX')['passed'])

    def test_wrong_digits_never_pass(self):
        self.assertFalse(compare_reading('134.952', '00134.95X')['wrong_digits'])
        self.assertFalse(compare_reading('134.952', '00134.91X')['passed'])
        self.assertFalse(compare_reading('134.952', None)['passed'])

    def test_names_are_optional_reference_labels(self):
        self.assertEqual(expected_from_name(Path('00134_952.jpg')), '134.952')
        with self.assertRaises(ValueError):
            expected_from_name(Path('01.jpg'))

    def test_blank_and_empty_crops(self):
        for crop in (np.full((100, 60, 3), 255, np.uint8), np.empty((0, 0, 3), np.uint8)):
            for mode in ('otsu', 'adaptive'):
                self.assertTrue(np.all(clean(crop, threshold=mode) == 255))
        self.assertEqual(grids(np.zeros((30, 40, 3), np.uint8), [0, 0, 0, 0]), [])


    def test_rotation_coordinate_roundtrip(self):
        image = np.zeros((100, 180, 3), np.uint8)
        turned, matrix = rotate(image, -48)
        self.assertGreater(turned.shape[0], image.shape[0])
        points = np.array([[[0., 0.], [90., 50.], [179., 99.]]], np.float32)
        roundtrip = cv2.transform(cv2.transform(points, matrix), cv2.invertAffineTransform(matrix))
        np.testing.assert_allclose(roundtrip, points, atol=1e-4)

    def test_angle_detection_accepts_both_opencv_line_layouts(self):
        image = np.zeros((120, 160, 3), np.uint8)
        lines = np.array([[10, 10, 110, 10], [10, 20, 110, 20]], np.int32)
        with patch('counter_vision.cv2.HoughLinesP', return_value=lines):
            self.assertIn(0, angles(image))
        with patch('counter_vision.cv2.HoughLinesP', return_value=lines[:, None, :]):
            self.assertIn(0, angles(image))

    def test_direct_digit_windows_prefers_a_regular_large_row(self):
        image = np.full((600, 1000, 3), 255, np.uint8)
        for index in range(8):
            cv2.rectangle(image, (60 + index * 90, 200),
                          (110 + index * 90, 290), (0, 0, 0), -1)
        boxes = direct_digit_windows(image, 8)
        self.assertEqual(len(boxes), 8)
        self.assertEqual(boxes[0][0], 60)
        self.assertEqual(boxes[-1][2], 741)

    def test_no_proposal_does_not_fabricate_digits(self):
        reader = CounterReader.__new__(CounterReader)
        reader.integer_digits, reader.decimal_places = 5, 3
        reader.reader = Mock()
        result = reader.read(np.full((100, 300, 3), 255, np.uint8), [])
        self.assertIsNone(result['value'])
        reader.reader.recognize.assert_not_called()

    def test_existing_numeric_ocr_row_is_used_without_colour(self):
        image = np.full((500, 900, 3), 255, np.uint8)
        items = [{'text': '00123456',
                  'polygon': [[100, 120], [700, 120], [700, 220], [100, 220]]}]
        seeds = locate_counter_rows(image, items)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]['kind'], 'polygon')
        self.assertNotIn('text', seeds[0])


if __name__ == '__main__':
    unittest.main()
