"""Structural mechanical-counter reader shared by the test and HTTP service.

It detects a complete row of eight large digit windows in grayscale before
falling back to the conservative OCR-seeded reader.
"""
from __future__ import annotations

import cv2
import numpy as np

from counter_reader import CounterReader, combine_votes, paddle_vote, vote
from counter_vision import proposals


def _seed_bounds(seeds):
    polygons = [np.asarray(seed["polygon"], np.float32) for seed in seeds
                if seed["kind"] == "polygon"]
    if not polygons:
        return None
    points = np.concatenate(polygons)
    return points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()


def _local_vote(observations):
    """Accept three agreeing moderate-confidence digit reads in this experiment."""
    normal = vote(observations)
    if normal[0] != "X":
        return normal
    numeric = [(text, score) for text, score in observations
               if len(text) == 1 and text.isdigit()]
    if len(numeric) == 3 and len({text for text, _ in numeric}) == 1:
        digit = numeric[0][0]
        confidence = min(float(score) for _, score in numeric)
        if confidence >= .70:
            return digit, confidence
    return "X", None


def _integral_windows(image, bounds, count):
    """Detect similarly sized dark drum panels in a horizontal seed band."""
    _, y0, _, y1 = bounds
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 61, 9)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    seed_height = max(1, y1 - y0)
    boxes = []
    for left, top, width, height, area in stats[1:]:
        ratio = width / max(height, 1)
        centre_y = top + height / 2
        if not (.42 <= ratio <= .85 and .55 * seed_height <= height <= 1.05 * seed_height):
            continue
        if abs(centre_y - (y0 + y1) / 2) > .55 * seed_height or area < height * 12:
            continue
        boxes.append((int(left), int(top), int(left + width), int(top + height)))
    boxes.sort()
    chains = []
    for box in boxes:
        if chains and box[0] - chains[-1][-1][2] < seed_height * .85:
            chains[-1].append(box)
        else:
            chains.append([box])
    candidates = [chain for chain in chains if len(chain) >= count]
    if not candidates:
        return []
    return max(candidates, key=lambda chain: sum(box[2] - box[0] for box in chain))[:count]


def _fractional_windows(image, integral, bounds, count):
    """Find one fractional panel after the integral drums using only edges."""
    _, y0, _, y1 = bounds
    typical = float(np.median([right - left for left, _, right, _ in integral]))
    after = integral[-1][2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top = max(0, round(y0 + (y1 - y0) * .16))
    bottom = min(gray.shape[0], round(y1 - (y1 - y0) * .16))
    profile = np.abs(cv2.Sobel(gray[top:bottom], cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)

    def strongest(left, right):
        left, right = max(1, round(left)), min(len(profile) - 2, round(right))
        return None if right <= left else left + int(np.argmax(profile[left:right + 1]))

    panel_left = strongest(after, after + typical * .7)
    if panel_left is None:
        return []
    panel_right = strongest(panel_left + typical * count * .75,
                            panel_left + typical * count * 1.5)
    if panel_right is None or panel_right - panel_left < typical * count * .65:
        return []
    edges = np.linspace(panel_left, panel_right, count + 1).round().astype(int)
    return [(int(edges[i]), round(y0), int(edges[i + 1]), round(y1)) for i in range(count)]


def detect_windows(image, seeds, integer_digits, decimal_places):
    bounds = _seed_bounds(seeds)
    if bounds is None:
        return []
    integral = _integral_windows(image, bounds, integer_digits)
    if len(integral) != integer_digits:
        return []
    fractional = _fractional_windows(image, integral, bounds, decimal_places)
    return integral + fractional if len(fractional) == decimal_places else []


def direct_digit_windows(image, count):
    """Find a complete row of large, regularly-spaced digit components.

    A serial number is text: its glyphs are normally much smaller and its
    spacing is irregular.  Mechanical counter windows create a distinctive
    sequence of eight tall, similarly sized components.  This deliberately
    works on grayscale threshold images, not red/black colour masks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height = gray.shape[0]
    candidates = []
    for block in (61, 101):
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, 9)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for left, top, width, height, area in stats[1:]:
            ratio = width / max(height, 1)
            if not (.045 * image_height <= height <= .16 * image_height):
                continue
            if not .25 <= ratio <= .9 or area < height * 9:
                continue
            boxes.append((int(left), int(top), int(left + width), int(top + height)))
        boxes.sort()
        for anchor in boxes:
            anchor_y = (anchor[1] + anchor[3]) / 2
            row = [box for box in boxes
                   if abs((box[1] + box[3]) / 2 - anchor_y) <= max(anchor[3] - anchor[1], box[3] - box[1]) * .22]
            row.sort()
            for start in range(len(row) - count + 1):
                chain = row[start:start + count]
                heights = np.asarray([bottom - top for _, top, _, bottom in chain], np.float32)
                centres = np.asarray([(left + right) / 2 for left, _, right, _ in chain], np.float32)
                pitches = np.diff(centres)
                median_height = float(np.median(heights))
                median_pitch = float(np.median(pitches))
                if median_pitch <= 0 or pitches.min() < median_height * .45 or pitches.max() > median_height * 1.55:
                    continue
                if heights.min() < median_height * .68 or heights.max() > median_height * 1.45:
                    continue
                pitch_cv = float(pitches.std() / max(median_pitch, 1))
                height_cv = float(heights.std() / max(median_height, 1))
                if pitch_cv > .20 or height_cv > .22:
                    continue
                # Height makes the real display outrank a smaller serial row.
                score = median_height * (1 - pitch_cv) * (1 - height_cv)
                candidates.append((score, chain))
    return max(candidates, key=lambda candidate: candidate[0])[1] if candidates else []


def display_frame_windows(image, seeds, count):
    """Use a long grayscale display frame when individual glyphs merge."""
    bounds = _seed_bounds(seeds)
    expected_y = (bounds[1] + bounds[3]) / 2 if bounds is not None else image.shape[0] / 2
    frames = []
    for left, top, right, bottom in proposals(image):
        width, height = right - left, bottom - top
        if height < image.shape[0] * .035 or not 3.5 <= width / max(height, 1) <= 8.0:
            continue
        distance = abs((top + bottom) / 2 - expected_y)
        if distance > height * 1.1:
            continue
        # Prefer a broad frame close to the numeric seed, never the seed
        # itself.  The seed may be only the first few digits.
        frames.append((width * height - distance * width * .5, left, top, right, bottom))
    if not frames:
        return []
    _, left, top, right, bottom = max(frames)
    edges = np.linspace(left, right, count + 1).round().astype(int)
    return [(int(edges[index]), int(top), int(edges[index + 1]), int(bottom))
            for index in range(count)]


def _cells(boxes, digits, sources):
    cells = []
    for (left, top, right, bottom), (digit, confidence), observations in zip(boxes, digits, sources):
        cells.append(dict(digit=digit, confidence=confidence,
                          polygon=[[left, top], [right, top], [right, bottom], [left, bottom]],
                          ocr_candidates=[dict(text=text, confidence=score, preprocessing=name)
                                          for name, values in observations.items()
                                          for text, score in values]))
    return cells


class DisplayCounterReader(CounterReader):
    """Structural reader with the previous reader retained as a fallback."""

    def read(self, image, seeds, digit_recognizer=None, debug=None):
        boxes = direct_digit_windows(image, self.integer_digits + self.decimal_places)
        if not boxes:
            boxes = display_frame_windows(image, seeds, self.integer_digits + self.decimal_places)
        if not boxes:
            rotated = next((seed for seed in seeds if seed["kind"] == "rotated"), None)
        else:
            rotated = None
        if rotated is not None:
            # A geometric proposal is already a complete rectangular display.
            # Do not allow the old periodic-grid search to begin three cells
            # before it: on steeply rotated photos that captures serial text.
            import counter_reader
            original_grids = counter_reader.grids

            def whole_display_grid(_image, box, count):
                x0, y0, x1, y1 = box
                return [(1.0, x0, max(1, round((x1 - x0) / count)), y0, y1)]

            counter_reader.grids = whole_display_grid
            try:
                result = super().read(image, [rotated], digit_recognizer, debug)
            finally:
                counter_reader.grids = original_grids
            result["method"] = "structural_fixed_geometric_display"
            return result
        if not boxes:
            boxes = detect_windows(image, seeds, self.integer_digits, self.decimal_places)
        if len(boxes) != self.integer_digits + self.decimal_places:
            # The experiment recognises one common layout with individually
            # visible black windows. Other layouts retain the previous local
            # reader rather than being downgraded to ``not_found``.
            return super().read(image, seeds, digit_recognizer, debug)
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        crops = [image[top:bottom, left:right] for left, top, right, bottom in boxes]
        all_obs, modes = {}, {}
        from counter_vision import clean
        for mode in ("adaptive", "otsu"):
            glyphs = [clean(crop, trim, mode) for trim in (.03, .12, .2) for crop in crops]
            canvas = np.concatenate(glyphs, axis=1)
            slots = [[i * 64, (i + 1) * 64, 0, 96] for i in range(len(glyphs))]
            results = self.reader.recognize(canvas, horizontal_list=slots, free_list=[],
                                            allowlist="0123456789", batch_size=32, detail=1)
            observations = [[(results[trial * len(crops) + index][1],
                              float(results[trial * len(crops) + index][2])) for trial in range(3)]
                            for index in range(len(crops))]
            all_obs[mode] = observations
            modes[mode] = [_local_vote(values) for values in observations]
        easy_digits = [combine_votes(modes["adaptive"][index], modes["otsu"][index],
                                     all_obs["adaptive"][index], all_obs["otsu"][index])
                       for index in range(len(crops))]
        candidates = [(easy_digits, all_obs, "easyocr")]
        if digit_recognizer is not None:
            prepared = []
            for crop in crops:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 31, 9)
                prepared.extend([crop, cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
                                 cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)])
            results = list(digit_recognizer.predict(prepared, batch_size=32))
            paddle = [[(results[index * 3 + trial]["rec_text"],
                        float(results[index * 3 + trial]["rec_score"])) for trial in range(3)]
                      for index in range(len(crops))]
            candidates.append(([paddle_vote(values) for values in paddle], {"paddle": paddle}, "paddle"))
        digits, sources, backend = max(candidates, key=lambda item: sum(digit != "X" for digit, _ in item[0]))
        text = "".join(digit for digit, _ in digits)
        return dict(value=text[:-self.decimal_places] + "." + text[-self.decimal_places:],
                    status="partial" if "X" in text else "recognized",
                    decimal_places=self.decimal_places, digits_count=len(digits),
                    cells=_cells(boxes, digits,
                                 [{name: values[index] for name, values in sources.items()}
                                  for index in range(len(boxes))]), error=None,
                    method="structural_per_window_grayscale_geometry_" + backend,
                    deskew_angle=0.0, preprocessing=backend)


# Compatibility name for already-created local test commands.
ExperimentalCounterReader = DisplayCounterReader
