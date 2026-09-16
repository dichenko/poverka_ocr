"""Conservative geometric drum-window detection and per-cell recognition."""
from __future__ import annotations

import cv2
import numpy as np


def _periodic_correlation(signal, lag):
    """Return normalized correlation between a signal and its shifted copy."""
    left, right = signal[..., :-lag], signal[..., lag:]
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float((left * right).sum() / denominator) if denominator > 1e-6 else -1.0


def _detect_achromatic_cells(image, items):
    """Find a complete periodic row when all drums are achromatic.

    This fallback deliberately uses the unexpanded OCR polygon: unlike the colour
    path, it expects OCR to have found the complete digit row. Requiring agreement
    between OCR digit count, geometric count and two image correlations prevents
    ordinary long numbers from being accepted too easily.
    """
    proposals = []
    # A decimal separator in a long numeric OCR token is a very strong cue for
    # the counter window.  In particular, it keeps a clearly read value from
    # losing to the (also periodic) serial number printed lower on the body.
    decimal_numeric_items = [item for item in items
                             if any(mark in item["text"] for mark in ".,")
                             and sum(char in "0123456789" for char in item["text"]) >= 5]
    candidates = decimal_numeric_items or items
    for item in candidates:
        digits_count = sum(character in "0123456789" for character in item["text"])
        if digits_count < 5:
            continue
        polygon = np.asarray(item["polygon"], np.float32)
        width = float(np.linalg.norm(polygon[1] - polygon[0]))
        height = float(np.linalg.norm(polygon[3] - polygon[0]))
        if width / max(height, 1) < 3.0 or height < image.shape[0] * .025:
            continue
        band = crop_cell(image, polygon)
        h, w = band.shape[:2]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
        central = gray[int(h*.12):max(int(h*.88), int(h*.12)+2)]
        if central.shape[1] < 5:
            continue
        illumination = cv2.GaussianBlur(central, (0, 0), max(2.0, h*.25), 0)
        high_pass = central - illumination
        edges = np.abs(cv2.Sobel(central, cv2.CV_32F, 1, 0, ksize=3))
        edges -= cv2.GaussianBlur(edges, (0, 0), max(2.0, h*.18), 0)
        minimum_lag = max(8, round(h*.32))
        maximum_lag = min(w//2-1, round(h*.95))
        for lag in range(minimum_lag, maximum_lag + 1):
            count_float = w / lag
            count = round(count_float)
            if not 5 <= count <= 15 or abs(count_float-count) > .28:
                continue
            if abs(digits_count-count) > 1:
                continue
            gray_correlation = _periodic_correlation(high_pass, lag)
            edge_correlation = _periodic_correlation(edges, lag)
            if gray_correlation < .24 and edge_correlation < .32:
                continue
            # A real drum row normally repeats in both brightness and vertical edges.
            if gray_correlation < .12 or edge_correlation < .18:
                continue
            source = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
            inverse = cv2.getPerspectiveTransform(source, polygon)
            cell_width = w / count
            polygons = []
            for index in range(count):
                left = index * cell_width
                right = (index + 1) * cell_width
                rect = np.array([[left, h*.04], [right, h*.04],
                                 [right, h*.96], [left, h*.96]], np.float32)
                cell = cv2.perspectiveTransform(rect[None], inverse)[0]
                if (cell[:, 0].min() < 0 or cell[:, 1].min() < 0 or
                    cell[:, 0].max() >= image.shape[1] or cell[:, 1].max() >= image.shape[0]):
                    polygons = []
                    break
                polygons.append(cell)
            if polygons:
                count_score = 1.0 - abs(count_float-count)/.28
                score = max(gray_correlation, edge_correlation) + .35*min(gray_correlation, edge_correlation) + .15*count_score
                proposals.append((score, polygons))
    if proposals:
        return max(proposals, key=lambda proposal: proposal[0])[1]
    # OCR can already outline a clean value while the weak separators between
    # its black drums are invisible to the periodicity test.  A decimal numeric
    # token is specific enough to use its own evenly spaced cells as a final
    # fallback; this path is intentionally unavailable to serial-number text.
    for item in decimal_numeric_items:
        count = sum(character in "0123456789" for character in item["text"])
        if not 5 <= count <= 12:
            continue
        polygon = np.asarray(item["polygon"], np.float32)
        source_width = max(8, round(np.linalg.norm(polygon[1] - polygon[0])))
        source_height = max(8, round(np.linalg.norm(polygon[3] - polygon[0])))
        source = np.array([[0, 0], [source_width - 1, 0],
                           [source_width - 1, source_height - 1], [0, source_height - 1]], np.float32)
        inverse = cv2.getPerspectiveTransform(source, polygon)
        cells = []
        for index in range(count):
            left, right = index * source_width / count, (index + 1) * source_width / count
            rect = np.array([[left, source_height * .04], [right, source_height * .04],
                             [right, source_height * .96], [left, source_height * .96]], np.float32)
            cells.append(cv2.perspectiveTransform(rect[None], inverse)[0])
        return cells
    return []


def detect_cells(image, items):
    """Locate periodic drum windows, anchored by three red fractional drums.

    Count integer cells from measured periodicity, never from OCR text length.
    A missing colour/geometry cue returns no field rather than a guessed count.
    """
    from itertools import combinations
    proposals = []
    geometry_only_proposals = []
    for item in items:
        p = np.asarray(item["polygon"], np.float32)
        width = float(np.linalg.norm(p[1] - p[0]))
        height = float(np.linalg.norm(p[3] - p[0]))
        if width / max(height, 1) < 2.5 or height < image.shape[0] * .025:
            continue
        digits_count = sum(c in "0123456789" for c in item["text"])
        geometry_only = digits_count < 3
        # PaddleOCR occasionally sees a complete, strongly tilted drum row as
        # one letter. Permit such a candidate only under strict geometric rules.
        if geometry_only and not (len(item["text"].strip()) <= 2 and
                                  width / max(height, 1) >= 3.0 and
                                  height >= image.shape[0] * .08):
            continue
        # The raw detector may omit the red digits; search beyond its right edge.
        vector = p[1] - p[0]
        p[[0, 3]] -= vector * .10
        p[[1, 2]] += vector * .65
        band = crop_cell(image, p)
        h, w = band.shape[:2]
        _, green, red = cv2.split(band.astype(float))
        chroma = (red - green) / (red + green + 1)
        chroma -= np.median(chroma)
        profile = np.maximum(chroma - .005, 0)[int(h*.12):int(h*.85)].sum(axis=0).astype(np.float32)
        profile = cv2.GaussianBlur(profile[None], (0, 0), max(1, h*.05))[0]
        peaks = [x for x in range(1, w-1) if profile[x] > profile[x-1]
                 and profile[x] >= profile[x+1] and profile[x] > .3]
        # Limit noisy proposals and require three similarly spaced colour peaks.
        peaks = sorted(sorted(peaks, key=lambda x: profile[x], reverse=True)[:30])
        offset = width * .10
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(float)
        for triple in combinations(peaks, 3):
            gaps = np.diff(triple)
            pitch = float(np.mean(gaps))
            if not .32*h < pitch < .95*h or abs(gaps[0]-gaps[1]) > pitch*.23:
                continue
            if not offset + width*.45 < triple[0] < offset + width + pitch*.5:
                continue
            trace = gray[int(h*.18):int(h*.82), int(offset):triple[0]].mean(axis=0)
            if len(trace) < pitch*2.5:
                continue
            trace -= cv2.GaussianBlur(trace[None], (0, 0), pitch)[0]
            matches = []
            for lag in range(max(8, int(pitch*.85)), int(pitch*1.22)+1):
                if len(trace)-lag < lag or np.std(trace[:-lag]) < 1 or np.std(trace[lag:]) < 1:
                    continue
                correlation = float(np.corrcoef(trace[:-lag], trace[lag:])[0, 1])
                matches.append((correlation, lag))
            if not matches:
                continue
            correlation, black_pitch = max(matches)
            count_float = (triple[0]-offset)/black_pitch - .5
            black_count = round(count_float)
            if correlation < .45 or not 2 <= black_count <= 12 or abs(count_float-black_count) > .38:
                continue
            first = triple[0] - black_count*black_pitch
            if first - black_pitch*.45 < 0:
                continue
            centres = [first+i*black_pitch for i in range(black_count)] + list(triple)
            # Require the complete last window to be within the search band.
            if centres[-1]+pitch*.45 >= w:
                continue
            source = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32)
            inverse = cv2.getPerspectiveTransform(source, p)
            polygons = []
            for index, centre in enumerate(centres):
                step = black_pitch if index < black_count else pitch
                rect = np.array([[centre-step*.46, h*.05], [centre+step*.46, h*.05],
                                 [centre+step*.46, h*.96], [centre-step*.46, h*.96]], np.float32)
                polygon = cv2.perspectiveTransform(rect[None], inverse)[0]
                if (polygon[:,0].min()<0 or polygon[:,1].min()<0 or
                    polygon[:,0].max()>=image.shape[1] or polygon[:,1].max()>=image.shape[0]):
                    break
                polygons.append(polygon)
            if len(polygons) != len(centres):
                continue
            strength = float(min(profile[x] for x in triple))
            if geometry_only and (black_count != 5 or correlation < .60 or strength < 8.0):
                continue
            score = correlation * strength * height
            (geometry_only_proposals if geometry_only else proposals).append((score, polygons))
    if proposals:
        return max(proposals, key=lambda x:x[0])[1]
    if geometry_only_proposals:
        return max(geometry_only_proposals, key=lambda x:x[0])[1]
    return _detect_achromatic_cells(image, items)


def crop_cell(image, polygon):
    p = np.asarray(polygon, np.float32)
    width = max(8, round(np.linalg.norm(p[1] - p[0])))
    height = max(8, round(np.linalg.norm(p[3] - p[0])))
    transform = cv2.getPerspectiveTransform(p, np.array([[0, 0], [width-1, 0],
                                                       [width-1, height-1], [0, height-1]], np.float32))
    return cv2.warpPerspective(image, transform, (width, height), borderValue=(255,255,255))


def empty_reading(status="not_found", message=None):
    return {"value": None, "status": status, "decimal_places": 3,
            "digits_count": None, "cells": [], "error": message}


def recognize_reading(image, recognizer, items):
    polygons = detect_cells(image, items)
    if not polygons:
        return empty_reading()
    cells = []
    for polygon in polygons:
        crop = crop_cell(image, polygon)
        h, w = crop.shape[:2]
        inputs = []
        for trim in (.10, .16):
            inner = crop[int(h*.08):int(h*.92), int(w*trim):int(w*(1-trim))]
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # Use colour separation only when red ink is actually present. Fully
            # black counters keep the ordinary Otsu mask for every position.
            if len(cells) >= len(polygons)-3:
                blue, green, red = cv2.split(inner.astype(float))
                chroma = (red-green)/(red+green+1)
                median_chroma = float(np.median(chroma))
                if float(np.percentile(chroma, 95)) - median_chroma > .025:
                    mask = ((chroma > median_chroma+.025)*255).astype(np.uint8)
            count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
            choices = []
            for label in range(1, count):
                x,y,cw,ch,area = stats[label]
                cx,cy = centres[label]
                if (.20*inner.shape[1] < cx < .80*inner.shape[1] and
                    ch > inner.shape[0]*.35 and cw < inner.shape[1]*.98 and
                    area < mask.size*.60):
                    choices.append((area, label))
            if choices:
                _, label = max(choices)
                mask = np.where(labels==label, 0, 255).astype(np.uint8)
            else:
                mask = 255-mask
            resized = cv2.resize(mask, (max(24, round(mask.shape[1]*96/mask.shape[0])), 96))
            resized = cv2.copyMakeBorder(resized, 12, 12, 20, 20, cv2.BORDER_CONSTANT, value=255)
            inputs.append(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))
        for trim in (.10, .16):
            original = crop[int(h*.08):int(h*.92), int(w*trim):int(w*(1-trim))]
            original = cv2.resize(original, (max(24, round(original.shape[1]*96/original.shape[0])), 96))
            inputs.append(cv2.copyMakeBorder(original, 8, 8, 16, 16, cv2.BORDER_CONSTANT, value=(255,255,255)))
        results = recognizer.predict(inputs)
        observations = [(str(r["rec_text"]), float(r["rec_score"])) for r in results]
        accepted = [t for t, confidence in observations if len(t) == 1 and t in "0123456789" and confidence >= .85]
        digit = accepted[0] if len(accepted) >= 2 and len(set(accepted)) == 1 else "X"
        if digit == "X" and len(cells) < len(polygons)-3:
            strong = {t for t,c in observations if len(t)==1 and t in "0123456789" and c>=.90}
            if len(strong)==1:
                proposed = next(iter(strong))
                support = sum(t==proposed and c>=.65 for t,c in observations)
                conflicts = any(t!=proposed and len(t)==1 and t in "0123456789" and c>=.85 for t,c in observations)
                if support>=2 and not conflicts:
                    digit = proposed
        cells.append({"digit": digit, "confidence": min(c for t, c in observations if t == digit and c >= .85) if digit != "X" else None,
                      "polygon": np.asarray(polygon).astype(float).tolist(),
                      "ocr_candidates": [{"text": t, "confidence": c} for t,c in observations]})
    digits = "".join(cell["digit"] for cell in cells)
    return {"value": digits[:-3] + "." + digits[-3:],
            "status": "partial" if "X" in digits else "recognized",
            "decimal_places": 3, "digits_count": len(cells), "cells": cells, "error": None}
