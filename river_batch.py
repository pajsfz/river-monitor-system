from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Callable

import pandas as pd


def detect_image_batch(
    images: list[tuple[str, bytes]],
    detect_one: Callable[[bytes, str, Path, float], dict],
    model_path: Path,
    confidence: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], list[str]]:
    results = []
    errors = []
    for index, (filename, data) in enumerate(images, start=1):
        try:
            results.append(detect_one(data, filename, model_path, confidence))
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        if progress_callback:
            progress_callback(index, len(images))
    return results, errors


def batch_results_zip(results: list[dict]) -> bytes:
    output = io.BytesIO()
    rows = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, result in enumerate(results, start=1):
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(result["source_name"]).stem).strip("_") or "image"
            archive.writestr(f"annotated/{index:03d}_{stem}.png", result["annotated_image"])
            rows.extend({"文件名": result["source_name"], **item} for item in result["detections"])
        archive.writestr("detections.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"))
    return output.getvalue()
