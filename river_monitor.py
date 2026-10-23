from __future__ import annotations

import subprocess
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class ModelRuntime:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray, confidence: float) -> tuple[np.ndarray, list[dict]]:
        with self.lock:
            result = self.model.predict(frame_bgr, conf=confidence, verbose=False)[0]
            annotated = result.plot()
            detections = []
            if result.boxes is not None:
                boxes = result.boxes
                coords = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                names = result.names
                for box, score, class_id in zip(coords, confidences, class_ids):
                    class_id = int(class_id)
                    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    x1, y1, x2, y2 = (int(round(value)) for value in box)
                    detections.append(
                        {
                            "类别": str(label),
                            "置信度": float(score),
                            "左上角X": x1,
                            "左上角Y": y1,
                            "右下角X": x2,
                            "右下角Y": y2,
                        }
                    )
        return annotated, detections


def summarize(detections: list[dict]) -> dict:
    counts = Counter(row["类别"] for row in detections)
    mean_confidence = (
        sum(row["置信度"] for row in detections) / len(detections) if detections else None
    )
    return {
        "目标框总数": len(detections),
        "类别数": len(counts),
        "平均置信度": mean_confidence,
        "分类统计": dict(counts),
    }


def process_video(
    input_path: Path,
    output_path: Path,
    runtime: ModelRuntime,
    confidence: float,
    ffmpeg_path: str,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[list[dict], int, float]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("无法打开视频。请尝试 MP4、MOV 或 AVI 格式的视频文件。")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("视频尺寸无效或编码格式不受支持。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_width = width + width % 2
    output_height = height + height % 2
    all_detections: list[dict] = []
    frame_index = 0
    temp_avi_path: Path | None = None
    writer: cv2.VideoWriter | None = None
    processing_complete = False

    try:
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False, dir=output_path.parent) as temp_file:
            temp_avi_path = Path(temp_file.name)

        writer = cv2.VideoWriter(
            str(temp_avi_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
        )
        if not writer.isOpened():
            raise RuntimeError("无法创建标注视频文件，请检查 runs/processed 目录的写入权限。")

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, frame_detections = runtime.detect(frame, confidence)
            if output_width != width or output_height != height:
                annotated = cv2.copyMakeBorder(
                    annotated, 0, output_height - height, 0, output_width - width, cv2.BORDER_REPLICATE
                )
            writer.write(annotated)
            seconds = frame_index / fps
            for detection in frame_detections:
                all_detections.append(
                    {"帧序号": frame_index + 1, "时间(秒)": round(seconds, 3), **detection}
                )
            frame_index += 1
            if progress_callback and (frame_index % 5 == 0 or frame_index == reported_frames):
                progress_callback(frame_index, reported_frames, len(all_detections))

        if progress_callback:
            progress_callback(frame_index, reported_frames or frame_index, len(all_detections))
        if frame_index == 0:
            raise ValueError("视频中没有读取到有效画面。")
        processing_complete = True
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not processing_complete and temp_avi_path:
            temp_avi_path.unlink(missing_ok=True)

    try:
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temp_avi_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
            detail = completed.stderr.strip()[-1200:]
            raise RuntimeError(f"标注视频转码失败。{detail}")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_avi_path and temp_avi_path.exists():
            temp_avi_path.unlink(missing_ok=True)

    return all_detections, frame_index, fps
