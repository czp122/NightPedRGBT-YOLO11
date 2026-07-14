from __future__ import annotations

import csv
import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


DEFAULT_SETTINGS = {
    "dataset_path": "",
    "imgsz": 384,
    "conf": 0.25,
    "iou": 0.45,
    "performance_mode": "balanced",
    "tracking_enabled": True,
    "alert_enabled": True,
    "ir_frame_offset": 0,
    "ir_shift_x": 0,
    "ir_shift_y": 0,
    "pre_event_seconds": 3.0,
    "post_event_seconds": 5.0,
    "alarm_cooldown_seconds": 5.0,
    "alert_roi": None,
}


def _bounded_number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(maximum, max(minimum, number))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def normalize_settings(values: dict[str, Any] | None) -> dict[str, Any]:
    values = values if isinstance(values, dict) else {}
    settings = dict(DEFAULT_SETTINGS)
    settings["dataset_path"] = str(values.get("dataset_path", ""))
    settings["imgsz"] = int(_bounded_number(values.get("imgsz"), 384, 32, 2048))
    settings["conf"] = _bounded_number(values.get("conf"), 0.25, 0.01, 1.0)
    settings["iou"] = _bounded_number(values.get("iou"), 0.45, 0.01, 1.0)
    performance = str(values.get("performance_mode", "balanced")).lower()
    settings["performance_mode"] = performance if performance in {"quality", "balanced", "smooth"} else "balanced"
    settings["tracking_enabled"] = _as_bool(values.get("tracking_enabled"), True)
    settings["alert_enabled"] = _as_bool(values.get("alert_enabled"), True)
    settings["ir_frame_offset"] = int(_bounded_number(values.get("ir_frame_offset"), 0, -60, 60))
    settings["ir_shift_x"] = int(_bounded_number(values.get("ir_shift_x"), 0, -100, 100))
    settings["ir_shift_y"] = int(_bounded_number(values.get("ir_shift_y"), 0, -100, 100))
    settings["pre_event_seconds"] = _bounded_number(values.get("pre_event_seconds"), 3.0, 0.0, 10.0)
    settings["post_event_seconds"] = _bounded_number(values.get("post_event_seconds"), 5.0, 1.0, 30.0)
    settings["alarm_cooldown_seconds"] = _bounded_number(
        values.get("alarm_cooldown_seconds"), 5.0, 1.0, 60.0
    )

    roi = values.get("alert_roi")
    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        coordinates = [_bounded_number(value, math.nan, 0.0, 1.0) for value in roi]
        if all(math.isfinite(value) for value in coordinates):
            x1, y1, x2, y2 = coordinates
            settings["alert_roi"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return settings


class SettingsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            saved = {}
        return normalize_settings(saved)

    def save(self, settings: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        normalized = normalize_settings(settings)
        temp.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)


class DetectionRecorder:
    FIELDNAMES = (
        "timestamp",
        "source",
        "frame",
        "track_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "inside_alert_roi",
        "current_count",
        "unique_count",
        "device",
        "fps",
    )

    def __init__(self, max_rows: int = 100_000):
        self.max_rows = max(1, int(max_rows))
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.max_rows)
        self._lock = threading.Lock()

    def append_frame(
        self,
        *,
        source: str,
        frame: int | str,
        detections: list[dict[str, Any]],
        inside_ids: set[int],
        current_count: int,
        unique_count: int,
        device: str,
        fps: float,
    ) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            track_id = detection.get("track_id")
            rows.append(
                {
                    "timestamp": timestamp,
                    "source": source,
                    "frame": frame,
                    "track_id": "" if track_id is None else int(track_id),
                    "confidence": round(float(detection["confidence"]), 4),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "inside_alert_roi": bool(
                        detection.get("inside_alert_roi", False)
                        or (track_id is not None and int(track_id) in inside_ids)
                    ),
                    "current_count": int(current_count),
                    "unique_count": int(unique_count),
                    "device": device,
                    "fps": round(float(fps), 2),
                }
            )
        if rows:
            with self._lock:
                self._rows.extend(rows)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def export(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self._rows)
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            with output.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
        return output


class EventClipRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        pre_seconds: float = 3.0,
        post_seconds: float = 5.0,
        cooldown_seconds: float = 5.0,
        logger: Callable[[str], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_seconds = min(10.0, max(0.0, float(pre_seconds)))
        self.post_seconds = min(30.0, max(1.0, float(post_seconds)))
        self.cooldown_seconds = min(60.0, max(1.0, float(cooldown_seconds)))
        self.logger = logger
        self._clock = clock or time.monotonic
        self._prebuffer: deque[tuple[float, bytes]] = deque()
        self._active: dict[str, Any] | None = None
        self._cooldown_until = 0.0
        self._workers: list[threading.Thread] = []
        self._lock = threading.RLock()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active is not None

    def configure(self, pre_seconds: float, post_seconds: float, cooldown_seconds: float) -> None:
        with self._lock:
            self.pre_seconds = min(10.0, max(0.0, float(pre_seconds)))
            self.post_seconds = min(30.0, max(1.0, float(post_seconds)))
            self.cooldown_seconds = min(60.0, max(1.0, float(cooldown_seconds)))
            self._prune_prebuffer(self._clock())

    def add_frame(
        self,
        frame,
        *,
        fps: float,
        trigger: bool,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return False

        packet = encoded.tobytes()
        now = self._clock()
        with self._lock:
            self._prebuffer.append((now, packet))
            self._prune_prebuffer(now)
            started = False

            if trigger and self._active is None and now >= self._cooldown_until:
                self._active = {
                    "frames": list(self._prebuffer),
                    "snapshot": packet,
                    "initial_fps": max(1.0, min(30.0, float(fps) or 10.0)),
                    "deadline": now + self.post_seconds,
                    "metadata": dict(metadata or {}),
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                }
                self._cooldown_until = now + max(self.cooldown_seconds, self.post_seconds)
                started = True
            elif self._active is not None:
                self._active["frames"].append((now, packet))

            if self._active is not None and now >= self._active["deadline"]:
                self._finish_active_locked()
            return started

    def _prune_prebuffer(self, now: float) -> None:
        cutoff = now - self.pre_seconds
        while self._prebuffer and self._prebuffer[0][0] < cutoff:
            self._prebuffer.popleft()

    def flush(self) -> None:
        with self._lock:
            if self._active is not None:
                self._finish_active_locked()

    def reset(self) -> None:
        self.flush()
        with self._lock:
            self._prebuffer.clear()
            self._cooldown_until = 0.0

    def close(self, timeout: float | None = None) -> None:
        self.flush()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                workers = [worker for worker in self._workers if worker.is_alive()]
            if not workers:
                return
            for worker in workers:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return
                worker.join(timeout=remaining)

    def _finish_active(self) -> None:
        with self._lock:
            self._finish_active_locked()

    def _finish_active_locked(self) -> None:
        event, self._active = self._active, None
        if not event or not event["frames"]:
            return
        timestamps = [timestamp for timestamp, _packet in event["frames"]]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        event["fps"] = (
            min(30.0, max(1.0, (len(timestamps) - 1) / duration))
            if duration > 0
            else event["initial_fps"]
        )
        worker = threading.Thread(target=self._write_event_safe, args=(event,), daemon=False)
        self._workers = [item for item in self._workers if item.is_alive()]
        self._workers.append(worker)
        worker.start()

    def _write_event_safe(self, event: dict[str, Any]) -> None:
        try:
            self._write_event(event)
        except Exception as error:
            if self.logger:
                self.logger(f"[ALERT] 事件录像保存失败: {error}")
        finally:
            current = threading.current_thread()
            with self._lock:
                self._workers = [worker for worker in self._workers if worker is not current]

    def _write_event(self, event: dict[str, Any]) -> None:
        stem = f"alert_{event['timestamp']}"
        video_path = self.output_dir / f"{stem}.mp4"
        image_path = self.output_dir / f"{stem}.jpg"
        json_path = self.output_dir / f"{stem}.json"

        first_packet = event["frames"][0][1]
        first = cv2.imdecode(np.frombuffer(first_packet, dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            event["fps"],
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建事件视频: {video_path}")
        try:
            for _timestamp, packet in event["frames"]:
                frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)
        finally:
            writer.release()

        snapshot = cv2.imdecode(np.frombuffer(event["snapshot"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if snapshot is None or not cv2.imwrite(str(image_path), snapshot):
            raise RuntimeError(f"无法保存事件截图: {image_path}")

        metadata = dict(event["metadata"])
        metadata.update(
            {
                "video": str(video_path),
                "snapshot": str(image_path),
                "frame_count": len(event["frames"]),
                "fps": event["fps"],
            }
        )
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.logger:
            self.logger(f"[ALERT] 事件录像已保存: {video_path}")
