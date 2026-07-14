import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app_utils.camera import CameraDevice, choose_auto_camera_setup
from app_utils.fusion import FusionEngine
from app_utils.session import DetectionRecorder, EventClipRecorder, SettingsStore
from app_utils.preprocess import preprocess_ir_for_model
from gui.app import MainApp


class RuntimeFeatureTests(unittest.TestCase):
    def test_settings_and_detection_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SettingsStore(root / "settings.json")
            settings = store.load()
            settings["ir_shift_x"] = 4
            store.save(settings)
            self.assertEqual(store.load()["ir_shift_x"], 4)

            recorder = DetectionRecorder()
            recorder.append_frame(
                source="test.mp4",
                frame=1,
                detections=[{"box": (1, 2, 30, 40), "confidence": 0.8, "track_id": 7}],
                inside_ids={7},
                current_count=1,
                unique_count=1,
                device="cpu",
                fps=10.0,
            )
            csv_path = recorder.export(root / "records.csv")
            json_path = recorder.export(root / "records.json")
            self.assertTrue(csv_path.exists())
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))[0]["track_id"], 7)

    def test_invalid_settings_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "imgsz": "invalid",
                        "conf": 8,
                        "iou": float("nan"),
                        "tracking_enabled": "false",
                        "performance_mode": "unknown",
                        "alert_roi": [0.9, 0.8, 0.1, 0.2],
                    }
                ),
                encoding="utf-8",
            )
            settings = SettingsStore(path).load()
            self.assertEqual(settings["imgsz"], 384)
            self.assertEqual(settings["conf"], 1.0)
            self.assertEqual(settings["iou"], 0.45)
            self.assertFalse(settings["tracking_enabled"])
            self.assertEqual(settings["performance_mode"], "balanced")
            self.assertEqual(settings["alert_roi"], [0.1, 0.2, 0.9, 0.8])

    def test_ir_shift(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[10, 10] = 255
        shifted = FusionEngine.shift_ir(image, 3, -2)
        y, x = np.unravel_index(np.argmax(shifted[:, :, 0]), shifted[:, :, 0].shape)
        self.assertEqual((x, y), (13, 8))
        self.assertEqual(int(shifted[10, 0, 0]), 0)

    def test_uint16_infrared_preprocessing_and_preview(self):
        ir = np.linspace(1000, 5000, 24 * 32, dtype=np.uint16).reshape(24, 32)
        processed = preprocess_ir_for_model(ir, realtime=True)
        self.assertEqual(processed.dtype, np.uint8)
        self.assertEqual(processed.shape, (24, 32, 3))
        preview = FusionEngine(use_clahe=False).for_display_fast(
            np.zeros((24, 32, 3), dtype=np.uint8),
            ir,
        )
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape, (24, 32, 3))

    def test_event_clip_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            recorder = EventClipRecorder(
                output,
                pre_seconds=0.1,
                post_seconds=1.0,
                cooldown_seconds=1.0,
            )
            frame = np.zeros((96, 128, 3), dtype=np.uint8)
            for index in range(5):
                recorder.add_frame(frame, fps=10.0, trigger=index == 1, metadata={"track_id": 1})
            recorder.close()
            self.assertTrue(list(output.glob("*.mp4")))
            self.assertTrue(list(output.glob("*.jpg")))
            metadata = list(output.glob("*.json"))
            self.assertTrue(metadata)
            capture = cv2.VideoCapture(str(list(output.glob("*.mp4"))[0]))
            try:
                self.assertGreater(capture.get(cv2.CAP_PROP_FRAME_COUNT), 0)
            finally:
                capture.release()

    def test_event_prebuffer_uses_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0.0]
            recorder = EventClipRecorder(
                directory,
                pre_seconds=1.5,
                post_seconds=1.0,
                cooldown_seconds=1.0,
                clock=lambda: now[0],
            )
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            for timestamp in (0.0, 1.0, 2.0, 3.0):
                now[0] = timestamp
                recorder.add_frame(frame, fps=1.0, trigger=timestamp == 3.0)
            recorder.close()
            metadata = json.loads(next(Path(directory).glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_count"], 2)
            self.assertAlmostEqual(metadata["fps"], 1.0)

    def test_anonymous_alert_entries_and_export_flag(self):
        app = MainApp.__new__(MainApp)
        app.video_alert_enabled = True
        app.video_tracking_enabled = False
        app.alert_roi = (0.0, 0.0, 1.0, 1.0)
        app._seen_track_ids = set()
        app._inside_track_ids = set()
        app._anonymous_inside_count = 0
        app._anonymous_scene_count = 0
        app._anonymous_seen_count = 0
        app._alert_entry_count = 0
        app._alert_flash_until = 0.0
        app.count_var = type("Counter", (), {"set": lambda self, value: None})()
        app._post_ui = lambda callback, *args: callback(*args)
        app.log = lambda message: None
        app._play_alarm_sound = lambda: None

        detections = [
            {"box": (1, 1, 10, 20), "confidence": 0.8, "track_id": None},
            {"box": (20, 1, 30, 20), "confidence": 0.7, "track_id": None},
        ]
        inside, triggered, unique = app._evaluate_alert(detections, (40, 40, 3))
        self.assertEqual(inside, set())
        self.assertTrue(triggered)
        self.assertEqual(unique, 2)
        self.assertEqual(app._alert_entry_count, 2)
        self.assertTrue(all(item["inside_alert_roi"] for item in detections))

        app._evaluate_alert(detections[:1], (40, 40, 3))
        _inside, triggered, unique = app._evaluate_alert(detections, (40, 40, 3))
        self.assertTrue(triggered)
        self.assertEqual(unique, 3)
        self.assertEqual(app._alert_entry_count, 3)

    def test_stop_timeout_does_not_release_live_worker_resources(self):
        app = MainApp.__new__(MainApp)
        release_worker = threading.Event()
        worker = threading.Thread(target=release_worker.wait)
        worker.start()
        app.video_worker = worker
        app.video_running = True
        app.video_paused = True
        releases = []
        app._release_video_io = lambda: releases.append(True)
        app.log = lambda message: None
        try:
            self.assertFalse(app.stop_video(wait_timeout=0.01))
            self.assertFalse(releases)
        finally:
            release_worker.set()
            worker.join()
        self.assertTrue(app.stop_video(wait_timeout=0.1))
        self.assertEqual(releases, [True])

    def test_mp4_writer_keeps_frames_and_can_switch_path(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MainApp.__new__(MainApp)
            app.output_writer = None
            app.output_writer_path = None
            app._fps_ema = 10.0
            app.log = lambda message: None
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            paths = [Path(directory) / "first.mp4", Path(directory) / "second.mp4"]
            for path in paths:
                app.output_path = str(path)
                for index in range(3):
                    frame[:] = index * 30
                    app._write_output_frame(frame, source_fps=10.0, is_camera=False)
            app._close_output_writer(clear_request=True)
            for path in paths:
                capture = cv2.VideoCapture(str(path))
                try:
                    self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                finally:
                    capture.release()

    def test_camera_scan_uses_ui_queue(self):
        app = MainApp.__new__(MainApp)
        calls = []
        app._post_ui = lambda callback, *args: calls.append((callback.__name__, args))
        with patch("gui.app.discover_cameras", return_value=[]):
            app._scan_cameras(12)
        self.assertEqual(calls, [("_start_auto_camera", (12, []))])

    def test_camera_selection_prefers_external_rgb_and_ir_pair(self):
        devices = [
            CameraDevice(0, "Integrated Camera", cv2.CAP_DSHOW),
            CameraDevice(1, "USB RGB Camera", cv2.CAP_DSHOW),
            CameraDevice(2, "USB IR Camera", cv2.CAP_DSHOW),
        ]
        setup = choose_auto_camera_setup(devices)
        self.assertIsNotNone(setup)
        self.assertEqual(setup.mode, "rgbt")
        self.assertEqual(setup.rgb.index, 1)
        self.assertEqual(setup.ir.index, 2)

    def test_generic_camera_is_not_mislabeled_as_external(self):
        device = CameraDevice(0, "Camera 0", cv2.CAP_DSHOW)
        setup = choose_auto_camera_setup([device])
        self.assertEqual(device.kind, "RGB")
        self.assertIsNotNone(setup)
        self.assertEqual(setup.mode, "rgb")
        self.assertEqual(setup.rgb.index, 0)

    def test_roi_press_ignores_letterbox_margin(self):
        app = MainApp.__new__(MainApp)
        label = object()
        app.lbl_result = SimpleNamespace(inner_label=label)
        app._display_geometry = {label: (640, 512, 320, 256, 20, 10)}
        self.assertIsNone(app._event_to_normalized(SimpleNamespace(x=5, y=100)))
        self.assertEqual(
            app._event_to_normalized(SimpleNamespace(x=5, y=100), clamp=True),
            (0.0, 90 / 256),
        )


if __name__ == "__main__":
    unittest.main()
