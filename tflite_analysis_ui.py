#!/usr/bin/env python3
"""
tflite_analysis_ui.py

Segmentation TFLite inference review UI with optimised FPS.

Key optimisations vs a naïve approach:
  1. Class filter applied before any mask work  → process ~40 of 230 classes.
  2. Bounding boxes computed in model output space then mapped back to
     original frame coords via PreprocessMeta  → no full-res mask resize for
     bboxes.
  3. Full-res mask restoration only when overlay_masks is enabled.
  4. Grad-CAM computed strictly on demand (button) and cached per
     (video_path, frame_index, class_id).
  5. Background worker thread + queue.Queue(maxsize=2) decouples inference
     from Tkinter display; UI thread only shows the most-recent rendered frame.
  6. cv2.resize(INTER_AREA) instead of PIL LANCZOS for display downscaling.
"""

from __future__ import annotations

import os
import time
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as _tflite_rt

    _Interpreter = _tflite_rt.Interpreter
except ImportError:
    try:
        import tensorflow as _tf  # type: ignore

        _Interpreter = _tf.lite.Interpreter
    except ImportError:
        _Interpreter = None  # type: ignore

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PreprocessMeta:
    """Records how a frame was preprocessed before model inference.

    Provides helpers to map model-output coordinates back to original frame
    coordinates without re-running the full preprocessing pipeline.
    """

    orig_h: int
    orig_w: int
    input_h: int    # model input canvas height
    input_w: int    # model input canvas width
    scale: float    # uniform resize scale applied to the original frame
    pad_top: int    # letterbox top padding (pixels, in input-canvas space)
    pad_left: int   # letterbox left padding (pixels, in input-canvas space)
    resized_h: int  # orig_h * scale (before padding)
    resized_w: int  # orig_w * scale (before padding)

    def output_bbox_to_original(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        out_h: int,
        out_w: int,
    ) -> Tuple[int, int, int, int]:
        """Map a bounding box from model output space → original frame coords.

        Steps:
          output (out_h × out_w)
            → input/canvas (input_h × input_w)  [scale by input/out ratio]
            → remove letterbox padding
            → undo resize scale
            → clip to original image bounds
        """
        sx = self.input_w / out_w
        sy = self.input_h / out_h

        xi1, yi1 = x1 * sx - self.pad_left, y1 * sy - self.pad_top
        xi2, yi2 = x2 * sx - self.pad_left, y2 * sy - self.pad_top

        # x1/y1 are left/top (inclusive) edges; clip to [0, orig_dim-1].
        # x2/y2 are right/bottom (exclusive) edges; clip to [1, orig_dim].
        ox1 = max(0, min(self.orig_w - 1, int(xi1 / self.scale)))
        oy1 = max(0, min(self.orig_h - 1, int(yi1 / self.scale)))
        ox2 = max(ox1 + 1, min(self.orig_w, int(xi2 / self.scale)))
        oy2 = max(oy1 + 1, min(self.orig_h, int(yi2 / self.scale)))
        return ox1, oy1, ox2, oy2

    def output_mask_to_original(self, mask_small: np.ndarray) -> np.ndarray:
        """Restore a binary mask from output resolution → original frame resolution.

        Args:
            mask_small: bool/uint8 array of shape (out_h, out_w).
        Returns:
            bool array of shape (orig_h, orig_w).
        """
        # Upsample to input/canvas size
        canvas = cv2.resize(
            mask_small.astype(np.uint8),
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_NEAREST,
        )
        # Crop out letterbox padding to get the valid resized region
        valid = canvas[
            self.pad_top : self.pad_top + self.resized_h,
            self.pad_left : self.pad_left + self.resized_w,
        ]
        # Resize to original dimensions
        full = cv2.resize(
            valid,
            (self.orig_w, self.orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return full.astype(bool)


@dataclass
class Detection:
    """A single detected segmentation region.

    mask_small  – binary mask at model output resolution; always available.
    mask_full   – binary mask at original frame resolution; computed lazily
                  only when mask overlay is requested.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) in original frame
    mask_small: np.ndarray              # shape (out_h, out_w), dtype bool
    mask_full: Optional[np.ndarray] = field(default=None)

    def ensure_full_mask(self, meta: PreprocessMeta) -> np.ndarray:
        """Lazily compute and cache the full-resolution mask."""
        if self.mask_full is None:
            self.mask_full = meta.output_mask_to_original(self.mask_small)
        return self.mask_full


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def letterbox_frame(
    frame: np.ndarray,
    input_h: int,
    input_w: int,
) -> Tuple[np.ndarray, PreprocessMeta]:
    """Letterbox a BGR frame to (input_h, input_w) keeping aspect ratio.

    Returns the padded canvas (uint8 BGR) and a PreprocessMeta object that
    records all inverse-transform parameters.
    """
    orig_h, orig_w = frame.shape[:2]
    scale = min(input_h / orig_h, input_w / orig_w)
    resized_h = round(orig_h * scale)
    resized_w = round(orig_w * scale)

    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (input_h - resized_h) // 2
    pad_left = (input_w - resized_w) // 2
    pad_bottom = input_h - resized_h - pad_top
    pad_right = input_w - resized_w - pad_left

    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    meta = PreprocessMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        input_h=input_h,
        input_w=input_w,
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
        resized_h=resized_h,
        resized_w=resized_w,
    )
    return canvas, meta


# ─────────────────────────────────────────────────────────────────────────────
# Model runner
# ─────────────────────────────────────────────────────────────────────────────


class ModelRunner:
    """Thin wrapper around a TFLite segmentation model.

    Supports output layouts:
      • (1, out_h, out_w, num_classes)  – channel-last  (default)
      • (1, num_classes, out_h, out_w)  – channel-first (transposed automatically)
    """

    def __init__(self, model_path: str, num_threads: int = 4) -> None:
        if _Interpreter is None:
            raise RuntimeError(
                "Neither tflite_runtime nor tensorflow is installed.\n"
                "Install with: pip install tflite-runtime"
            )
        self.interpreter = _Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()

        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        self.input_index: int = inp["index"]
        self.output_index: int = out["index"]
        self.input_dtype = inp["dtype"]

        _, h, w, _ = inp["shape"]
        self.input_h: int = h
        self.input_w: int = w
        self.num_classes: Optional[int] = None

    def infer(
        self, bgr_frame: np.ndarray
    ) -> Tuple[np.ndarray, PreprocessMeta]:
        """Preprocess *bgr_frame*, run TFLite inference, return logits + meta.

        Returns:
            logits: float32 ndarray of shape (out_h, out_w, num_classes).
            meta:   PreprocessMeta for inverse coordinate mapping.
        """
        canvas, meta = letterbox_frame(bgr_frame, self.input_h, self.input_w)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)  # (1, H, W, 3)

        if self.input_dtype == np.uint8:
            inp = (inp * 255).clip(0, 255).astype(np.uint8)

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()

        out = self.interpreter.get_tensor(self.output_index)[0]  # drop batch

        # Normalise to channel-last (H, W, C)
        if out.ndim == 3:
            if out.shape[0] < out.shape[1] and out.shape[0] < out.shape[2]:
                out = np.transpose(out, (1, 2, 0))

        if self.num_classes is None:
            self.num_classes = out.shape[-1]

        return out, meta


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────


def _make_palette(n: int) -> np.ndarray:
    """Generate *n* visually distinct BGR colours using golden-ratio hue spacing."""
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        h6 = hue * 6.0
        seg = int(h6)
        f = h6 - seg
        v, s = 0.9, 0.8
        p = int(v * (1 - s) * 255)
        q = int(v * (1 - s * f) * 255)
        t = int(v * (1 - s * (1 - f)) * 255)
        vv = int(v * 255)
        rgb_options = [(vv, t, p), (q, vv, p), (p, vv, t), (p, q, vv), (t, p, vv), (vv, p, q)]
        r, g, b = rgb_options[seg % 6]
        colors.append((b, g, r))  # BGR
    return np.array(colors, dtype=np.uint8)


PALETTE = _make_palette(256)


# ─────────────────────────────────────────────────────────────────────────────
# Detection building
# ─────────────────────────────────────────────────────────────────────────────


def build_detections(
    logits: np.ndarray,
    meta: PreprocessMeta,
    class_names: List[str],
    active_class_ids: List[int],
    conf_threshold: float = 0.5,
    overlay_masks: bool = False,
) -> List[Detection]:
    """Build detections from model logits.

    Performance contract:
      • Only iterates over *active_class_ids* (the filtered ~40 classes), not
        all ~230 classes.
      • Bounding boxes are derived from the small output-space mask and then
        mapped back via *meta*  — no full-res resize is performed unless
        *overlay_masks* is True.
      • Full-resolution masks are only computed when *overlay_masks=True*.

    Args:
        logits:           float32 (out_h, out_w, num_classes).
        meta:             PreprocessMeta from letterbox_frame.
        class_names:      Name list indexed by class_id.
        active_class_ids: Only these class IDs are processed.
        conf_threshold:   Minimum per-pixel sigmoid confidence to count as
                          a positive detection.
        overlay_masks:    If True, also compute full-resolution masks.

    Returns:
        List of Detection objects.
    """
    out_h, out_w, num_cls = logits.shape
    detections: List[Detection] = []

    for cid in active_class_ids:
        if cid >= num_cls:
            continue

        class_logits = logits[:, :, cid]  # (out_h, out_w)

        # Sigmoid confidence map
        conf_map: np.ndarray = 1.0 / (1.0 + np.exp(-class_logits))
        conf = float(conf_map.max())
        if conf < conf_threshold:
            continue

        # Binary mask in model output space (cheap – no full resize)
        mask_small = conf_map >= conf_threshold  # bool (out_h, out_w)
        if not mask_small.any():
            continue

        # Bounding box in output space → mapped to original frame coords.
        # x2/y2 are the exclusive right/bottom edges (pixel_index + 1) so
        # that the full pixel width is included in the mapped bbox.
        ys, xs = np.where(mask_small)
        bbox = meta.output_bbox_to_original(
            float(xs.min()), float(ys.min()),
            float(xs.max()) + 1.0, float(ys.max()) + 1.0,
            out_h, out_w,
        )

        # Full-res mask only when the overlay is enabled
        mask_full: Optional[np.ndarray] = None
        if overlay_masks:
            mask_full = meta.output_mask_to_original(mask_small)

        cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        detections.append(
            Detection(
                class_id=cid,
                class_name=cname,
                confidence=conf,
                bbox=bbox,
                mask_small=mask_small,
                mask_full=mask_full,
            )
        )

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    overlay_masks: bool = False,
    label_mode: str = "name",   # "name" | "id" | "conf" | "none"
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """Draw bounding boxes (and optional mask overlays) onto a copy of *frame*.

    Uses cv2.addWeighted for blending (vectorised C++) instead of per-pixel
    Python loops.
    """
    canvas = frame.copy()

    for det in detections:
        color = tuple(int(c) for c in PALETTE[det.class_id % len(PALETTE)])
        x1, y1, x2, y2 = det.bbox

        # Mask overlay (only when full mask is available)
        if overlay_masks and det.mask_full is not None and det.mask_full.any():
            color_layer = np.zeros_like(canvas)
            color_layer[det.mask_full] = color
            cv2.addWeighted(canvas, 1.0 - mask_alpha, color_layer, mask_alpha, 0, canvas)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label
        if label_mode != "none":
            if label_mode == "id":
                label = str(det.class_id)
            elif label_mode == "conf":
                label = f"{det.class_name} {det.confidence:.2f}"
            else:
                label = det.class_name

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx = max(x1, 0)
            ly = max(y1 - 5, th + 2)
            cv2.rectangle(canvas, (lx, ly - th - 2), (lx + tw, ly + 2), color, -1)
            cv2.putText(
                canvas, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM – on-demand, cached
# ─────────────────────────────────────────────────────────────────────────────


class GradCAMCache:
    """Approximate class activation maps for TFLite segmentation models.

    TFLite does not expose gradient computation, so we approximate CAM by
    using the sigmoid of the raw class logit channel as a per-pixel saliency
    map and rendering it as a colour heatmap.

    Results are cached per (video_path, frame_index, class_id) to avoid
    recomputation on repeated requests (e.g. re-renders while paused).
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def get(
        self,
        video_path: str,
        frame_index: int,
        class_id: int,
        logits: np.ndarray,
        meta: PreprocessMeta,
    ) -> np.ndarray:
        """Return a BGR heatmap at original frame resolution (cached)."""
        key = (video_path, frame_index, class_id)
        if key not in self._cache:
            self._cache[key] = self._compute(logits, class_id, meta)
        return self._cache[key]

    def _compute(
        self,
        logits: np.ndarray,
        class_id: int,
        meta: PreprocessMeta,
    ) -> np.ndarray:
        if class_id >= logits.shape[2]:
            return np.zeros((meta.orig_h, meta.orig_w, 3), dtype=np.uint8)

        conf_map = 1.0 / (1.0 + np.exp(-logits[:, :, class_id]))

        cam_min = float(conf_map.min())
        cam_range = float(conf_map.max()) - cam_min
        if cam_range > 0:
            cam_norm = ((conf_map - cam_min) / cam_range * 255).astype(np.uint8)
        else:
            cam_norm = np.zeros(conf_map.shape, dtype=np.uint8)

        # Upsample to input canvas, then crop letterbox padding, then to original
        canvas_up = cv2.resize(
            cam_norm, (meta.input_w, meta.input_h), interpolation=cv2.INTER_LINEAR
        )
        valid = canvas_up[
            meta.pad_top : meta.pad_top + meta.resized_h,
            meta.pad_left : meta.pad_left + meta.resized_w,
        ]
        original_cam = cv2.resize(
            valid, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR
        )
        return cv2.applyColorMap(original_cam, cv2.COLORMAP_JET)

    def clear(self) -> None:
        self._cache.clear()


GRADCAM_CACHE = GradCAMCache()

# Sentinel used in _poll_queue to distinguish "queue was empty" from the
# None EOF sentinel that the worker pushes when a video ends.
_QUEUE_EMPTY = object()


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameResult:
    """Everything the UI thread needs to display one frame."""

    rendered_bgr: np.ndarray       # BGR frame with boxes (+ masks if enabled)
    detections: List[Detection]
    frame_index: int
    logits: Optional[np.ndarray]   # kept for on-demand Grad-CAM
    meta: Optional[PreprocessMeta]
    elapsed_ms: float              # inference + postprocess time


class PlaybackWorker(threading.Thread):
    """Background thread: read frames → infer → draw → push to queue.

    The Tkinter main thread only reads the most-recent FrameResult; stale
    frames in the queue are dropped so the UI never lags behind inference.

    Thread-safe commands are sent via :meth:`send_command`.
    """

    def __init__(
        self,
        model_runner: Optional[ModelRunner],
        result_queue: "queue.Queue[Optional[FrameResult]]",
    ) -> None:
        super().__init__(daemon=True)
        self.model_runner = model_runner
        self.result_queue = result_queue

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()   # start in playing state
        self._cmd_queue: "queue.Queue[Tuple]" = queue.Queue()
        self._step_one = False    # process exactly one frame then auto-pause

        # Playback state (mutated only from the worker thread via commands)
        self.video_path: Optional[str] = None
        self.class_names: List[str] = []
        self.active_class_ids: List[int] = []
        self.conf_threshold: float = 0.5
        self.overlay_masks: bool = False
        self.label_mode: str = "name"
        self.frame_index: int = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._seek_to: Optional[int] = None

    # ── Public thread-safe API ────────────────────────────────────────────

    def send_command(self, cmd: str, **kwargs: object) -> None:
        """Enqueue a command to be processed by the worker thread."""
        self._cmd_queue.put((cmd, kwargs))

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # unblock any wait

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _drain_commands(self) -> None:
        while not self._cmd_queue.empty():
            try:
                cmd, kwargs = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_command(cmd, kwargs)

    def _apply_command(self, cmd: str, kwargs: dict) -> None:
        if cmd == "set_video":
            self._open_video(kwargs["path"])
        elif cmd == "seek":
            self._seek_to = int(kwargs["frame_index"])
        elif cmd == "step_one":
            self._step_one = True
            self._pause_event.set()  # ensure at least one frame runs
        elif cmd == "set_active_classes":
            self.active_class_ids = list(kwargs["class_ids"])
        elif cmd == "set_conf_threshold":
            self.conf_threshold = float(kwargs["value"])
        elif cmd == "set_overlay_masks":
            self.overlay_masks = bool(kwargs["value"])
        elif cmd == "set_label_mode":
            self.label_mode = str(kwargs["value"])
        elif cmd == "set_class_names":
            self.class_names = list(kwargs["names"])

    def _open_video(self, path: str) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        self.video_path = path
        self.frame_index = 0

    def _put_result(self, result: Optional[FrameResult]) -> None:
        """Non-blocking push; drop the oldest entry if the queue is full."""
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._drain_commands()

            if not self._pause_event.is_set():
                self._pause_event.wait(timeout=0.05)
                continue

            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.05)
                continue

            # Seek if requested
            if self._seek_to is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                self.frame_index = self._seek_to
                self._seek_to = None

            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                # End of video
                self._put_result(None)   # EOF sentinel
                self._pause_event.clear()
                continue

            fi = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.frame_index = fi

            logits: Optional[np.ndarray] = None
            meta: Optional[PreprocessMeta] = None
            detections: List[Detection] = []

            if self.model_runner is not None and self.active_class_ids:
                try:
                    logits, meta = self.model_runner.infer(frame)
                    detections = build_detections(
                        logits=logits,
                        meta=meta,
                        class_names=self.class_names,
                        active_class_ids=self.active_class_ids,
                        conf_threshold=self.conf_threshold,
                        overlay_masks=self.overlay_masks,
                    )
                except Exception as exc:
                    print(f"[Worker] Inference error frame {fi}: {exc}")

            rendered = draw_detections(
                frame,
                detections,
                overlay_masks=self.overlay_masks,
                label_mode=self.label_mode,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._put_result(
                FrameResult(
                    rendered_bgr=rendered,
                    detections=detections,
                    frame_index=fi,
                    logits=logits,
                    meta=meta,
                    elapsed_ms=elapsed_ms,
                )
            )

            # Single-step mode: auto-pause after one frame
            if self._step_one:
                self._step_one = False
                self._pause_event.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Display constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DISPLAY_W = 900
_DEFAULT_DISPLAY_H = 675


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────


class VideoReviewApp:
    """Tkinter review UI.

    Rendering is decoupled from inference via a background PlaybackWorker.
    The Tkinter main thread polls the result queue every ~16 ms and only
    renders the most-recent frame (stale frames are discarded).
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Segmentation Review")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Application state
        self.model_runner: Optional[ModelRunner] = None
        self.class_names: List[str] = []
        self.video_paths: List[str] = []
        self.video_idx: int = 0
        self.total_frames: int = 0

        # Current-frame state (UI thread only)
        self._last_result: Optional[FrameResult] = None
        self._current_photo: Optional[ImageTk.PhotoImage] = None
        self._gradcam_overlay: Optional[np.ndarray] = None
        self._show_gradcam: bool = False
        self._playing: bool = False

        # Display size (updated on canvas resize)
        self._display_w: int = _DEFAULT_DISPLAY_W
        self._display_h: int = _DEFAULT_DISPLAY_H

        # FPS tracking (display side)
        self._display_times: List[float] = []
        self._fps_window: int = 20

        # Worker + queue
        self._result_queue: "queue.Queue[Optional[FrameResult]]" = queue.Queue(maxsize=2)
        self._worker: Optional[PlaybackWorker] = None

        self._build_ui()
        self._start_worker()
        self._poll_queue()   # kick off Tkinter polling loop

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Top toolbar ──────────────────────────────────────────────────
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

        ttk.Button(top, text="Load Model", command=self._load_model).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(top, text="Load Videos", command=self._load_videos).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(top, text="Conf:").pack(side=tk.LEFT, padx=(8, 0))
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            top,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            length=100,
            command=lambda _: self._on_settings_change(),
        ).pack(side=tk.LEFT, padx=2)
        self.conf_label = ttk.Label(top, text="0.50")
        self.conf_label.pack(side=tk.LEFT)

        ttk.Label(top, text="Labels:").pack(side=tk.LEFT, padx=(8, 0))
        self.label_mode_var = tk.StringVar(value="name")
        lm_combo = ttk.Combobox(
            top,
            textvariable=self.label_mode_var,
            values=["name", "id", "conf", "none"],
            width=6,
            state="readonly",
        )
        lm_combo.pack(side=tk.LEFT, padx=2)
        lm_combo.bind("<<ComboboxSelected>>", lambda _: self._on_settings_change())

        # ── Main area: canvas + sidebar ───────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            main,
            width=_DEFAULT_DISPLAY_W,
            height=_DEFAULT_DISPLAY_H,
            bg="black",
            cursor="crosshair",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        sidebar = ttk.Frame(main, width=230)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        sidebar.pack_propagate(False)

        # Class filter
        ttk.Label(sidebar, text="Class filter", font=("", 10, "bold")).pack(
            anchor=tk.W
        )
        filter_frame = ttk.Frame(sidebar)
        filter_frame.pack(fill=tk.X, pady=2)

        ttk.Label(filter_frame, text="From:").grid(row=0, column=0, sticky=tk.W)
        self.filter_from_var = tk.IntVar(value=0)
        ttk.Spinbox(
            filter_frame,
            textvariable=self.filter_from_var,
            from_=0,
            to=9999,
            width=6,
            command=self._on_filter_change,
        ).grid(row=0, column=1, padx=2)

        ttk.Label(filter_frame, text="To:").grid(row=1, column=0, sticky=tk.W)
        self.filter_to_var = tk.IntVar(value=229)
        ttk.Spinbox(
            filter_frame,
            textvariable=self.filter_to_var,
            from_=0,
            to=9999,
            width=6,
            command=self._on_filter_change,
        ).grid(row=1, column=1, padx=2)

        # Mask overlay
        self.mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sidebar,
            text="Overlay masks",
            variable=self.mask_var,
            command=self._on_settings_change,
        ).pack(anchor=tk.W, pady=(8, 0))

        # Grad-CAM section
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(sidebar, text="Grad-CAM", font=("", 10, "bold")).pack(anchor=tk.W)
        gc_frame = ttk.Frame(sidebar)
        gc_frame.pack(fill=tk.X)
        ttk.Label(gc_frame, text="Class ID:").grid(row=0, column=0, sticky=tk.W)
        self.gradcam_class_var = tk.IntVar(value=0)
        ttk.Spinbox(
            gc_frame,
            textvariable=self.gradcam_class_var,
            from_=0,
            to=9999,
            width=6,
        ).grid(row=0, column=1, padx=2)
        ttk.Button(
            sidebar,
            text="Compute Grad-CAM (current frame)",
            command=self._compute_gradcam,
        ).pack(fill=tk.X, pady=(4, 2))
        self.show_gradcam_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sidebar,
            text="Show Grad-CAM overlay",
            variable=self.show_gradcam_var,
            command=self._on_gradcam_toggle,
        ).pack(anchor=tk.W)

        # Export
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(
            sidebar, text="Export current frame", command=self._export_frame
        ).pack(fill=tk.X)

        # Playlist navigation
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(sidebar, text="Playlist", font=("", 10, "bold")).pack(anchor=tk.W)
        nav = ttk.Frame(sidebar)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="◀ Prev", command=self._prev_video).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(nav, text="Next ▶", command=self._next_video).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        self.video_name_var = tk.StringVar(value="(no video)")
        ttk.Label(sidebar, textvariable=self.video_name_var, wraplength=210).pack(
            anchor=tk.W, pady=2
        )

        # ── Bottom bar: playback controls + status ────────────────────────
        bottom = ttk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

        self.play_btn = ttk.Button(
            bottom, text="▶ Play", command=self._toggle_play, width=8
        )
        self.play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="|◀ Step", command=self._step_back).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bottom, text="Step ▶|", command=self._step_forward).pack(
            side=tk.LEFT, padx=2
        )

        self.seek_var = tk.IntVar(value=0)
        self.seek_bar = ttk.Scale(
            bottom,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            command=self._on_seek,
        )
        self.seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.status_var = tk.StringVar(value="Ready.  Load a model and videos to begin.")
        ttk.Label(bottom, textvariable=self.status_var, anchor=tk.W).pack(
            side=tk.LEFT, padx=4
        )

    # ── Worker lifecycle ──────────────────────────────────────────────────

    def _start_worker(self) -> None:
        self._worker = PlaybackWorker(self.model_runner, self._result_queue)
        if self.class_names:
            self._worker.send_command("set_class_names", names=self.class_names)
        self._push_settings_to_worker()
        self._worker.start()

    def _push_settings_to_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.send_command("set_conf_threshold", value=self.conf_var.get())
        self._worker.send_command("set_overlay_masks", value=self.mask_var.get())
        self._worker.send_command("set_label_mode", value=self.label_mode_var.get())
        self._worker.send_command(
            "set_active_classes", class_ids=self._compute_active_ids()
        )

    # ── Queue polling (Tkinter main thread) ───────────────────────────────

    def _poll_queue(self) -> None:
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            result = _QUEUE_EMPTY

        if result is _QUEUE_EMPTY:
            pass  # no new frame yet
        elif result is None:
            # EOF sentinel from worker
            self._playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self._last_result = result  # type: ignore[assignment]
            self._render_result(result)  # type: ignore[arg-type]

        self.root.after(16, self._poll_queue)  # ~60 FPS polling rate

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render_result(self, result: FrameResult) -> None:
        # Track display FPS
        now = time.perf_counter()
        self._display_times.append(now)
        if len(self._display_times) > self._fps_window:
            self._display_times.pop(0)

        display_fps = 0.0
        if len(self._display_times) >= 2:
            span = self._display_times[-1] - self._display_times[0]
            if span > 0:
                display_fps = (len(self._display_times) - 1) / span

        bgr = result.rendered_bgr

        # Blend Grad-CAM overlay if active
        if self._show_gradcam and self._gradcam_overlay is not None:
            if self._gradcam_overlay.shape[:2] == bgr.shape[:2]:
                bgr = cv2.addWeighted(bgr, 0.6, self._gradcam_overlay, 0.4, 0)

        # Downscale for display using INTER_AREA (faster + sharper than PIL LANCZOS)
        dw, dh = self._display_w, self._display_h
        bgr_small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)

        # BGR → RGB → PIL → PhotoImage
        rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self._current_photo = photo   # prevent garbage collection

        # Update seek bar position
        if self.total_frames > 1:
            pos = int(result.frame_index / (self.total_frames - 1) * 1000)
            self.seek_var.set(pos)
        elif self.total_frames == 1:
            self.seek_var.set(0)

        # Status line
        self.status_var.set(
            f"Frame {result.frame_index}/{self.total_frames}  |  "
            f"Infer: {result.elapsed_ms:.0f} ms  |  "
            f"Display: {display_fps:.1f} fps"
        )

    # ── Playback controls ─────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self._worker is None:
            return
        if self._playing:
            self._worker.pause()
            self._playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self._worker.resume()
            self._playing = True
            self.play_btn.config(text="⏸ Pause")

    def _step_forward(self) -> None:
        if self._worker is None:
            return
        fi = (self._last_result.frame_index + 1) if self._last_result else 0
        self._worker.send_command("seek", frame_index=fi)
        self._worker.send_command("step_one")

    def _step_back(self) -> None:
        if self._worker is None:
            return
        fi = max(0, (self._last_result.frame_index - 1) if self._last_result else 0)
        self._worker.send_command("seek", frame_index=fi)
        self._worker.send_command("step_one")

    def _on_seek(self, _: object = None) -> None:
        if self._worker is None or self.total_frames <= 0:
            return
        pos = self.seek_var.get()
        fi = max(0, int(pos / 1000.0 * max(self.total_frames - 1, 0)))
        self._worker.send_command("seek", frame_index=fi)

    # ── Settings callbacks ────────────────────────────────────────────────

    def _on_settings_change(self, *_: object) -> None:
        conf = self.conf_var.get()
        self.conf_label.config(text=f"{conf:.2f}")
        if self._worker:
            self._worker.send_command("set_conf_threshold", value=conf)
            self._worker.send_command("set_overlay_masks", value=self.mask_var.get())
            self._worker.send_command("set_label_mode", value=self.label_mode_var.get())
        GRADCAM_CACHE.clear()

    def _on_filter_change(self, *_: object) -> None:
        if self._worker:
            self._worker.send_command(
                "set_active_classes", class_ids=self._compute_active_ids()
            )

    def _compute_active_ids(self) -> List[int]:
        fr = self.filter_from_var.get()
        to = self.filter_to_var.get()
        n = len(self.class_names) if self.class_names else 230
        return list(range(max(0, fr), min(to + 1, n)))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._display_w = event.width
        self._display_h = event.height

    # ── Grad-CAM (on-demand only) ─────────────────────────────────────────

    def _compute_gradcam(self) -> None:
        """Compute and display Grad-CAM for the current frame, on demand.

        Grad-CAM is NEVER computed during normal playback; it is triggered only
        by this button.  The result is cached so repeated renders while paused
        do not recompute.
        """
        result = self._last_result
        if result is None or result.logits is None or result.meta is None:
            messagebox.showinfo(
                "Grad-CAM",
                "No inference result available for the current frame.\n"
                "Make sure a model is loaded and the video is paused on a frame.",
            )
            return

        cid = self.gradcam_class_var.get()
        vpath = self._current_video_path()

        self._gradcam_overlay = GRADCAM_CACHE.get(
            video_path=vpath,
            frame_index=result.frame_index,
            class_id=cid,
            logits=result.logits,
            meta=result.meta,
        )
        self._show_gradcam = True
        self.show_gradcam_var.set(True)

        # Re-render immediately with the overlay
        self._render_result(result)
        self.status_var.set(
            f"Grad-CAM computed for class {cid} on frame {result.frame_index}"
        )

    def _on_gradcam_toggle(self) -> None:
        self._show_gradcam = self.show_gradcam_var.get()
        if self._last_result:
            self._render_result(self._last_result)

    # ── Model and video loading ────────────────────────────────────────────

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select TFLite model",
            filetypes=[("TFLite model", "*.tflite"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.model_runner = ModelRunner(path)
        except Exception as exc:
            messagebox.showerror("Model load error", str(exc))
            return

        # Try to load class names from a companion .txt file
        names_path = Path(path).with_suffix(".txt")
        if names_path.exists():
            with open(names_path) as fh:
                self.class_names = [ln.strip() for ln in fh if ln.strip()]
            self.filter_to_var.set(len(self.class_names) - 1)
        else:
            nc = self.model_runner.num_classes or 230
            self.class_names = [f"class_{i}" for i in range(nc)]

        # Restart worker with new model
        if self._worker:
            self._worker.stop()
        self._result_queue = queue.Queue(maxsize=2)
        self._worker = PlaybackWorker(self.model_runner, self._result_queue)
        self._worker.send_command("set_class_names", names=self.class_names)
        self._push_settings_to_worker()
        self._worker.start()

        GRADCAM_CACHE.clear()
        self.status_var.set(f"Model loaded: {Path(path).name}  ({len(self.class_names)} classes)")

    def _load_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select videos",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self.video_paths = list(paths)
        self.video_idx = 0
        self._open_current_video()

    def _open_current_video(self) -> None:
        if not self.video_paths:
            return
        path = self.video_paths[self.video_idx]

        cap = cv2.VideoCapture(path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.video_name_var.set(
            f"[{self.video_idx + 1}/{len(self.video_paths)}] {Path(path).name}"
        )

        if self._worker:
            self._worker.send_command("set_video", path=path)

        GRADCAM_CACHE.clear()
        self._gradcam_overlay = None
        self._show_gradcam = False
        self.show_gradcam_var.set(False)

    def _current_video_path(self) -> str:
        if self.video_paths:
            return self.video_paths[self.video_idx]
        return ""

    def _prev_video(self) -> None:
        if self.video_paths and self.video_idx > 0:
            self.video_idx -= 1
            self._open_current_video()

    def _next_video(self) -> None:
        if self.video_paths and self.video_idx < len(self.video_paths) - 1:
            self.video_idx += 1
            self._open_current_video()

    # ── Export ────────────────────────────────────────────────────────────

    def _export_frame(self) -> None:
        result = self._last_result
        if result is None:
            messagebox.showinfo("Export", "No frame to export.")
            return

        vpath = self._current_video_path()
        default_name = (
            f"{Path(vpath).stem}_frame{result.frame_index:06d}.png"
            if vpath
            else "frame.png"
        )
        save_path = filedialog.asksaveasfilename(
            title="Export frame",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if not save_path:
            return

        bgr = result.rendered_bgr
        if self._show_gradcam and self._gradcam_overlay is not None:
            if self._gradcam_overlay.shape[:2] == bgr.shape[:2]:
                bgr = cv2.addWeighted(bgr, 0.6, self._gradcam_overlay, 0.4, 0)

        cv2.imwrite(save_path, bgr)
        self.status_var.set(f"Exported: {save_path}")

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2.0)
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    root = tk.Tk()
    root.geometry(f"{_DEFAULT_DISPLAY_W + 250}x{_DEFAULT_DISPLAY_H + 110}")
    root.minsize(700, 500)
    VideoReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
