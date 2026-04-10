"""
pipeline/latency_tracker.py
============================
Centralised, thread-safe latency and FPS tracker for the
Moshi ─► Bridge ─► Ditto ─► WebSocket pipeline.

Key features
-------------
* Stopwatch helpers for fine-grained per-stage timing.
* Rolling-window FPS calculator (default: last 30 frames).
* Moving-average per-module latency tracking.
* Threshold-based bottleneck warning system.
* Structured log lines with [TIMESTAMP] prefix.
* Queue-depth snapshot for backpressure detection.
* JSON summary at session end (printed to logger).

Usage (inside streaming_server.py / adapters)
----------------------------------------------
    from pipeline.latency_tracker import PipelineTracker, Stopwatch

    tracker = PipelineTracker(session_id="abc123")

    # ─ Moshi ──────────────────────────────────────────────────────────
    with Stopwatch() as sw:
        codes = mimi.encode(chunk)
    tracker.record_moshi(
        mimi_encode_ms  = sw.ms,
        lm_step_ms      = ...,
        token_emit_ms   = ...,
        audio_decode_ms = ...,
    )

    # ─ Bridge ─────────────────────────────────────────────────────────
    tracker.record_bridge(
        queue_wait_ms = ...,
        transform_ms  = ...,
        push_ms       = ...,
    )

    # ─ Ditto ──────────────────────────────────────────────────────────
    tracker.record_ditto(
        writer_wait_ms = ...,
        jpeg_encode_ms = ...,
        emit_ms        = ...,
    )

    # ─ Sender (WebSocket) ─────────────────────────────────────────────
    tracker.record_sender(
        frame_wait_ms = ...,
        ws_send_ms    = ...,
    )

    # ─ Queue depths ───────────────────────────────────────────────────
    tracker.snapshot_queues(token_q=token_queue.qsize(),
                            frame_q=frame_queue.qsize())

    # ─ Session end ────────────────────────────────────────────────────
    tracker.log_summary()
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("latency_tracker")

# ─────────────────────────────────────────────────────────────────────────────
# Performance thresholds  (all in milliseconds unless noted)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS: Dict[str, float] = {
    # Moshi
    "moshi_total":    25.0,   # warn if full Moshi frame > 25 ms  (target ≤ 15 ms)
    "mimi_encode":    10.0,
    "lm_step":        15.0,
    "audio_decode":    8.0,
    # Bridge
    "bridge_total":   20.0,   # warn if full bridge step > 20 ms  (target ≤ 10 ms)
    "bridge_queue_wait": 30.0,
    "bridge_transform":  15.0,
    # Ditto
    "ditto_total":    60.0,   # warn if full Ditto frame > 60 ms  (target ≤ 40 ms)
    "ditto_writer_wait": 30.0,
    "jpeg_encode":     8.0,
    # Sender
    "ws_send":         8.0,
    # Pipeline
    "fps_warn":       15.0,   # warn if FPS drops below this
    "fps_target":     20.0,   # ideal minimum FPS
    # Queue depth (count)
    "token_q_warn":   200,
    "frame_q_warn":   100,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stopwatch context manager
# ─────────────────────────────────────────────────────────────────────────────

class Stopwatch:
    """
    Simple context-manager stopwatch.

        with Stopwatch() as sw:
            do_something()
        elapsed_ms = sw.ms
    """

    def __init__(self):
        self._start: float = 0.0
        self.ms: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self._start) * 1_000

    def elapsed_ms(self) -> float:
        """Return elapsed ms without stopping (for mid-flight reads)."""
        return (time.perf_counter() - self._start) * 1_000


def _ts() -> str:
    """Return a compact timestamp string  HH:MM:SS.mmm"""
    t = time.time()
    lt = time.localtime(t)
    ms = int((t % 1) * 1000)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# Rolling FPS calculator
# ─────────────────────────────────────────────────────────────────────────────

class RollingFPS:
    """Thread-safe rolling FPS tracker (based on frame timestamps)."""

    def __init__(self, window: int = 30):
        self._window = window
        self._times: Deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self) -> float:
        """Record a frame and return the current FPS."""
        now = time.perf_counter()
        with self._lock:
            self._times.append(now)
            cutoff = now - 5.0          # keep at most 5s of history
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            n = len(self._times)
            if n < 2:
                return 0.0
            window_n = min(n, self._window)
            elapsed = self._times[-1] - self._times[-window_n]
            return (window_n - 1) / elapsed if elapsed > 0 else 0.0

    def current(self) -> float:
        """Return current FPS without recording a new frame."""
        with self._lock:
            n = len(self._times)
            if n < 2:
                return 0.0
            window_n = min(n, self._window)
            elapsed = self._times[-1] - self._times[-window_n]
            return (window_n - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MovingAverage helper
# ─────────────────────────────────────────────────────────────────────────────

class MovingAverage:
    """Thread-safe exponential moving average (EMA)."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._value: Optional[float] = None
        self._lock = threading.Lock()

    def update(self, x: float) -> float:
        with self._lock:
            if self._value is None:
                self._value = x
            else:
                self._value = self._alpha * x + (1 - self._alpha) * self._value
            return self._value

    @property
    def value(self) -> float:
        with self._lock:
            return self._value or 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PipelineTracker  (main public class)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineTracker:
    """
    Thread-safe pipeline latency tracker.

    One instance should be created per WebSocket session and passed into
    each pipeline stage (Moshi, Bridge, Ditto, Sender).

    Parameters
    ----------
    session_id    : string identifier for log lines
    fps_window    : number of frames for rolling FPS
    log_every     : emit a [PIPELINE] summary line every N frames
    verbose       : if True, log every module record (not just pipeline lines)
    """

    def __init__(
        self,
        session_id: str = "default",
        fps_window: int = 30,
        log_every: int = 25,
        verbose: bool = True,
    ):
        self.session_id = session_id
        self.log_every  = log_every
        self.verbose    = verbose

        self._lock = threading.Lock()

        # Rolling metrics
        self._fps     = RollingFPS(window=fps_window)
        self._frame_count = 0

        # Per-module EMA latencies
        self._ema: Dict[str, MovingAverage] = {
            "moshi_total":     MovingAverage(),
            "bridge_total":    MovingAverage(),
            "ditto_total":     MovingAverage(),
            "pipeline_total":  MovingAverage(),
        }

        # Last queue depth snapshot
        self._token_q: int = 0
        self._frame_q: int = 0

        # History lists for end-of-session summary (capped at 2000 entries)
        self._history_fps:      List[float] = []
        self._history_pipeline: List[float] = []
        self._worst_frames: List[dict] = []   # top-5 slowest frames

        # Session wall-clock start
        self._session_start = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────────
    # Module recorders
    # ──────────────────────────────────────────────────────────────────────────

    def record_moshi(
        self,
        mimi_encode_ms:  float,
        lm_step_ms:      float,
        token_emit_ms:   float = 0.0,
        audio_decode_ms: float = 0.0,
    ) -> None:
        """Record one Moshi frame's latency breakdown."""
        total = mimi_encode_ms + lm_step_ms + audio_decode_ms + token_emit_ms
        self._ema["moshi_total"].update(total)

        if not self.verbose:
            return

        line = (
            f"[{_ts()}] [Moshi] "
            f"chunk={total:.1f}ms  "
            f"mimi_encode={mimi_encode_ms:.1f}ms  "
            f"lm_step={lm_step_ms:.1f}ms  "
            f"audio_decode={audio_decode_ms:.1f}ms  "
            f"token_emit={token_emit_ms:.2f}ms"
        )
        logger.info(line)
        self._check_threshold("moshi_total",  total,         "Moshi full frame")
        self._check_threshold("mimi_encode",  mimi_encode_ms, "Moshi mimi_encode")
        self._check_threshold("lm_step",      lm_step_ms,     "Moshi lm_step")
        self._check_threshold("audio_decode", audio_decode_ms,"Moshi audio_decode")

    def record_bridge(
        self,
        queue_wait_ms: float,
        transform_ms:  float,
        push_ms:       float = 0.0,
    ) -> None:
        """Record one Bridge step's latency breakdown."""
        total = queue_wait_ms + transform_ms + push_ms
        self._ema["bridge_total"].update(total)

        if not self.verbose:
            return

        line = (
            f"[{_ts()}] [Bridge] "
            f"total={total:.1f}ms  "
            f"queue_wait={queue_wait_ms:.1f}ms  "
            f"transform={transform_ms:.1f}ms  "
            f"push={push_ms:.2f}ms"
        )
        logger.info(line)
        self._check_threshold("bridge_total",     total,         "Bridge full step")
        self._check_threshold("bridge_queue_wait",queue_wait_ms, "Bridge queue_wait")
        self._check_threshold("bridge_transform", transform_ms,  "Bridge transform")

    def record_ditto(
        self,
        writer_wait_ms: float,
        jpeg_encode_ms: float,
        emit_ms:        float = 0.0,
    ) -> None:
        """Record one Ditto frame's latency breakdown."""
        total = writer_wait_ms + jpeg_encode_ms + emit_ms
        self._ema["ditto_total"].update(total)

        if not self.verbose:
            return

        line = (
            f"[{_ts()}] [Ditto] "
            f"total={total:.1f}ms  "
            f"writer_wait={writer_wait_ms:.1f}ms  "
            f"jpeg_encode={jpeg_encode_ms:.1f}ms  "
            f"emit={emit_ms:.2f}ms"
        )
        logger.info(line)
        self._check_threshold("ditto_total",      total,          "Ditto full frame")
        self._check_threshold("ditto_writer_wait",writer_wait_ms, "Ditto writer_wait")
        self._check_threshold("jpeg_encode",      jpeg_encode_ms, "Ditto jpeg_encode")

    def record_sender(
        self,
        frame_wait_ms: float,
        ws_send_ms:    float,
        pipeline_total_ms: Optional[float] = None,
    ) -> None:
        """
        Record one WebSocket send event (called when a frame is sent).
        Also ticks the FPS counter and emits the periodic [PIPELINE] summary.

        Parameters
        ----------
        frame_wait_ms     : time blocked waiting for a frame from Ditto
        ws_send_ms        : time taken by websocket.send_bytes()
        pipeline_total_ms : optional end-to-end latency (Moshi-token → frame sent)
        """
        fps = self._fps.tick()

        with self._lock:
            self._frame_count += 1
            fc = self._frame_count

        if pipeline_total_ms is not None:
            self._ema["pipeline_total"].update(pipeline_total_ms)

        if self.verbose:
            line = (
                f"[{_ts()}] [Sender] "
                f"frame_wait={frame_wait_ms:.1f}ms  "
                f"ws_send={ws_send_ms:.1f}ms"
            )
            if pipeline_total_ms is not None:
                line += f"  pipe_total={pipeline_total_ms:.1f}ms"
            logger.info(line)

        self._check_threshold("ws_send", ws_send_ms, "Sender ws_send")

        # ── Per-frame history (for summary) ──────────────────────────────────
        if len(self._history_fps) < 2000:
            self._history_fps.append(fps)
        if pipeline_total_ms is not None and len(self._history_pipeline) < 2000:
            self._history_pipeline.append(pipeline_total_ms)

        # ── Track worst frames ────────────────────────────────────────────────
        if pipeline_total_ms is not None:
            entry = {
                "frame": fc,
                "pipeline_ms": pipeline_total_ms,
                "fps": fps,
                "token_q": self._token_q,
                "frame_q": self._frame_q,
            }
            self._worst_frames.append(entry)
            self._worst_frames.sort(key=lambda x: x["pipeline_ms"], reverse=True)
            self._worst_frames = self._worst_frames[:5]

        # ── FPS warning ───────────────────────────────────────────────────────
        if fps > 0 and fps < THRESHOLDS["fps_warn"]:
            logger.warning(
                f"[{_ts()}] ⚠️  [FPS_LOW] fps={fps:.1f} < threshold={THRESHOLDS['fps_warn']:.0f}"
                f"  (target={THRESHOLDS['fps_target']:.0f})"
            )

        # ── Periodic pipeline summary ─────────────────────────────────────────
        if fc % self.log_every == 0:
            self._log_pipeline_line(fps, pipeline_total_ms)

    def snapshot_queues(self, token_q: int, frame_q: int) -> None:
        """
        Record current queue depths.  Call this from the main asyncio loop
        or a monitoring task every N seconds.
        """
        with self._lock:
            self._token_q = token_q
            self._frame_q = frame_q

        if token_q > THRESHOLDS["token_q_warn"]:
            logger.warning(
                f"[{_ts()}] ⚠️  [BACKPRESSURE] token_queue depth={token_q}"
                f" > threshold={int(THRESHOLDS['token_q_warn'])}"
                "  → Bridge is falling behind Moshi"
            )
        if frame_q > THRESHOLDS["frame_q_warn"]:
            logger.warning(
                f"[{_ts()}] ⚠️  [BACKPRESSURE] frame_queue depth={frame_q}"
                f" > threshold={int(THRESHOLDS['frame_q_warn'])}"
                "  → Sender is falling behind Ditto"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline summary line  (logged every `log_every` frames)
    # ──────────────────────────────────────────────────────────────────────────

    def _log_pipeline_line(
        self,
        fps: float,
        pipeline_total_ms: Optional[float],
    ) -> None:
        moshi_avg   = self._ema["moshi_total"].value
        bridge_avg  = self._ema["bridge_total"].value
        ditto_avg   = self._ema["ditto_total"].value
        pipe_avg    = self._ema["pipeline_total"].value

        fps_status  = "✅" if fps >= THRESHOLDS["fps_target"] else (
                      "⚠️ " if fps >= THRESHOLDS["fps_warn"] else "🔴")

        line = (
            f"[{_ts()}] [PIPELINE]"
            f"  frame={self._frame_count}"
            f"  fps={fps:.1f} {fps_status}"
            f"  avg_pipe={pipe_avg:.0f}ms"
            f"  moshi_avg={moshi_avg:.0f}ms"
            f"  bridge_avg={bridge_avg:.0f}ms"
            f"  ditto_avg={ditto_avg:.0f}ms"
            f"  token_q={self._token_q}"
            f"  frame_q={self._frame_q}"
        )
        logger.info(line)

        # ── Bottleneck identification ──────────────────────────────────────────
        bottlenecks = []
        if moshi_avg  > THRESHOLDS["moshi_total"]:
            bottlenecks.append(f"Moshi={moshi_avg:.0f}ms(>{THRESHOLDS['moshi_total']:.0f}ms)")
        if bridge_avg > THRESHOLDS["bridge_total"]:
            bottlenecks.append(f"Bridge={bridge_avg:.0f}ms(>{THRESHOLDS['bridge_total']:.0f}ms)")
        if ditto_avg  > THRESHOLDS["ditto_total"]:
            bottlenecks.append(f"Ditto={ditto_avg:.0f}ms(>{THRESHOLDS['ditto_total']:.0f}ms)")

        if bottlenecks:
            logger.warning(
                f"[{_ts()}] 🔴 [BOTTLENECK] Slow modules: {', '.join(bottlenecks)}"
            )
        elif fps >= THRESHOLDS["fps_target"]:
            logger.info(
                f"[{_ts()}] ✅ [PERF_OK] fps={fps:.1f} meets target "
                f"{THRESHOLDS['fps_target']:.0f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Threshold checker (internal)
    # ──────────────────────────────────────────────────────────────────────────

    def _check_threshold(self, key: str, value: float, label: str) -> None:
        threshold = THRESHOLDS.get(key)
        if threshold is not None and value > threshold:
            logger.warning(
                f"[{_ts()}] ⚠️  [SLOW] {label}={value:.1f}ms "
                f"exceeds threshold={threshold:.0f}ms"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Session-end summary
    # ──────────────────────────────────────────────────────────────────────────

    def log_summary(self) -> None:
        """
        Emit a comprehensive JSON session summary at end-of-session.
        This is the primary tool for post-session bottleneck analysis.
        """
        duration = time.perf_counter() - self._session_start
        fps_list  = self._history_fps
        pipe_list = self._history_pipeline

        summary = {
            "session_id":     self.session_id,
            "duration_s":     round(duration, 2),
            "total_frames":   self._frame_count,
            "fps": {
                "avg":    round(statistics.mean(fps_list), 2)   if fps_list else 0,
                "median": round(statistics.median(fps_list), 2) if fps_list else 0,
                "min":    round(min(fps_list), 2)               if fps_list else 0,
                "max":    round(max(fps_list), 2)               if fps_list else 0,
                "p5":     round(sorted(fps_list)[int(0.05 * len(fps_list))], 2)
                          if len(fps_list) > 20 else 0,
            },
            "pipeline_latency_ms": {
                "avg":    round(statistics.mean(pipe_list), 2)   if pipe_list else 0,
                "median": round(statistics.median(pipe_list), 2) if pipe_list else 0,
                "p95":    round(sorted(pipe_list)[int(0.95 * len(pipe_list))], 2)
                          if len(pipe_list) > 20 else 0,
                "max":    round(max(pipe_list), 2) if pipe_list else 0,
            },
            "ema_at_close": {
                "moshi_total_ms":    round(self._ema["moshi_total"].value, 2),
                "bridge_total_ms":   round(self._ema["bridge_total"].value, 2),
                "ditto_total_ms":    round(self._ema["ditto_total"].value, 2),
                "pipeline_total_ms": round(self._ema["pipeline_total"].value, 2),
            },
            "targets": {
                "fps_target":      THRESHOLDS["fps_target"],
                "moshi_target_ms": THRESHOLDS["moshi_total"],
                "bridge_target_ms":THRESHOLDS["bridge_total"],
                "ditto_target_ms": THRESHOLDS["ditto_total"],
            },
            "top5_worst_frames": self._worst_frames,
        }

        # ── Pass/fail verdict ─────────────────────────────────────────────────
        avg_fps = summary["fps"]["avg"]
        if avg_fps >= THRESHOLDS["fps_target"]:
            verdict = f"✅ PASS  avg_fps={avg_fps:.1f} ≥ target={THRESHOLDS['fps_target']:.0f}"
        elif avg_fps >= THRESHOLDS["fps_warn"]:
            verdict = f"⚠️  MARGINAL  avg_fps={avg_fps:.1f} (target={THRESHOLDS['fps_target']:.0f})"
        else:
            verdict = f"🔴 FAIL  avg_fps={avg_fps:.1f} < warn_threshold={THRESHOLDS['fps_warn']:.0f}"

        logger.info(
            f"\n{'='*66}\n"
            f"  📊 SESSION LATENCY SUMMARY  [{self.session_id}]\n"
            f"  {verdict}\n"
            f"{'='*66}\n"
            + json.dumps(summary, indent=2) +
            f"\n{'='*66}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: log a single free-form annotation
    # ──────────────────────────────────────────────────────────────────────────

    def log_event(self, msg: str) -> None:
        """Log a free-form annotated event with a timestamp prefix."""
        logger.info(f"[{_ts()}] [EVENT] {msg}")

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def current_fps(self) -> float:
        return self._fps.current()
