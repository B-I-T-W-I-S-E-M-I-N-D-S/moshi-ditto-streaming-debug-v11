"""
pipeline/ditto_stream_adapter.py
==================================
Wraps Ditto's ``StreamSDK`` (online mode from stream_pipeline_online.py)
to intercept rendered video frames instead of writing them to an MP4 file.

How it works
------------
1. On ``setup()``, ``StreamSDK.setup()`` is called with ``online_mode=True``.
   The SDK spins up its internal threading pipeline:
       audio2motion → motion_stitch → warp_f3d → decode_f3d → putback → writer

2. We **monkey-patch** the ``writer_worker`` so that rendered frames
   (numpy uint8 RGB arrays) are pushed into a ``threading.Queue`` instead
   of being written to disk.

3. The caller pushes feature chunks via ``push_features(feat_np)`` —
   this feeds directly into ``sdk.audio2motion_queue``, bypassing HuBERT
   extraction entirely.

4. ``iter_frames()`` is a blocking generator that yields encoded JPEG bytes
   for each rendered frame.  Run it in a thread or via
   ``asyncio.to_thread()``.

Usage
-----
    adapter = DittoStreamAdapter(cfg_pkl=..., data_root=...)
    adapter.setup(image_path="/workspace/portrait.jpg")

    # In one thread: push feature chunks as they arrive
    adapter.push_features(feat_chunk_np)   # (N, 1024) float32

    # In another thread / async task: consume frames
    for jpeg_bytes in adapter.iter_frames():
        await websocket.send_bytes(b"\\x02" + jpeg_bytes)

    adapter.close()
"""

import io
import logging
import os
import queue
import sys
import threading
import time
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Latency tracker (optional)
try:
    from pipeline.latency_tracker import PipelineTracker
except ImportError:
    try:
        from latency_tracker import PipelineTracker
    except ImportError:
        PipelineTracker = None  # type: ignore

# ---------------------------------------------------------------------------
# Ensure ditto-inference is importable
# ---------------------------------------------------------------------------
_DITTO_DIR = os.path.join(os.path.dirname(__file__), "..", "ditto-inference")
if _DITTO_DIR not in sys.path:
    sys.path.insert(0, _DITTO_DIR)

from stream_pipeline_online import StreamSDK  # online version (supports streaming)


# ---------------------------------------------------------------------------
# JPEG encoder helper
# ---------------------------------------------------------------------------

def _encode_jpeg(rgb_array: np.ndarray, quality: int = 80) -> bytes:
    """Encode an HWC uint8 RGB numpy array as JPEG bytes."""
    try:
        from PIL import Image
        img = Image.fromarray(rgb_array, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        import cv2
        bgr = rgb_array[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()


# ---------------------------------------------------------------------------
# DittoStreamAdapter
# ---------------------------------------------------------------------------

def _ts_ditto() -> str:
    """Return HH:MM:SS.mmm timestamp string for Ditto log lines."""
    t = time.time()
    lt = time.localtime(t)
    ms = int((t % 1) * 1000)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"


_SENTINEL = object()   # signals end-of-stream in frame_queue


class DittoStreamAdapter:
    """
    Wraps StreamSDK for real-time per-frame output instead of file writing.

    Parameters
    ----------
    cfg_pkl   : path to Ditto .pkl config file
    data_root : path to Ditto TRT model directory
    jpeg_quality : JPEG encoding quality for streamed frames (default 80)
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        jpeg_quality: int = 80,
        tracker: Optional["PipelineTracker"] = None,
    ):
        cfg_pkl   = os.path.abspath(cfg_pkl)
        data_root = os.path.abspath(data_root)

        if not os.path.isfile(cfg_pkl):
            raise FileNotFoundError(f"Ditto config .pkl not found: {cfg_pkl}")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Ditto TRT model directory not found: {data_root}")

        logger.info(f"[DittoStreamAdapter] Loading StreamSDK from {data_root} …")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.jpeg_quality = jpeg_quality
        self.tracker: Optional["PipelineTracker"] = tracker
        self._frame_queue: Optional[queue.Queue] = None
        self._is_setup = False
        logger.info("[DittoStreamAdapter] StreamSDK loaded.")

    # ------------------------------------------------------------------
    # Setup for a session (one portrait image)
    # ------------------------------------------------------------------

    def setup(
        self,
        image_path: str,
        N_d: int = 10_000,      # large upper-bound; adapter closes when bridge is done
        emo: int = 4,
        sampling_timesteps: int = 50,
        overlap_v2: int = 10,
    ):
        """
        Initialise Ditto SDK for a new session and monkey-patch the writer
        to capture frames into a queue.

        Parameters
        ----------
        image_path        : path to the portrait image (.jpg / .png)
        N_d               : max expected frames (set large; we stop by closing)
        emo               : emotion index (Ditto default = 4)
        sampling_timesteps: diffusion steps (lower = faster, lower quality)
        overlap_v2        : overlap frames for the sliding window (default 10)
        """
        image_path = os.path.abspath(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Portrait image not found: {image_path}")

        self._frame_queue = queue.Queue(maxsize=200)

        # We pass a dummy output path; the writer is monkey-patched below
        _DUMMY_OUT = "/tmp/ditto_stream_dummy.mp4"

        self.sdk.setup(
            source_path        = image_path,
            output_path        = _DUMMY_OUT,
            online_mode        = True,
            N_d                = N_d,
            emo                = emo,
            sampling_timesteps = sampling_timesteps,
            overlap_v2         = overlap_v2,
        )

        # Monkey-patch the SDK's writer_worker so frames go to our queue
        # instead of the VideoWriterByImageIO.
        self._patch_writer_worker()

        self._is_setup = True
        logger.info(f"[DittoStreamAdapter] Session ready for image: {image_path}")

    # ------------------------------------------------------------------
    # Monkey-patch the writer worker
    # ------------------------------------------------------------------

    def _patch_writer_worker(self):
        """
        Replace the SDK's writer_worker with our own so rendered frames
        land in self._frame_queue rather than on disk.
        Adds fine-grained timing for writer_wait, jpeg_encode, and emit.
        """
        frame_queue   = self._frame_queue
        jpeg_quality  = self.jpeg_quality
        stop_event    = self.sdk.stop_event
        writer_queue  = self.sdk.writer_queue
        tracker       = self.tracker

        # Frame counter for in-thread FPS display
        _frame_times: list = []
        _MAX_HIST = 60   # keep last 60 frames for in-thread FPS

        def _patched_writer_worker():
            nonlocal _frame_times
            frame_idx = 0
            try:
                while not stop_event.is_set():
                    # ── Wait for next frame from Ditto's internal pipeline ─
                    _t_wait = time.perf_counter()
                    try:
                        item = writer_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    writer_wait_ms = (time.perf_counter() - _t_wait) * 1_000

                    if item is None:
                        break

                    rgb = item   # numpy uint8 HWC RGB

                    # ── JPEG encode ─────────────────────────────────
                    _t_enc = time.perf_counter()
                    try:
                        jpeg = _encode_jpeg(rgb, quality=jpeg_quality)
                    except Exception as enc_err:
                        logger.error(
                            f"[DittoStreamAdapter] JPEG encode error: {enc_err}"
                        )
                        continue
                    jpeg_encode_ms = (time.perf_counter() - _t_enc) * 1_000

                    # ── Emit frame to output queue ──────────────────────
                    _t_emit = time.perf_counter()
                    frame_queue.put(jpeg)
                    emit_ms = (time.perf_counter() - _t_emit) * 1_000

                    # ── Record to tracker ──────────────────────────────
                    if tracker is not None:
                        tracker.record_ditto(
                            writer_wait_ms = writer_wait_ms,
                            jpeg_encode_ms = jpeg_encode_ms,
                            emit_ms        = emit_ms,
                        )
                    else:
                        # Fallback: log raw timing at debug level
                        total_ms = writer_wait_ms + jpeg_encode_ms + emit_ms
                        logger.debug(
                            f"[Ditto] frame#{frame_idx}"
                            f"  total={total_ms:.1f}ms"
                            f"  writer_wait={writer_wait_ms:.1f}ms"
                            f"  jpeg_encode={jpeg_encode_ms:.1f}ms"
                            f"  emit={emit_ms:.2f}ms"
                        )

                    # ── In-thread rolling FPS ───────────────────────────
                    now = time.perf_counter()
                    _frame_times.append(now)
                    if len(_frame_times) > _MAX_HIST:
                        _frame_times = _frame_times[-_MAX_HIST:]
                    if len(_frame_times) >= 2:
                        elapsed = _frame_times[-1] - _frame_times[0]
                        thread_fps = (len(_frame_times) - 1) / elapsed if elapsed > 0 else 0
                        if frame_idx % 25 == 0:   # log every 25 frames
                            logger.info(
                                f"[{_ts_ditto()}] [Ditto] "
                                f"frame#{frame_idx+1}  "
                                f"fps={thread_fps:.1f}  "
                                f"writer_wait={writer_wait_ms:.1f}ms  "
                                f"jpeg={jpeg_encode_ms:.1f}ms"
                            )

                    frame_idx += 1

            finally:
                frame_queue.put(_SENTINEL)   # signal end of stream
                logger.info(
                    f"[DittoStreamAdapter] writer worker exited  "
                    f"total_frames={frame_idx}"
                )

        # Replace the already-started thread.
        # The SDK starts threads in setup() → we replace the last one
        # (writer_worker), which is index -1 in thread_list.
        writer_thread = self.sdk.thread_list[-1]
        writer_thread.join(timeout=0)   # shouldn't be blocked yet

        new_thread = threading.Thread(target=_patched_writer_worker, daemon=True)
        self.sdk.thread_list[-1] = new_thread
        new_thread.start()

    # ------------------------------------------------------------------
    # Feature input
    # ------------------------------------------------------------------

    def push_features(self, features: np.ndarray):
        """
        Push a chunk of HuBERT-like features into the Ditto audio2motion queue.

        Parameters
        ----------
        features : numpy (N, 1024) float32 — bridge module output
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before push_features().")
        features = features.astype(np.float32)
        if features.ndim != 2 or features.shape[1] != 1024:
            raise ValueError(
                f"Expected (N, 1024) features, got {features.shape}"
            )
        self.sdk.audio2motion_queue.put(features)

    # ------------------------------------------------------------------
    # Frame output
    # ------------------------------------------------------------------

    def iter_frames(self) -> Iterator[bytes]:
        """
        Blocking generator that yields JPEG-encoded frame bytes.
        Stops when the SDK signals end-of-stream.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before iter_frames().")

        while True:
            item = self._frame_queue.get()
            if item is _SENTINEL:
                break
            yield item   # JPEG bytes

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self):
        """
        Signal that no more features will be pushed and wait for the
        SDK thread pipeline to drain.
        """
        if not self._is_setup:
            return
        logger.info("[DittoStreamAdapter] Closing SDK …")
        self.sdk.close()   # puts None into audio2motion_queue, joins threads
        self._is_setup = False
        logger.info("[DittoStreamAdapter] Closed.")
