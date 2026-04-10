"""
streaming_server.py
===================
FastAPI WebSocket server for the real-time Moshi + Bridge + Ditto pipeline.

Endpoints
---------
  GET  /                    → browser UI (static/index.html)
  POST /upload_image        → multipart upload; returns {"session_id": "..."}
  GET  /session/{sid}/status→ {"ready": true/false}
  WS   /ws/{session_id}     → bidirectional audio/video stream

WebSocket message protocol
--------------------------
  Browser → Server:
    0x01 <opus_bytes>      user's mic audio (Opus encoded)

  Server → Browser:
    0x00                   handshake / ready signal
    0x01 <opus_bytes>      Moshi response audio (Opus encoded)
    0x02 <jpeg_bytes>      animated talking-head video frame
    0x03 <utf8_text>       text token / transcript piece
    0xFF <utf8_error>      error message

Startup
-------
  python streaming_server.py [--host HOST] [--port PORT] [options]

  Or via uvicorn:
  uvicorn streaming_server:app --host 0.0.0.0 --port 7860

RunPod
------
  The server listens on 0.0.0.0 so RunPod's public port proxy can forward
  traffic.  Set port to match your RunPod "HTTP Port" setting (default 7860).
"""

import argparse
import asyncio
import collections
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from pipeline.latency_tracker import PipelineTracker

import torch
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.streaming_moshi import StreamingMoshiState
from pipeline.ditto_stream_adapter import DittoStreamAdapter

# Bridge imports
_BRIDGE_DIR = os.path.join(_ROOT, "bridge_module")
if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)
from inference import StreamingBridgeInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("streaming_server")

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration  (overridden by CLI args or environment variables)
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Moshi
    MOSHI_HF_REPO:     str = os.environ.get("MOSHI_HF_REPO", "kyutai/moshiko-pytorch-bf16")
    MOSHI_WEIGHT:      Optional[str] = os.environ.get("MOSHI_WEIGHT")
    MIMI_WEIGHT:       Optional[str] = os.environ.get("MIMI_WEIGHT")
    TOKENIZER:         Optional[str] = os.environ.get("MOSHI_TOKENIZER")

    # Bridge
    BRIDGE_CKPT:   str = os.environ.get("BRIDGE_CKPT",
                         os.path.join(_ROOT, "checkpoints", "bridge_best.pt"))
    BRIDGE_CONFIG: str = os.environ.get("BRIDGE_CONFIG",
                         os.path.join(_ROOT, "bridge_module", "config.yaml"))
    BRIDGE_CHUNK:  int = int(os.environ.get("BRIDGE_CHUNK", "4"))  # Mimi frames per chunk

    # Ditto
    DITTO_DATA_ROOT: str = os.environ.get("DITTO_DATA_ROOT",
                           os.path.join(_ROOT, "ditto-inference", "checkpoints",
                                        "ditto_trt_Ampere_Plus"))
    DITTO_CFG_PKL:   str = os.environ.get("DITTO_CFG_PKL",
                           os.path.join(_ROOT, "ditto-inference", "checkpoints",
                                        "ditto_cfg", "v0.4_hubert_cfg_trt.pkl"))
    DITTO_EMO:               int = int(os.environ.get("DITTO_EMO", "4"))
    # NOTE: Default lowered from 50 → 10.  50 diffusion steps give ~25 FPS
    # in offline (batched) mode but collapse to 6–9 FPS in streaming because
    # every pushed feature chunk triggers a full diffusion pass.  10 steps
    # are sufficient for real-time quality.  Override via env DITTO_SAMPLING_STEPS.
    DITTO_SAMPLING_STEPS:    int = int(os.environ.get("DITTO_SAMPLING_STEPS", "10"))
    DITTO_JPEG_QUALITY:      int = int(os.environ.get("DITTO_JPEG_QUALITY", "80"))

    # Runtime
    DEVICE:     str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE:      torch.dtype = torch.bfloat16
    UPLOAD_DIR: str = os.path.join(_ROOT, "_uploads")
    STATIC_DIR: str = os.path.join(_ROOT, "static")


cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded model singletons
# ─────────────────────────────────────────────────────────────────────────────

_moshi_state:    Optional[StreamingMoshiState]     = None
_bridge_stream:  Optional[StreamingBridgeInference] = None
_ditto_adapter:  Optional[DittoStreamAdapter]       = None

# Per-session state: session_id → {"image_path": str, "ready": bool}
_sessions: dict = {}


def get_moshi() -> StreamingMoshiState:
    global _moshi_state
    if _moshi_state is None:
        raise RuntimeError("Moshi model not loaded. Call /startup or wait for server init.")
    return _moshi_state


def get_bridge() -> StreamingBridgeInference:
    global _bridge_stream
    if _bridge_stream is None:
        raise RuntimeError("Bridge model not loaded.")
    return _bridge_stream


def get_ditto() -> DittoStreamAdapter:
    global _ditto_adapter
    if _ditto_adapter is None:
        raise RuntimeError("Ditto adapter not initialised.")
    return _ditto_adapter


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Moshi-Bridge-Ditto Streaming API", version="1.0.0")

# Serve static files (browser UI)
os.makedirs(cfg.STATIC_DIR, exist_ok=True)
os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)

# Mount static dir
app.mount("/static", StaticFiles(directory=cfg.STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the browser UI."""
    index_path = os.path.join(cfg.STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse("<h2>UI not found — place index.html in ./static/</h2>", status_code=404)
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ─────────────────────────────────────────────────────────────────────────────
# Image upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a portrait image.
    Returns {"session_id": "<uuid>", "filename": "<saved name>"}.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    session_id = str(uuid.uuid4())[:8]
    dest = os.path.join(cfg.UPLOAD_DIR, f"{session_id}{ext}")

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    _sessions[session_id] = {"image_path": dest, "ready": True}
    logger.info(f"[upload_image] Session {session_id} → {dest}")
    return JSONResponse({"session_id": session_id, "filename": os.path.basename(dest)})


@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# Health / info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": cfg.DEVICE,
        "moshi_loaded":  _moshi_state  is not None,
        "bridge_loaded": _bridge_stream is not None,
        "ditto_loaded":  _ditto_adapter is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[WS] Client connected — session {session_id}")

    # Validate session
    if session_id not in _sessions:
        await websocket.send_bytes(b"\xff" + b"Unknown session_id")
        await websocket.close()
        return

    session = _sessions[session_id]
    image_path = session["image_path"]

    moshi  = get_moshi()
    bridge = get_bridge()

    # ── Per-session latency tracker ───────────────────────────────────────────
    tracker = PipelineTracker(
        session_id = session_id,
        fps_window = 30,
        log_every  = 25,
        verbose    = True,
    )
    tracker.log_event(
        f"Session started  image={image_path}"
        f"  ditto_steps={cfg.DITTO_SAMPLING_STEPS}"
        f"  bridge_chunk={cfg.BRIDGE_CHUNK}"
        f"  jpeg_q={cfg.DITTO_JPEG_QUALITY}"
    )

    # Per-session Ditto adapter (new setup per connection so avatar is fresh)
    ditto = DittoStreamAdapter(
        cfg_pkl      = cfg.DITTO_CFG_PKL,
        data_root    = cfg.DITTO_DATA_ROOT,
        jpeg_quality = cfg.DITTO_JPEG_QUALITY,
        tracker      = tracker,          # ← pass tracker for Ditto timing
    )
    ditto.setup(
        image_path         = image_path,
        emo                = cfg.DITTO_EMO,
        sampling_timesteps = cfg.DITTO_SAMPLING_STEPS,
    )

    # Queue from Moshi → Bridge
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    # Queue for Ditto frames → WebSocket sender
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # Token-emit timestamp ring buffer: maps token seq# → wall-clock time
    # Used to compute end-to-end pipeline latency (Moshi-token → frame sent).
    # Sized to hold ~10 seconds of tokens at 12.5 Hz.
    _token_emit_times: collections.deque = collections.deque(maxlen=256)
    _token_seq   = 0   # incremented each time Moshi pushes a token
    _frame_seq   = 0   # incremented each time a frame is sent

    # Lock to serialize WebSocket writes and prevent protocol crashes
    ws_lock = asyncio.Lock()

    async def safe_send_bytes(data: bytes):
        async with ws_lock:
            try:
                await websocket.send_bytes(data)
            except RuntimeError as exc:
                if 'Cannot call "send" once a close message has been sent' in str(exc):
                    raise WebSocketDisconnect()
                raise

    # ── async receive wrapper ─────────────────────────────────────────────────
    async def receive_fn():
        try:
            data = await websocket.receive_bytes()
            return data
        except (WebSocketDisconnect, Exception):
            return None

    # ── Bridge task: token_queue → features → Ditto ──────────────────────────
    async def bridge_task():
        """Pull token chunks, run bridge streaming, push features to Ditto."""
        bridge.reset()
        token_buffer = []
        chunk_size = cfg.BRIDGE_CHUNK

        try:
            while True:
                # ── Wait for token from Moshi ───────────────────────
                _t_qwait = time.perf_counter()
                item = await token_queue.get()
                queue_wait_ms = (time.perf_counter() - _t_qwait) * 1_000

                if item is None:
                    # Flush remaining tokens
                    if token_buffer:
                        chunk = torch.cat(token_buffer, dim=0)
                        _t_tr = time.perf_counter()
                        features = await asyncio.to_thread(bridge.step, chunk)
                        transform_ms = (time.perf_counter() - _t_tr) * 1_000
                        features_np = features.numpy().astype(np.float32)
                        _t_push = time.perf_counter()
                        ditto.push_features(features_np)
                        push_ms = (time.perf_counter() - _t_push) * 1_000
                        tracker.record_bridge(queue_wait_ms=0,
                                              transform_ms=transform_ms,
                                              push_ms=push_ms)
                    break

                token_buffer.append(item)  # each is (1, 8)

                if len(token_buffer) >= chunk_size:
                    chunk = torch.cat(token_buffer, dim=0)  # (chunk_size, 8)
                    token_buffer = []

                    # ── Bridge inference (GPU, off event-loop) ────────
                    _t_tr = time.perf_counter()
                    features = await asyncio.to_thread(bridge.step, chunk)
                    transform_ms = (time.perf_counter() - _t_tr) * 1_000

                    features_np = features.numpy().astype(np.float32)

                    # ── Push features to Ditto ────────────────────
                    _t_push = time.perf_counter()
                    ditto.push_features(features_np)
                    push_ms = (time.perf_counter() - _t_push) * 1_000

                    tracker.record_bridge(
                        queue_wait_ms = queue_wait_ms,
                        transform_ms  = transform_ms,
                        push_ms       = push_ms,
                    )

        except Exception as exc:
            logger.exception(f"[bridge_task] Error: {exc}")
        finally:
            ditto.close()
            logger.info("[bridge_task] Done.")

    # ── Frame reader task: Ditto frames → frame_queue ────────────────────────
    async def frame_reader_task():
        """Read JPEG frames from Ditto (blocking iter) in a thread."""
        loop = asyncio.get_running_loop()
        try:
            def _blocking_iter():
                for jpeg in ditto.iter_frames():
                    loop.call_soon_threadsafe(frame_queue.put_nowait, jpeg)
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)  # sentinel

            await asyncio.to_thread(_blocking_iter)
        except Exception as exc:
            logger.exception(f"[frame_reader_task] Error: {exc}")
            await frame_queue.put(None)

    # ── Frame sender task: frame_queue → WebSocket ────────────────────────────
    async def frame_sender_task():
        """Send JPEG frames over WebSocket as 0x02 messages."""
        try:
            while True:
                # ── Wait for a rendered frame ──────────────────────
                _t_fwait = time.perf_counter()
                jpeg = await frame_queue.get()
                frame_wait_ms = (time.perf_counter() - _t_fwait) * 1_000

                if jpeg is None:
                    break

                # ── WebSocket send ─────────────────────────────
                _t_send = time.perf_counter()
                try:
                    await safe_send_bytes(b"\x02" + jpeg)
                except Exception:
                    break
                ws_send_ms = (time.perf_counter() - _t_send) * 1_000

                # ── E2E pipeline latency estimate ──────────────────
                # Approximate: use the oldest recorded token-emit timestamp
                # that hasn't been consumed yet as the "start" of this frame.
                pipeline_total_ms = None
                if _token_emit_times:
                    oldest_token_t = _token_emit_times.popleft()
                    pipeline_total_ms = (time.perf_counter() - oldest_token_t) * 1_000

                tracker.record_sender(
                    frame_wait_ms     = frame_wait_ms,
                    ws_send_ms        = ws_send_ms,
                    pipeline_total_ms = pipeline_total_ms,
                )

                # Snapshot queue depths for backpressure logging
                tracker.snapshot_queues(
                    token_q = token_queue.qsize(),
                    frame_q = frame_queue.qsize(),
                )

        except Exception as exc:
            logger.exception(f"[frame_sender_task] Error: {exc}")
        logger.info("[frame_sender_task] Done.")

    # ── Queue monitor task: log queue depths every 5 s ────────────────────────
    async def queue_monitor_task():
        """Periodically log queue depths to detect backpressure."""
        try:
            while True:
                await asyncio.sleep(5)
                tq = token_queue.qsize()
                fq = frame_queue.qsize()
                fps = tracker.current_fps
                logger.info(
                    f"[QUEUE_MONITOR] session={session_id}"
                    f"  token_q={tq}/{token_queue.maxsize}"
                    f"  frame_q={fq}/{frame_queue.maxsize}"
                    f"  fps={fps:.1f}"
                    f"  frames={tracker.frame_count}"
                )
                if tq > 400:
                    logger.warning(
                        f"[QUEUE_MONITOR] ⚠️  token_queue critically full ({tq}/500)"
                        "  — Moshi is producing faster than Bridge+Ditto can consume!"
                    )
                if fq > 150:
                    logger.warning(
                        f"[QUEUE_MONITOR] ⚠️  frame_queue critically full ({fq}/200)"
                        "  — WebSocket sender is falling behind Ditto output!"
                    )
        except asyncio.CancelledError:
            pass

    # ── Start background tasks ────────────────────────────────────────────────
    t_bridge        = asyncio.create_task(bridge_task())
    t_frame_reader  = asyncio.create_task(frame_reader_task())
    t_frame_sender  = asyncio.create_task(frame_sender_task())
    t_queue_monitor = asyncio.create_task(queue_monitor_task())

    # ── Moshi main loop (drives audio I/O + token capture) ───────────────────
    # Wrap token_queue.put_nowait to record emit timestamps for E2E latency.
    _orig_put_nowait = token_queue.put_nowait

    def _timed_put_nowait(item):
        if item is not None:
            _token_emit_times.append(time.perf_counter())
        return _orig_put_nowait(item)

    token_queue.put_nowait = _timed_put_nowait  # type: ignore[method-assign]

    try:
        async for kind, payload in moshi.handle_connection(
            receive_fn, token_queue, tracker=tracker
        ):
            if kind == "handshake":
                await safe_send_bytes(b"\x00")
            elif kind == "audio":
                await safe_send_bytes(b"\x01" + payload)
            elif kind == "text":
                await safe_send_bytes(b"\x03" + payload.encode("utf-8"))
    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected — session {session_id}")
    except Exception as exc:
        logger.exception(f"[WS] Unexpected error in session {session_id}: {exc}")
        try:
            await safe_send_bytes(b"\xff" + str(exc).encode())
        except Exception:
            pass
    finally:
        # Cancel queue monitor first (it loops forever)
        t_queue_monitor.cancel()
        # Wait for pipeline tasks to finish
        await asyncio.gather(t_bridge, t_frame_reader, t_frame_sender,
                             t_queue_monitor,
                             return_exceptions=True)
        # Emit session-end summary
        tracker.log_summary()
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[WS] Session {session_id} fully closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup event: load all models once
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global _moshi_state, _bridge_stream, _ditto_adapter

    logger.info("=" * 60)
    logger.info("  Moshi + Bridge + Ditto — Streaming Server Starting")
    logger.info("=" * 60)
    logger.info(f"  Device : {cfg.DEVICE}")

    # ── Moshi ────────────────────────────────────────────────────────────────
    logger.info("\n[1/3] Loading Moshi …")
    t0 = time.time()
    _moshi_state = StreamingMoshiState(
        hf_repo      = cfg.MOSHI_HF_REPO,
        moshi_weight = cfg.MOSHI_WEIGHT,
        mimi_weight  = cfg.MIMI_WEIGHT,
        tokenizer    = cfg.TOKENIZER,
        device       = cfg.DEVICE,
        dtype        = cfg.DTYPE,
    )
    _moshi_state.warmup()
    logger.info(f"[1/3] ✅ Moshi ready ({time.time()-t0:.1f}s)")

    # ── Bridge ───────────────────────────────────────────────────────────────
    logger.info("\n[2/3] Loading Bridge …")
    t0 = time.time()
    _bridge_stream = StreamingBridgeInference(
        checkpoint_path = cfg.BRIDGE_CKPT,
        config_path     = cfg.BRIDGE_CONFIG,
        chunk_size      = cfg.BRIDGE_CHUNK,
        device          = cfg.DEVICE,
    )
    logger.info(f"[2/3] ✅ Bridge ready ({time.time()-t0:.1f}s)")

    # ── Ditto ────────────────────────────────────────────────────────────────
    # NOTE: DittoStreamAdapter is instantiated per-session to get a fresh SDK
    # per portrait image.  We do NOT pre-load it here because StreamSDK.setup()
    # must be called with a specific image before threads start.
    logger.info("\n[3/3] Ditto: adapter will be created per-session (image-specific).")
    logger.info(f"       TRT models: {cfg.DITTO_DATA_ROOT}")

    logger.info("\n" + "=" * 60)
    logger.info("  ✅  All models loaded. Server ready for connections.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Moshi + Bridge + Ditto — Real-Time Streaming Server"
    )
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=7860)
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])

    # Model path overrides (all have env-var equivalents, see Config above)
    p.add_argument("--hf-repo",         default=None)
    p.add_argument("--moshi-weight",    default=None)
    p.add_argument("--mimi-weight",     default=None)
    p.add_argument("--tokenizer",       default=None)
    p.add_argument("--bridge-ckpt",     default=None)
    p.add_argument("--bridge-config",   default=None)
    p.add_argument("--bridge-chunk",    type=int, default=None)
    p.add_argument("--ditto-data-root", default=None)
    p.add_argument("--ditto-cfg-pkl",   default=None)
    p.add_argument("--ditto-emo",            type=int, default=None)
    p.add_argument("--ditto-sampling-steps", type=int, default=None)
    p.add_argument("--jpeg-quality",    type=int, default=None)
    p.add_argument("--half", action="store_const",
                   const=torch.float16, default=None, dest="dtype",
                   help="Use float16 instead of bfloat16")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Apply CLI overrides to Config
    if args.hf_repo:         cfg.MOSHI_HF_REPO    = args.hf_repo
    if args.moshi_weight:    cfg.MOSHI_WEIGHT      = args.moshi_weight
    if args.mimi_weight:     cfg.MIMI_WEIGHT       = args.mimi_weight
    if args.tokenizer:       cfg.TOKENIZER         = args.tokenizer
    if args.bridge_ckpt:     cfg.BRIDGE_CKPT       = args.bridge_ckpt
    if args.bridge_config:   cfg.BRIDGE_CONFIG      = args.bridge_config
    if args.bridge_chunk:    cfg.BRIDGE_CHUNK       = args.bridge_chunk
    if args.ditto_data_root: cfg.DITTO_DATA_ROOT    = args.ditto_data_root
    if args.ditto_cfg_pkl:   cfg.DITTO_CFG_PKL      = args.ditto_cfg_pkl
    if args.ditto_emo:            cfg.DITTO_EMO             = args.ditto_emo
    if args.ditto_sampling_steps: cfg.DITTO_SAMPLING_STEPS  = args.ditto_sampling_steps
    if args.jpeg_quality:    cfg.DITTO_JPEG_QUALITY = args.jpeg_quality
    if args.dtype:           cfg.DTYPE             = args.dtype

    uvicorn.run(
        "streaming_server:app",
        host      = args.host,
        port      = args.port,
        log_level = args.log_level,
        loop      = "asyncio",
        ws_ping_interval = None,
        ws_ping_timeout  = None,
    )
