"""
pipeline/streaming_moshi.py
============================
Adapts the official ``moshi/server.py`` ServerState for real-time token
streaming inside our FastAPI server.

Key differences from the original server.py
--------------------------------------------
* Works with ``asyncio`` queues instead of WebSocket writes, so our server
  can interleave audio and video frame messages independently.
* After every ``lm_gen.step()`` call the raw acoustic tokens
  ``(1, dep_q)`` are pushed onto ``token_queue`` so the Bridge module
  can process them concurrently.
* Exposes ``handle_connection(ws)`` which is an async generator:
    - yields  (kind, payload) tuples:
        kind == "audio"  → bytes (Opus packet)
        kind == "text"   → str  (partial word)
        kind == "tokens" → torch.Tensor (1, 8) int64  [bridge input]

Usage inside FastAPI
--------------------
    state = StreamingMoshiState(hf_repo=..., device="cuda")
    state.warmup()

    @app.websocket("/ws/{session_id}")
    async def endpoint(ws: WebSocket):
        await ws.accept()
        async for kind, payload in state.handle_connection(ws):
            if kind == "audio":
                await ws.send_bytes(b"\\x01" + payload)
            elif kind == "text":
                await ws.send_bytes(b"\\x03" + payload.encode())
            # "tokens" are not sent to the browser — forwarded internally
"""

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from typing import AsyncIterator, Optional, Tuple

# Latency instrumentation (optional – imported lazily to avoid circular deps)
try:
    from pipeline.latency_tracker import PipelineTracker, Stopwatch
except ImportError:
    try:
        from latency_tracker import PipelineTracker, Stopwatch
    except ImportError:
        PipelineTracker = None  # type: ignore
        Stopwatch = None        # type: ignore

import numpy as np
import torch
import sphn

# ---------------------------------------------------------------------------
# Ensure moshi-inference is importable
# ---------------------------------------------------------------------------
_MOSHI_PKG = os.path.join(os.path.dirname(__file__), "..", "moshi-inference")
if _MOSHI_PKG not in sys.path:
    sys.path.insert(0, _MOSHI_PKG)

from moshi.models import loaders, MimiModel, LMModel, LMGen
from moshi.run_inference import get_condition_tensors, seed_all

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _WebMAudioDecoder
# ---------------------------------------------------------------------------

# EBML magic bytes that identify a WebM/Matroska container
_EBML_MAGIC = bytes([0x1A, 0x45, 0xDF, 0xA3])
# Ogg capture pattern
_OGG_MAGIC  = b"OggS"


class _WebMAudioDecoder:
    """
    Accepts a continuous stream of raw WebM *or* Ogg Opus bytes from the
    browser and returns float32 mono PCM samples via a **persistent ffmpeg
    process** (one per session).

    Why a persistent process?
    -------------------------
    Spawning ffmpeg per-chunk causes it to re-probe the format on every
    call.  With a streaming WebM source it can't seek backward, so it
    misidentifies the codec (e.g. as MP3) and fails.  A single long-lived
    process receives a continuous byte stream on stdin and emits raw PCM
    on stdout — exactly how a real-time pipeline should work.

    For Ogg/Opus (Firefox) the fast ``sphn.OpusStreamReader`` path is
    used instead to avoid subprocess overhead entirely.

    Parameters
    ----------
    sample_rate : target PCM sample rate (must match Mimi's sample_rate)
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

        self._buf: bytes               = b""
        self._container_detected: bool = False
        self._use_ogg: bool            = False

        # sphn path (Ogg only)
        self._ogg_reader = None

        # ffmpeg persistent process (WebM path)
        self._proc: Optional[subprocess.Popen] = None
        self._pcm_buf: np.ndarray              = np.zeros(0, dtype=np.float32)
        self._reader_thread: Optional[threading.Thread] = None
        self._pcm_lock   = threading.Lock()
        self._closed     = False

    # ------------------------------------------------------------------
    # Public interface  (same as sphn.OpusStreamReader)
    # ------------------------------------------------------------------

    def append_bytes(self, data: bytes) -> np.ndarray:
        """
        Feed raw bytes and return all PCM samples decoded so far.
        Returns an empty array while the buffer is filling up.
        """
        self._buf += data

        # Detect container on the first call once we have >= 4 bytes
        if not self._container_detected and len(self._buf) >= 4:
            self._detect_container()
            if not self._container_detected:
                return np.zeros(0, dtype=np.float32)

        if self._use_ogg:
            pcm = self._ogg_reader.append_bytes(self._buf)
            self._buf = b""
            return pcm

        # WebM path — ensure ffmpeg is running, then feed bytes
        if self._proc is None:
            return np.zeros(0, dtype=np.float32)

        try:
            self._proc.stdin.write(self._buf)
            self._proc.stdin.flush()
            self._buf = b""
        except BrokenPipeError:
            logger.warning("[_WebMAudioDecoder] ffmpeg stdin pipe broken")
            return np.zeros(0, dtype=np.float32)

        # Collect whatever PCM the reader thread has accumulated
        with self._pcm_lock:
            out, self._pcm_buf = self._pcm_buf, np.zeros(0, dtype=np.float32)
        return out

    def close(self):
        """Shut down the ffmpeg process gracefully."""
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_container(self):
        header = self._buf[:4]
        if header == _OGG_MAGIC:
            logger.debug(
                "[_WebMAudioDecoder] Ogg container — using sphn fast path"
            )
            self._use_ogg = True
            self._ogg_reader = sphn.OpusStreamReader(self.sample_rate)
            self._container_detected = True

        elif header == _EBML_MAGIC:
            logger.debug(
                "[_WebMAudioDecoder] WebM/EBML container — starting persistent ffmpeg"
            )
            self._start_ffmpeg()
            self._container_detected = True

        else:
            # Unknown magic — wait until we have 32 bytes before giving up
            # and falling back to ffmpeg anyway
            if len(self._buf) >= 32:
                logger.debug(
                    f"[_WebMAudioDecoder] Unknown magic {list(header)} — "
                    "falling back to persistent ffmpeg"
                )
                self._start_ffmpeg()
                self._container_detected = True

    def _start_ffmpeg(self):
        """Launch a persistent ffmpeg process that reads from stdin."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            # Streaming-friendly flags: disable seeking/buffering
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", "pipe:0",
            # Output: raw 32-bit float PCM, mono, target sample rate
            "-f", "f32le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.debug("[_WebMAudioDecoder] Persistent ffmpeg process started.")
        except FileNotFoundError:
            logger.error(
                "[_WebMAudioDecoder] ffmpeg not found. "
                "Install with: apt-get install -y ffmpeg"
            )
            self._proc = None
            return

        # Background thread: continuously drain stdout into _pcm_buf
        self._reader_thread = threading.Thread(
            target=self._stdout_reader, daemon=True
        )
        self._reader_thread.start()

        # Background thread: log ffmpeg stderr without blocking
        threading.Thread(
            target=self._stderr_reader, daemon=True
        ).start()

    def _stdout_reader(self):
        """Continuously read raw PCM from ffmpeg stdout into _pcm_buf."""
        CHUNK = 4096  # bytes = 1024 float32 samples
        while not self._closed:
            try:
                raw = self._proc.stdout.read(CHUNK)
            except Exception:
                break
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.float32).copy()
            with self._pcm_lock:
                self._pcm_buf = np.concatenate((self._pcm_buf, samples))

    def _stderr_reader(self):
        """Log ffmpeg stderr lines at DEBUG level so they are visible but quiet."""
        for line in self._proc.stderr:
            msg = line.decode(errors="replace").rstrip()
            if msg:
                logger.debug(f"[_WebMAudioDecoder] ffmpeg: {msg}")


# ---------------------------------------------------------------------------
# StreamingMoshiState
# ---------------------------------------------------------------------------

class StreamingMoshiState:
    """
    Loads Moshi + Mimi in streaming-forever mode and handles one WebSocket
    session at a time (protected by an asyncio.Lock).

    Parameters
    ----------
    hf_repo       : HuggingFace repo slug (default: kyutai/moshiko-pytorch-bf16)
    moshi_weight  : optional local .safetensors path for Moshi LM
    mimi_weight   : optional local .safetensors path for Mimi
    tokenizer     : optional local .model path for the text tokenizer
    device        : "cuda" or "cpu"
    dtype         : bfloat16 (default) or float16
    cfg_coef      : classifier-free guidance coefficient
    seed          : RNG seed for reproducibility
    """

    def __init__(
        self,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        moshi_weight: Optional[str] = None,
        mimi_weight: Optional[str]  = None,
        tokenizer: Optional[str]    = None,
        device: str                 = "cuda",
        dtype: torch.dtype          = torch.bfloat16,
        cfg_coef: float             = 1.0,
        seed: int                   = 42424242,
    ):
        seed_all(seed)
        self.device = device
        self.dtype  = dtype

        logger.info("[StreamingMoshi] Loading checkpoint info ...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo, moshi_weight, mimi_weight, tokenizer
        )

        logger.info("[StreamingMoshi] Loading Mimi ...")
        self.mimi: MimiModel = checkpoint_info.get_mimi(device=device)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        logger.info("[StreamingMoshi] Loading Moshi LM ...")
        lm: LMModel = checkpoint_info.get_moshi(device=device, dtype=dtype)
        self.dep_q = lm.dep_q

        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        condition_tensors = get_condition_tensors(
            checkpoint_info.model_type, lm, batch_size=1, cfg_coef=cfg_coef
        )
        self.lm_gen: LMGen = LMGen(
            lm,
            cfg_coef=cfg_coef,
            condition_tensors=condition_tensors,
            **checkpoint_info.lm_gen_config,
        )
        self.model_type = checkpoint_info.model_type

        # Enable streaming state (kept alive across calls)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # One conversation at a time
        self._lock = asyncio.Lock()

        logger.info("[StreamingMoshi] Ready.")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self):
        """Run 4 silent frames to warm up TorchScript / CUDA graphs."""
        logger.info("[StreamingMoshi] Warming up ...")
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if self.device == "cuda":
            torch.cuda.synchronize()
        logger.info("[StreamingMoshi] Warmup complete.")

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    async def handle_connection(
        self,
        receive_fn,                       # async callable() → bytes | None
        token_queue: asyncio.Queue,       # receives torch.Tensor (1, dep_q) int64 per step
        tracker: Optional["PipelineTracker"] = None,  # latency tracker (optional)
    ) -> AsyncIterator[Tuple[str, object]]:
        """
        Async generator that drives a single conversation session.

        Yields
        ------
        ("handshake", bytes) — one-time handshake b"\\x00" at session start
        ("audio",     bytes) — Opus audio packet from Moshi  (send as 0x01)
        ("text",      str)   — Text token piece from Moshi   (send as 0x03)

        The acoustic tokens are pushed directly onto ``token_queue``;
        they are NOT yielded so that the caller can forward them to the
        Bridge concurrently without blocking this generator.

        Parameters
        ----------
        receive_fn    : async callable that returns raw WebSocket message bytes.
                        Should return None when the session ends.
        token_queue   : asyncio.Queue where (1, dep_q) token tensors are pushed.
                        A sentinel ``None`` is pushed when the session ends.
        """
        async with self._lock:
            # Reset streaming state for new session
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            # Persistent-process decoder handles WebM (Chrome) and Ogg (Firefox)
            audio_decoder = _WebMAudioDecoder(self.mimi.sample_rate)
            # NOTE: We do NOT use sphn.OpusStreamWriter for the outbound path.
            # Raw float32 LE PCM is sent directly so the browser AudioWorklet
            # can play it without dealing with container format wrapping issues.

            # Send handshake byte
            yield ("handshake", b"\x00")

            all_pcm_data = None
            skip_frames  = 1  # mirrors server.py behaviour

            try:
                while True:
                    raw = await receive_fn()
                    if raw is None:
                        break

                    # Expect binary messages prefixed with kind byte (0x01 = audio)
                    if len(raw) < 2:
                        continue
                    kind = raw[0]
                    if kind != 1:
                        continue
                    payload = bytes(raw[1:])

                    # Decode container (WebM or Ogg) → float32 PCM
                    pcm = audio_decoder.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue

                    all_pcm_data = (
                        pcm
                        if all_pcm_data is None
                        else np.concatenate((all_pcm_data, pcm))
                    )

                    # Process full frames
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        # ── Per-frame timing accumulators ─────────────────
                        t_frame_start = time.perf_counter()
                        mimi_encode_ms  = 0.0
                        lm_step_ms      = 0.0
                        token_emit_ms   = 0.0
                        audio_decode_ms = 0.0

                        chunk_np     = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size :]

                        chunk_t = (
                            torch.from_numpy(chunk_np)
                            .to(device=self.device)[None, None]
                        )

                        # ── Mimi encode ───────────────────────────────────
                        _t = time.perf_counter()
                        with torch.no_grad():
                            codes = self.mimi.encode(chunk_t)
                        mimi_encode_ms = (time.perf_counter() - _t) * 1_000

                        if skip_frames:
                            self.mimi.reset_streaming()
                            skip_frames -= 1
                            continue

                        for c in range(codes.shape[-1]):
                            # ── LM step ───────────────────────────────────
                            _t = time.perf_counter()
                            with torch.no_grad():
                                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                            lm_step_ms += (time.perf_counter() - _t) * 1_000

                            if tokens is None:
                                continue

                            # ── Capture acoustic tokens ───────────────────
                            if self.dep_q > 0:
                                # tokens: (1, dep_q+1, 1)
                                acoustic = tokens[:, 1:, 0].cpu()  # (1, dep_q)
                                _t = time.perf_counter()
                                try:
                                    token_queue.put_nowait(acoustic)
                                except asyncio.QueueFull:
                                    logger.warning(
                                        "[StreamingMoshi] token_queue full — dropping token"
                                        f"  (depth={token_queue.qsize()})"
                                    )
                                token_emit_ms += (time.perf_counter() - _t) * 1_000

                            # ── Decode + send audio ───────────────────────
                            if self.dep_q > 0:
                                _t = time.perf_counter()
                                with torch.no_grad():
                                    out_pcm = self.mimi.decode(tokens[:, 1:])
                                audio_decode_ms += (time.perf_counter() - _t) * 1_000

                                # Shape: (1, 1, T) — squeeze to (T,) float32
                                pcm_f32 = out_pcm[0, 0].float().cpu().numpy()
                                # Yield raw float32 LE PCM bytes.
                                # The browser AudioWorklet interprets these directly
                                # via Float32Array — no Opus/OGG container needed.
                                yield ("audio", pcm_f32.tobytes())

                                # ── Text token ───────────────────────────
                                text_tok = tokens[0, 0, 0].item()
                                if text_tok not in (0, 3):
                                    piece = self.text_tokenizer.id_to_piece(text_tok)
                                    piece = piece.replace("▁", " ")
                                    yield ("text", piece)

                        # ── Record to tracker ─────────────────────────────
                        total_frame_ms = (time.perf_counter() - t_frame_start) * 1_000
                        if tracker is not None:
                            tracker.record_moshi(
                                mimi_encode_ms  = mimi_encode_ms,
                                lm_step_ms      = lm_step_ms,
                                token_emit_ms   = token_emit_ms,
                                audio_decode_ms = audio_decode_ms,
                            )
                        else:
                            logger.debug(
                                f"[StreamingMoshi] frame={total_frame_ms:.1f}ms"
                                f"  mimi_encode={mimi_encode_ms:.1f}ms"
                                f"  lm_step={lm_step_ms:.1f}ms"
                                f"  audio_decode={audio_decode_ms:.1f}ms"
                            )

            except Exception as exc:
                logger.exception(f"[StreamingMoshi] Error in handle_connection: {exc}")
            finally:
                logger.info(
                    "[StreamingMoshi] Connection closed — resetting streaming state."
                )
                # Cleanly shut down the persistent ffmpeg process
                audio_decoder.close()
                # Signal bridge that this session is done
                await token_queue.put(None)