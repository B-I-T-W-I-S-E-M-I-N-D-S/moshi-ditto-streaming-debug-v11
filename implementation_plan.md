# Latency Instrumentation & Bottleneck Detection Plan

## Goal
Add full-pipeline latency instrumentation to identify why streaming FPS drops from 25 → 6–9 FPS.
All changes preserve existing functionality and are backward-compatible.

---

## Root Cause Hypothesis (Before Instrumentation)

Based on code analysis, the most likely bottlenecks are:

| Rank | Component | Likely Cause |
|------|-----------|--------------|
| 🔴 1 | **event loop blocking** | `bridge.step()` + `ditto.push_features()` are CPU/GPU-bound; `asyncio.to_thread()` helps but Ditto's internal thread queue adds latency |
| 🔴 2 | **Ditto sliding-window inference** | `sampling_timesteps=50` is very high — each diffusion step is slow; `overlap_v2=10` adds extra frames per window |
| 🟡 3 | **Token queue backpressure** | If Moshi produces tokens faster than Ditto consumes them, `frame_queue` stays empty → visible frame gaps |
| 🟡 4 | **JPEG encode on hot path** | PIL `Image.save()` called in Ditto writer thread — can take 3–8ms per frame on CPU |
| 🟢 5 | **WebSocket write lock** | `ws_lock` serializes all sends but adds minimal overhead |

> **Key offline vs streaming difference**: offline mode batches all tokens at once — Ditto gets a full feature sequence and can overlap windows efficiently. In streaming, Ditto gets features 1 at a time via `push_features()`, causing frequent pipeline flushes and cold-start diffusion overhead per chunk.

---

## Proposed Changes

### 1. New File: `pipeline/latency_tracker.py`

A centralized, thread-safe latency tracker with:
- Per-module stopwatch (Moshi / Bridge / Ditto / Sender)
- Rolling window FPS calculation (last 30 frames)
- Moving average latency tracking
- Threshold-based warning system
- Structured log output with `[TIMESTAMP]` prefix
- Queue depth monitoring
- JSON summary export at session end

### 2. Modified: `pipeline/streaming_moshi.py`

Add instrumentation to `handle_connection()`:
- `t_audio_buffer`: time spent waiting for enough PCM bytes
- `t_mimi_encode`: Moshi Mimi encoder inference time
- `t_lm_step`: LM generation step time
- `t_token_emit`: time to put token on queue
- `t_audio_decode`: Mimi decode back to PCM time
- `t_total_frame`: end-to-end frame time in Moshi
- Log comparison vs 15ms threshold

### 3. Modified: `streaming_server.py`

Add instrumentation to `bridge_task()` and session context:
- `t_queue_wait`: time blocked on `await token_queue.get()`
- `t_bridge_transform`: time for `bridge.step()` in thread
- `t_feature_push`: time to call `ditto.push_features()`
- Queue depth logging (token_queue, frame_queue)
- Backpressure detection (queue > 80% full)

### 4. Modified: `pipeline/ditto_stream_adapter.py`

Add instrumentation inside monkey-patched `_patched_writer_worker`:
- `t_writer_queue_wait`: time blocked on `writer_queue.get()`
- `t_jpeg_encode`: JPEG encoding time
- `t_frame_emit`: time to put frame in output queue
- Per-frame FPS rolling calculator
- Frame count tracking + running average

### 5. Modified: `streaming_server.py` (frame sender)

Add instrumentation in `frame_sender_task()`:
- `t_frame_queue_wait`: time waiting for frame from Ditto
- `t_ws_send`: WebSocket send time
- E2E pipeline latency: from Moshi token emit → frame sent

---

## Performance Thresholds (for warning logic)

| Module | Target | Warning |
|--------|--------|---------|
| Moshi encode+lm_step | ≤ 15 ms | > 25 ms |
| Bridge transform | ≤ 10 ms | > 20 ms |
| Ditto frame gen | ≤ 40 ms | > 60 ms |
| JPEG encode | ≤ 5 ms | > 10 ms |
| WebSocket send | ≤ 3 ms | > 8 ms |
| Pipeline FPS | ≥ 20 | < 15 |
| Token queue depth | ≤ 50 | > 200 |
| Frame queue depth | ≤ 20 | > 100 |

---

## Log Format

```
[HH:MM:SS.mmm] [Moshi] chunk_processed=12.3ms  mimi_encode=4.1ms  lm_step=8.2ms  token_emit=0.1ms
[HH:MM:SS.mmm] [Bridge] queue_wait=2.1ms  transform=6.8ms  feature_push=0.3ms
[HH:MM:SS.mmm] [Ditto] writer_wait=25.3ms  jpeg_encode=3.2ms  frame_emit=0.1ms
[HH:MM:SS.mmm] [Sender] frame_wait=1.2ms  ws_send=1.8ms
[HH:MM:SS.mmm] [PIPELINE] fps=18.3  avg_latency=54.7ms  token_q=12  frame_q=3
[HH:MM:SS.mmm] ⚠️ [BOTTLENECK] Ditto writer_wait=25.3ms exceeds threshold (20ms)
[HH:MM:SS.mmm] 📊 [SUMMARY] frames=150  avg_fps=19.2  avg_latency=52ms  peak_latency=180ms
```

---

## Verification Plan

### Automated
- Start server, connect browser, run 30 seconds of audio
- Read server logs — every frame should have a `[PIPELINE]` line
- Confirm FPS values appear in logs
- Confirm `[BOTTLENECK]` warnings appear if threshold exceeded

### Manual Analysis
- Compare `[Moshi]` latency vs 15ms target
- Look for `[Ditto] writer_wait` > 20ms (backpressure from diffusion)
- Check `token_q` depth — if always > 100, Bridge is too slow
- Check `frame_q` depth — if always 0, Ditto is producing but sender is slow; if always 200, Ditto is the bottleneck
