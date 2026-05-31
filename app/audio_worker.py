"""
AudioWorker — QThread that drives audio playback and model inference.

Architecture
------------
              ┌─ sounddevice audio callback (audio-hardware thread) ─┐
              │  Copies chunk to outdata (playback).                  │
              │  Puts chunk into _queue (non-blocking, drops on full)  │
              └───────────────────────────────────────────────────────┘
                                      │ queue.Queue
              ┌─ QThread.run() (inference thread) ──────────────────── ┘
              │  Drains queue → model.process_frame() → emit signal
              └─ position_updated(float, float) → Qt main thread → UI
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

import numpy as np
import librosa
from PyQt6.QtCore import QThread, pyqtSignal

from utils.audio_processing import AudioProcessor

# sounddevice is imported lazily inside run() because it probes PortAudio
# at import time; doing it lazily means a missing PortAudio installation
# is reported as a clear runtime error rather than an import crash.

if TYPE_CHECKING:
    from models.score_follower import ScoreFollower

# Number of samples per audio chunk delivered to both the speaker and the model.
# 2048 samples at 44100 Hz ≈ 46 ms per chunk (~21 fps update rate).
CHUNK_SIZE: int = 2048
SAMPLING_RATE = 22050 # half of 44100 for mono processing

# Maximum number of unprocessed chunks held in the queue.  If the model is
# slower than real-time the oldest chunks are silently dropped so the audio
# never hiccups and the UI stays in sync with actual playback time.
QUEUE_MAXSIZE: int = 64


class AudioWorker(QThread):
    """
    Plays audio from a WAV file and feeds each chunk to the score-following
    model in a dedicated thread.

    Signals
    -------
    position_updated(position_s, confidence)
        Emitted after every successful model inference step.
    playback_finished()
        Emitted when the audio stream reaches its end or stop() is called.
    error_occurred(message)
        Emitted on unrecoverable errors (audio or model).
    """

    position_updated = pyqtSignal(float, float)
    playback_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, wav_path: str, model: "ScoreFollower", parent=None) -> None:
        super().__init__(parent)
        self.wav_path = wav_path
        self.model = model
        self._running = False
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._audio_processor = AudioProcessor()

    # ------------------------------------------------------------------
    # Public API (called from the main/Qt thread)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the worker to stop.  Safe to call from any thread."""
        self._running = False
        # Unblock the inference loop in case it is waiting on an empty queue.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # QThread entry point (runs in the worker thread)
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901  (complexity is intentional here)
        self._running = True

        # ── 0. Lazy sounddevice import (PortAudio must exist at runtime) ──
        try:
            import sounddevice as sd
        except OSError as exc:
            self.error_occurred.emit(
                f"Cannot initialise audio output (PortAudio not found):\n{exc}"
            )
            self.playback_finished.emit()
            return

        # ── 1. Load audio ─────────────────────────────────────────────
        try:
            audio, sr = librosa.load(self.wav_path, sr=SAMPLING_RATE)
            audio = audio.astype(np.float32)
            #audio = self._audio_processor.time_stretch(audio, 0.5)
        except Exception as exc:
            self.error_occurred.emit(f"Cannot load audio file:\n{exc}")
            self.playback_finished.emit()
            return

        sr = int(sr)
        inf_queue = self._queue
        pos = [0]  # mutable int held in a list so the closure can write to it

        # ── 2. sounddevice callback (runs in the *audio* thread) ───────
        def _audio_callback(
            outdata: np.ndarray,
            frames: int,
            _time,
            _status,
        ) -> None:
            start = pos[0]
            end = start + frames

            if start >= len(audio):
                # Audio exhausted — fill silence and stop the stream.
                outdata[:] = 0.0
                raise sd.CallbackStop()

            chunk = audio[start : min(end, len(audio))]

            # Pad the very last chunk if shorter than blocksize.
            if len(chunk) < frames:
                padded = np.zeros(frames, dtype=np.float32)
                padded[: len(chunk)] = chunk
                outdata[:, 0] = padded
            else:
                outdata[:, 0] = chunk

            pos[0] = end

            # Hand the chunk to the inference loop (non-blocking).
            try:
                inf_queue.put_nowait(chunk.copy())
            except queue.Full:
                pass  # Inference is lagging; keep audio flowing, drop this chunk.

        def _on_stream_finished() -> None:
            # Put a sentinel so the inference loop knows to exit.
            inf_queue.put(None)

        # ── 3. Start playback stream ───────────────────────────────────
        try:
            stream = sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=_audio_callback,
                finished_callback=_on_stream_finished,
            )
        except Exception as exc:
            self.error_occurred.emit(f"Cannot open audio output:\n{exc}")
            self.playback_finished.emit()
            return

        # ── 4. Inference loop (runs in *this* QThread) ─────────────────
        with stream:
            while self._running:
                # Block until a chunk arrives or the timeout elapses.
                try:
                    chunk = inf_queue.get(timeout=0.5)
                except queue.Empty:
                    # No data yet — check whether the stream is still active.
                    if not stream.active:
                        break
                    continue

                if chunk is None:
                    # Sentinel: stream finished normally or stop() was called.
                    break

                # ── Drain stale chunks ────────────────────────────────────
                # If the model is slower than real-time (e.g. neural models
                # at ~40 ms/chunk vs 46 ms chunk interval), the queue builds
                # up and position falls behind.  Discard all but the newest
                # chunk so we always track the current playback position.
                # Fast models (OTW ~2 ms) never accumulate a backlog so this
                # branch is effectively never taken for them.
                sentinel_found = False
                while True:
                    try:
                        newer = inf_queue.get_nowait()
                        if newer is None:
                            sentinel_found = True
                            break
                        chunk = newer
                    except queue.Empty:
                        break

                if sentinel_found:
                    break

                try:
                    result = self.model.process_frame(chunk, sr)
                    position = float(result.get("position", 0.0))
                    confidence = float(result.get("confidence", 0.0))
                    self.position_updated.emit(position, confidence)
                except Exception as exc:
                    self.error_occurred.emit(f"Model inference error:\n{exc}")
                    break

        # Stream is automatically stopped when the `with` block exits.
        self.playback_finished.emit()