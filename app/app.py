"""
MainWindow — top-level application window.

Responsibilities
----------------
• Hosts the PianoRollWidget.
• Owns and manages the AudioWorker lifecycle.
• Handles keyboard input (SPACE = play/stop, ESC = stop).
• Displays real-time position / confidence / model info in a header bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QKeyEvent, QCloseEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.audio_worker import AudioWorker
from app.piano_roll import PianoRollWidget

if TYPE_CHECKING:
    from models.score_follower import ScoreFollower


# ── Stylesheet constants ──────────────────────────────────────────────────────
_HEADER_BG   = "background-color: #10101a; border-bottom: 1px solid #28283e;"
_STATUSBAR_SS = "background-color: #0c0c18; color: #505070; font-size: 11px;"
_LABEL_MONO  = "color: #d8d8f0; font-family: monospace; font-size: 13px;"
_LABEL_DIM   = "color: #8080a8; font-size: 12px;"
_LABEL_MODEL = "color: #7090d8; font-size: 12px; font-style: italic;"
_LABEL_HINT  = "color: #505068; font-size: 11px;"
_STATE_IDLE  = "color: #708870; font-size: 11px;"
_STATE_PLAY  = "color: #80c880; font-size: 11px;"
_STATE_ERR   = "color: #c87070; font-size: 11px;"


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(
        self,
        wav_path: str,
        midi_path: str,
        model: "ScoreFollower",
        model_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.wav_path   = wav_path
        self.midi_path  = midi_path
        self.model      = model
        self.model_name = model_name

        self._worker: AudioWorker | None = None
        self._playing: bool = False

        self._build_ui()
        self._apply_dark_frame()

        self.setWindowTitle(f"Score Follower  —  {model_name}")
        self.resize(1280, 680)
        self.setFocus()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar ─────────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(_HEADER_BG)
        header.setFixedHeight(38)
        hlayout = QHBoxLayout(header)
        hlayout.setContentsMargins(14, 0, 14, 0)
        hlayout.setSpacing(0)

        self._pos_label   = _label("Position: 0:00.00", _LABEL_MONO)
        self._conf_label  = _label("Confidence: —", _LABEL_DIM)
        self._state_label = _label("Ready", _STATE_IDLE)
        self._hint_label  = _label(
            "SPACE  play / stop   ·   ESC  stop   ·   scroll  H-zoom   ·   Ctrl+scroll  V-zoom",
            _LABEL_HINT,
        )
        self._model_label = _label(f"▶  {self.model_name}", _LABEL_MODEL)

        for w, stretch in [
            (self._pos_label,   0),
            (_spacer(20),       0),
            (self._conf_label,  0),
            (_spacer(16),       0),
            (self._state_label, 0),
            (None,              1),  # flexible gap
            (self._hint_label,  0),
            (None,              1),
            (self._model_label, 0),
        ]:
            if w is None:
                hlayout.addStretch(stretch)
            elif stretch:
                hlayout.addWidget(w, stretch)
            else:
                hlayout.addWidget(w)

        layout.addWidget(header)

        # Piano roll ─────────────────────────────────────────────────────────
        self.piano_roll = PianoRollWidget(self.midi_path)
        layout.addWidget(self.piano_roll, stretch=1)

        # Status bar ─────────────────────────────────────────────────────────
        sb = QStatusBar()
        sb.setStyleSheet(_STATUSBAR_SS)
        sb.showMessage(
            f"Loaded  ·  {self.wav_path}  →  {self.midi_path}"
            f"  ·  Press SPACE to start"
        )
        self.setStatusBar(sb)

    def _apply_dark_frame(self) -> None:
        self.setStyleSheet("QMainWindow { background: #0c0c18; }")

    # ── Keyboard handling ─────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return

        key = event.key()
        if key == Qt.Key.Key_Space:
            if self._playing:
                self._stop_playback()
            else:
                self._start_playback()
        elif key == Qt.Key.Key_Escape:
            self._stop_playback()
        else:
            super().keyPressEvent(event)

    # ── Playback control ──────────────────────────────────────────────────────

    def _start_playback(self) -> None:
        if self._playing:
            return

        self._playing = True
        self.model.reset()
        self.piano_roll.reset_position()

        self._state_label.setText("● Playing")
        self._state_label.setStyleSheet(_STATE_PLAY)
        self.statusBar().showMessage("Playing…  (SPACE or ESC to stop)")

        self._worker = AudioWorker(self.wav_path, self.model)
        self._worker.position_updated.connect(self._on_position_updated)
        self._worker.playback_finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _stop_playback(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)

        self._playing = False
        self._worker  = None

        self._state_label.setText("■ Stopped")
        self._state_label.setStyleSheet(_STATE_IDLE)
        self.statusBar().showMessage("Stopped  —  press SPACE to replay")

    # ── Slots (called from AudioWorker signals, always on the Qt main thread) ─

    @pyqtSlot(float, float)
    def _on_position_updated(self, position: float, confidence: float) -> None:
        # Forward to the piano roll first (most time-sensitive).
        self.piano_roll.update_position(position, confidence)

        # Update header labels.
        mins, secs = divmod(position, 60)
        self._pos_label.setText(f"Position: {int(mins)}:{secs:05.2f}")

        if confidence > 0.0:
            self._conf_label.setText(f"Confidence: {confidence:.0%}")
        else:
            self._conf_label.setText("Confidence: —")

    @pyqtSlot()
    def _on_finished(self) -> None:
        self._playing = False
        self._worker  = None
        self._state_label.setText("✓ Finished")
        self._state_label.setStyleSheet(_STATE_IDLE)
        self.statusBar().showMessage(
            "Playback finished  —  press SPACE to replay from the beginning"
        )

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._playing = False
        self._worker  = None
        self._state_label.setText("✗ Error")
        self._state_label.setStyleSheet(_STATE_ERR)
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Score Follower — Error", message)

    # ── Window close ─────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_playback()
        event.accept()


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _label(text: str, style: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(style)
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


def _spacer(w: int) -> QWidget:
    sp = QWidget()
    sp.setFixedWidth(w)
    return sp
