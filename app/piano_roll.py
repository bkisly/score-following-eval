"""
PianoRollWidget — real-time scrolling piano roll visualization.

Layout
------
┌──────────────────────────────────────────────────────────────────┐
│ Piano  │              Piano Roll Canvas                           │
│ Keys   │◄──── past ────  ┃  ──── future ────────────────────────►│
│  40px  │                 ┃  playhead (center)                     │
└────────┴─────────────────┻──────────────────────────────────────┘

Scrolling
---------
The score scrolls right-to-left so the playhead (fixed red line in the
center of the canvas) always shows the current position.

Zooming
-------
Mouse wheel: horizontal zoom (pixels per second).
Ctrl + mouse wheel: vertical zoom (note height).

Colour scheme (all overridable via class-level constants)
---------------------------------------------------------
• Note lane rows  — alternate slightly between white/black key shading
• Regular notes   — muted blue-teal gradient
• Active notes    — bright warm yellow/amber (notes intersecting playhead)
• Playhead        — red with a soft glow halo
• Piano keys      — classic black & white with C-octave labels
"""

from __future__ import annotations

from typing import List, Tuple

import pretty_midi
from PyQt6.QtCore import Qt, pyqtSlot, QRectF
from PyQt6.QtGui import (
    QColor, QFont, QLinearGradient, QPainter, QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

# ── Layout constants ──────────────────────────────────────────────────────────
PIANO_WIDTH: int = 44          # pixels for the keyboard strip on the left edge

# ── Default zoom ─────────────────────────────────────────────────────────────
DEFAULT_PPS: float = 120.0     # pixels per second (horizontal zoom)
MIN_PPS: float = 20.0
MAX_PPS: float = 800.0

DEFAULT_NOTE_H: float = 9.0    # pixels per semitone (vertical zoom)
MIN_NOTE_H: float = 3.0
MAX_NOTE_H: float = 30.0

# ── Colour palette ────────────────────────────────────────────────────────────
C_BACKGROUND   = QColor(16,  16,  26)
C_LANE_WHITE   = QColor(26,  26,  40)
C_LANE_BLACK   = QColor(14,  14,  24)
C_GRID_MINOR   = QColor(38,  38,  58)
C_GRID_MAJOR   = QColor(58,  58,  88)
C_TIME_LABEL   = QColor(120, 120, 160)
C_NOTE         = QColor(60,  130, 210)
C_NOTE_DARK    = QColor(40,  100, 175)      # gradient stop
C_NOTE_ACTIVE  = QColor(255, 200,  50)
C_NOTE_ACT_D   = QColor(230, 140,  20)      # gradient stop
C_PLAYHEAD     = QColor(255,  70,  70)
C_PLAYHEAD_GLOW= QColor(255,  70,  70,  30)
C_KEY_WHITE    = QColor(215, 215, 225)
C_KEY_BLACK    = QColor( 28,  28,  40)
C_KEY_BORDER   = QColor( 80,  80, 110)
C_KEY_LABEL    = QColor( 60,  60,  90)
C_SEPARATOR    = QColor( 50,  50,  80)
C_ACTIVE_LANE  = QColor(255, 200,  50,  18)  # tinted row for active notes

# Pitch classes that correspond to black piano keys (C# D# F# G# A#)
_BLACK_KEYS: frozenset[int] = frozenset({1, 3, 6, 8, 10})

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NoteData = Tuple[float, float, int]   # (start_s, end_s, pitch)


class PianoRollWidget(QWidget):
    """Scrolling piano-roll display driven by position_updated() slot."""

    def __init__(self, midi_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._notes: List[NoteData] = []
        self._duration: float = 0.0
        self._current_pos: float = 0.0
        self._confidence: float = 0.0

        self._pps: float = DEFAULT_PPS          # pixels / second (H zoom)
        self._note_h: float = DEFAULT_NOTE_H    # pixels / semitone (V zoom)
        self._min_pitch: int = 21               # A0
        self._max_pitch: int = 108              # C8

        self.setMinimumSize(900, 400)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self._load_midi(midi_path)

    # ── Public interface ──────────────────────────────────────────────────────

    def reset_position(self) -> None:
        self._current_pos = 0.0
        self._confidence = 0.0
        self.update()

    @pyqtSlot(float, float)
    def update_position(self, position: float, confidence: float) -> None:
        self._current_pos = position
        self._confidence = confidence
        self.update()

    # ── Mouse-wheel zoom ──────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else (1.0 / 1.12)
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            self._note_h = float(
                max(MIN_NOTE_H, min(MAX_NOTE_H, self._note_h * factor))
            )
        else:
            self._pps = float(
                max(MIN_PPS, min(MAX_PPS, self._pps * factor))
            )
        self.update()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_midi(self, path: str) -> None:
        pm = pretty_midi.PrettyMIDI(path)
        self._notes = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                self._notes.append((note.start, note.end, note.pitch))

        self._duration = pm.get_end_time()

        if self._notes:
            pitches = [n[2] for n in self._notes]
            self._min_pitch = max(0,   min(pitches) - 2)
            self._max_pitch = min(127, max(pitches) + 2)

        # Sort by start time so we could binary-search later if needed.
        self._notes.sort(key=lambda n: n[0])

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        W = self.width()
        H = self.height()

        canvas_x = PIANO_WIDTH               # x-origin of the roll canvas
        canvas_w = W - canvas_x              # canvas width in pixels

        # The playhead sits at the horizontal mid-point of the canvas area.
        ph_x = canvas_x + canvas_w // 2

        # Convert helpers
        def t2x(t: float) -> float:
            return ph_x + (t - self._current_pos) * self._pps

        def p2y(pitch: int) -> float:
            """Top-left y of the note row for `pitch`."""
            rows_from_bottom = pitch - self._min_pitch
            total_rows = self._max_pitch - self._min_pitch + 1
            # Pitch increases upward; (0,0) is top-left in Qt.
            roll_h = total_rows * self._note_h
            base_y = max(0.0, (H - roll_h) / 2)   # centre vertically
            return base_y + (total_rows - 1 - rows_from_bottom) * self._note_h

        total_rows = self._max_pitch - self._min_pitch + 1
        roll_h = total_rows * self._note_h
        base_y = max(0.0, (H - roll_h) / 2)

        # Visible time range
        t_left  = self._current_pos - (ph_x - canvas_x) / self._pps
        t_right = self._current_pos + (W - ph_x) / self._pps

        # ── 1. Background ─────────────────────────────────────────────
        painter.fillRect(canvas_x, 0, canvas_w, H, C_BACKGROUND)

        # ── 2. Pitch lane rows ────────────────────────────────────────
        for pitch in range(self._min_pitch, self._max_pitch + 1):
            is_black = (pitch % 12) in _BLACK_KEYS
            y = p2y(pitch)
            iy, ih = int(y), max(1, int(self._note_h))
            color = C_LANE_BLACK if is_black else C_LANE_WHITE
            painter.fillRect(canvas_x, iy, canvas_w, ih, color)

        # ── 3. Time grid & labels ─────────────────────────────────────
        t_step = _choose_grid_step(self._pps)
        t_start = int(t_left / t_step) * t_step
        t = t_start
        while t <= t_right + t_step:
            x = int(t2x(t))
            if x >= canvas_x:
                is_major = (round(t) % max(1, t_step * 5)) == 0
                painter.setPen(QPen(C_GRID_MAJOR if is_major else C_GRID_MINOR, 1))
                painter.drawLine(x, 0, x, H)

                if is_major and canvas_w > 60:
                    m, s = divmod(int(round(t)), 60)
                    label = f"{m}:{s:02d}"
                    painter.setPen(QPen(C_TIME_LABEL, 1))
                    painter.setFont(QFont("monospace", 8))
                    painter.drawText(x + 3, int(base_y) + 12, label)
            t += t_step

        # ── 4. Active-lane highlight ──────────────────────────────────
        #   For any note active right now, tint the full lane column.
        pos = self._current_pos
        active_pitches = {
            n[2] for n in self._notes if n[0] <= pos <= n[1]
        }
        for pitch in active_pitches:
            if self._min_pitch <= pitch <= self._max_pitch:
                y = p2y(pitch)
                painter.fillRect(
                    canvas_x, int(y), canvas_w, max(1, int(self._note_h)),
                    C_ACTIVE_LANE,
                )

        # ── 5. Notes ──────────────────────────────────────────────────
        nh = max(1.0, self._note_h - 1.0)   # slight gap between rows

        for start_s, end_s, pitch in self._notes:
            if pitch < self._min_pitch or pitch > self._max_pitch:
                continue
            if end_s < t_left or start_s > t_right:
                continue

            x1 = max(t2x(start_s), float(canvas_x))
            x2 = t2x(end_s)
            nw = max(2.0, x2 - x1)
            y  = p2y(pitch)

            is_active = start_s <= pos <= end_s

            if is_active:
                grad = QLinearGradient(x1, y, x1, y + nh)
                grad.setColorAt(0.0, C_NOTE_ACTIVE)
                grad.setColorAt(1.0, C_NOTE_ACT_D)
                painter.fillRect(QRectF(x1, y + 1, nw, nh), grad)
                # Bright border for active notes
                painter.setPen(QPen(C_NOTE_ACTIVE.lighter(140), 1))
                painter.drawRect(QRectF(x1, y + 1, nw, nh))
            else:
                grad = QLinearGradient(x1, y, x1, y + nh)
                grad.setColorAt(0.0, C_NOTE)
                grad.setColorAt(1.0, C_NOTE_DARK)
                painter.fillRect(QRectF(x1, y + 1, nw, nh), grad)
                # Subtle border
                if nh > 4 and nw > 4:
                    painter.setPen(QPen(C_NOTE.lighter(120), 1))
                    painter.drawRect(QRectF(x1, y + 1, nw - 1, nh - 1))

        # ── 6. Playhead glow ─────────────────────────────────────────
        # Soft radial halo — painted as a wide semi-transparent pen.
        for width, alpha in [(18, 12), (10, 22), (4, 50)]:
            glow = QColor(C_PLAYHEAD)
            glow.setAlpha(alpha)
            painter.setPen(QPen(glow, width))
            painter.drawLine(ph_x, 0, ph_x, H)

        # ── 7. Playhead line ─────────────────────────────────────────
        painter.setPen(QPen(C_PLAYHEAD, 2))
        painter.drawLine(ph_x, 0, ph_x, H)

        # ── 8. Piano keyboard strip ───────────────────────────────────
        _draw_keyboard(painter, PIANO_WIDTH, H, self._min_pitch, self._max_pitch,
                       p2y, self._note_h)

        # ── 9. Separator between keyboard and roll ────────────────────
        painter.setPen(QPen(C_SEPARATOR, 1))
        painter.drawLine(canvas_x, 0, canvas_x, H)


# ── Piano keyboard helper ─────────────────────────────────────────────────────

def _draw_keyboard(
    painter: QPainter,
    width: int,
    height: int,
    min_pitch: int,
    max_pitch: int,
    p2y,
    note_h: float,
) -> None:
    """Draw a piano keyboard strip on the left side of the widget."""
    painter.fillRect(0, 0, width, height, C_KEY_BLACK)

    # Draw white keys first (they are wider), then overlay black keys.
    for pass_n in range(2):
        for pitch in range(min_pitch, max_pitch + 1):
            is_black = (pitch % 12) in _BLACK_KEYS
            if pass_n == 0 and is_black:
                continue
            if pass_n == 1 and not is_black:
                continue

            y  = p2y(pitch)
            iy = int(y)
            ih = max(1, int(note_h))

            if is_black:
                bw = int(width * 0.65)
                painter.fillRect(0, iy, bw, ih, C_KEY_BLACK)
                painter.setPen(QPen(QColor(50, 50, 70), 1))
                painter.drawRect(0, iy, bw - 1, ih - 1)
            else:
                painter.fillRect(0, iy, width - 1, ih, C_KEY_WHITE)
                painter.setPen(QPen(C_KEY_BORDER, 1))
                painter.drawRect(0, iy, width - 2, ih - 1)

                # Label: C notes only (e.g. "C4")
                if pitch % 12 == 0 and ih >= 7:
                    octave = pitch // 12 - 1
                    painter.setPen(QPen(C_KEY_LABEL, 1))
                    painter.setFont(QFont("monospace", max(5, min(8, int(note_h) - 2))))
                    painter.drawText(2, iy + ih - 2, f"C{octave}")


# ── Grid step selection ───────────────────────────────────────────────────────

def _choose_grid_step(pps: float) -> int:
    """Return a sensible time-grid spacing (in whole seconds) for the current zoom."""
    thresholds = [
        (500, 1),
        (200, 2),
        (100, 5),
        (50,  10),
        (20,  30),
        (0,   60),
    ]
    for min_pps, step in thresholds:
        if pps >= min_pps:
            return step
    return 60
