"""Reusable Tkinter widgets."""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageCanvas(tk.Canvas):
    """Displays a cv2 / numpy image, scaled-to-fit on resize.

    Optionally supports drag-to-select an ROI rectangle (image coordinates).
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#222", highlightthickness=0, **kwargs)
        self._image: Optional[np.ndarray] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        # ROI state
        self._roi_enabled: bool = False
        self._roi: Optional[tuple[int, int, int, int]] = None  # in image coords
        self._drag_start_canvas: Optional[tuple[int, int]] = None

        # Render geometry (set by _render — used to map canvas <-> image coords)
        self._render_scale: float = 1.0
        self._render_offset: tuple[int, int] = (0, 0)

        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def show(self, img: Optional[np.ndarray]) -> None:
        self._image = img
        # Clear ROI when a new image (different shape) is loaded
        if img is None:
            self._roi = None
        self._render()

    def enable_roi(self, enabled: bool) -> None:
        self._roi_enabled = bool(enabled)
        if not enabled:
            self._roi = None
            self._render()

    def get_roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    def clear_roi(self) -> None:
        self._roi = None
        self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self) -> None:
        self.delete("all")
        if self._image is None:
            return

        cw = max(self.winfo_width(), 1)
        ch = max(self.winfo_height(), 1)
        h, w = self._image.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        off_x = (cw - new_w) // 2
        off_y = (ch - new_h) // 2

        if self._image.ndim == 2:
            rgb = cv2.cvtColor(self._image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(pil)
        self.create_image(off_x, off_y, image=self._tk_image, anchor="nw")

        self._render_scale = scale
        self._render_offset = (off_x, off_y)

        # Draw saved ROI (in image coords) on top
        if self._roi is not None:
            x, y, rw, rh = self._roi
            cx0, cy0 = self._image_to_canvas(x, y)
            cx1, cy1 = self._image_to_canvas(x + rw, y + rh)
            self.create_rectangle(cx0, cy0, cx1, cy1,
                                  outline="#ffeb3b", width=2, tags="roi")

    def _on_resize(self, _event) -> None:
        self._render()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------
    def _canvas_to_image(self, cx: int, cy: int) -> tuple[int, int]:
        ox, oy = self._render_offset
        s = self._render_scale
        if s <= 0:
            return 0, 0
        return int((cx - ox) / s), int((cy - oy) / s)

    def _image_to_canvas(self, ix: int, iy: int) -> tuple[float, float]:
        ox, oy = self._render_offset
        s = self._render_scale
        return ix * s + ox, iy * s + oy

    # ------------------------------------------------------------------
    # ROI mouse handlers
    # ------------------------------------------------------------------
    def _on_press(self, event) -> None:
        if not self._roi_enabled or self._image is None:
            return
        self._drag_start_canvas = (event.x, event.y)
        self._roi = None
        self._render()

    def _on_drag(self, event) -> None:
        if not self._roi_enabled or self._drag_start_canvas is None:
            return
        # Re-render base + draw live rect
        self._render()
        x0, y0 = self._drag_start_canvas
        self.create_rectangle(x0, y0, event.x, event.y,
                              outline="#ffeb3b", dash=(4, 2), width=2,
                              tags="roi-live")

    def _on_release(self, event) -> None:
        if not self._roi_enabled or self._drag_start_canvas is None:
            return
        x0, y0 = self._drag_start_canvas
        x1, y1 = event.x, event.y
        self._drag_start_canvas = None
        if self._image is None:
            return

        ix0, iy0 = self._canvas_to_image(min(x0, x1), min(y0, y1))
        ix1, iy1 = self._canvas_to_image(max(x0, x1), max(y0, y1))
        h, w = self._image.shape[:2]
        ix0 = max(0, min(w - 1, ix0))
        iy0 = max(0, min(h - 1, iy0))
        ix1 = max(0, min(w, ix1))
        iy1 = max(0, min(h, iy1))
        rw, rh = ix1 - ix0, iy1 - iy0
        if rw < 2 or rh < 2:
            self._roi = None
        else:
            self._roi = (ix0, iy0, rw, rh)
        self._render()
