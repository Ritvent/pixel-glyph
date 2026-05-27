"""Main Tkinter window for PixelGlyph."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Optional

import numpy as np

from gui.panels import ToolPanel
from gui.widgets import ImageCanvas
from utils.io import load_image, save_image


UNDO_LIMIT = 20


class AppState:
    """Holds the cover (loaded) image, the working image (stego or post-
    attack stego), and a bounded undo stack."""

    def __init__(self) -> None:
        self.cover: Optional[np.ndarray] = None
        self.working: Optional[np.ndarray] = None
        self.path: Optional[Path] = None
        self.undo_stack: list[np.ndarray] = []

    def set_cover(self, img: np.ndarray, path: str | Path) -> None:
        self.cover = img.copy()
        self.working = img.copy()
        self.path = Path(path)
        self.undo_stack.clear()

    def apply(self, new_img: np.ndarray) -> None:
        if self.working is not None:
            self.undo_stack.append(self.working.copy())
            if len(self.undo_stack) > UNDO_LIMIT:
                self.undo_stack.pop(0)
        self.working = new_img

    def undo(self) -> None:
        if self.undo_stack:
            self.working = self.undo_stack.pop()

    def reset(self) -> None:
        if self.cover is not None:
            self.working = self.cover.copy()
            self.undo_stack.clear()


class PixelGlyphApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PixelGlyph — Steganography Lab")
        self.geometry("1280x800")
        self.minsize(900, 600)
        self.state_: AppState = AppState()  # avoid clash with tk.Tk.state()
        self._build_menu()
        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        bar = tk.Menu(self)
        filem = tk.Menu(bar, tearoff=0)
        filem.add_command(label="Open Cover...", command=self.cmd_open,
                          accelerator="Ctrl+O")
        filem.add_command(label="Save Working As... (PNG)",
                          command=self.cmd_save, accelerator="Ctrl+S")
        filem.add_separator()
        filem.add_command(label="Reset Working to Cover", command=self.cmd_reset)
        filem.add_command(label="Undo", command=self.cmd_undo,
                          accelerator="Ctrl+Z")
        filem.add_separator()
        filem.add_command(label="Quit", command=self.destroy)
        bar.add_cascade(label="File", menu=filem)
        self.config(menu=bar)
        self.bind_all("<Control-o>", lambda _e: self.cmd_open())
        self.bind_all("<Control-s>", lambda _e: self.cmd_save())
        self.bind_all("<Control-z>", lambda _e: self.cmd_undo())

    def _build_layout(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Cover").pack(anchor="w", padx=4, pady=2)
        self.canvas_cover = ImageCanvas(left)
        self.canvas_cover.pack(fill="both", expand=True, padx=4, pady=2)

        mid = ttk.Frame(main)
        mid.pack(side="left", fill="both", expand=True)
        ttk.Label(mid, text="Working (stego / post-attack)"
                  ).pack(anchor="w", padx=4, pady=2)
        self.canvas_working = ImageCanvas(mid)
        self.canvas_working.pack(fill="both", expand=True, padx=4, pady=2)

        right = ttk.Frame(main, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.tool_panel = ToolPanel(right, self)
        self.tool_panel.pack(fill="both", expand=True)

        self.status = ttk.Label(self, text="Ready — open a cover image to begin.",
                                anchor="w")
        self.status.pack(side="bottom", fill="x")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def cmd_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Open cover image",
            filetypes=[
                ("Images", "*.png *.bmp *.tif *.tiff *.jpg *.jpeg"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            self.status.config(text=f"Error: {e}")
            return
        self.state_.set_cover(img, path)
        self.refresh()
        self.tool_panel.on_image_loaded()
        self.status.config(
            text=f"Loaded: {path}  ({img.shape[1]}x{img.shape[0]})"
        )

    def cmd_save(self) -> None:
        if self.state_.working is None:
            self.status.config(text="Nothing to save — open an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialdir="output",
            initialfile="stego.png",
            filetypes=[("PNG (lossless)", "*.png"),
                       ("BMP (lossless)", "*.bmp"),
                       ("JPEG (DESTROYS stego)", "*.jpg *.jpeg")],
        )
        if not path:
            return
        try:
            save_image(self.state_.working, path)
        except Exception as e:
            self.status.config(text=f"Save failed: {e}")
            return
        warn = ""
        if path.lower().endswith((".jpg", ".jpeg")):
            warn = "  ⚠ JPEG was used — any LSB payload is now destroyed."
        self.status.config(text=f"Saved: {path}{warn}")

    def cmd_undo(self) -> None:
        self.state_.undo()
        self.refresh()
        self.status.config(text=f"Undo. {len(self.state_.undo_stack)} step(s) remain.")

    def cmd_reset(self) -> None:
        self.state_.reset()
        self.refresh()
        self.status.config(text="Working reset to Cover.")

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.canvas_cover.show(self.state_.cover)
        self.canvas_working.show(self.state_.working)
