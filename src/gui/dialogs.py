"""Auxiliary Toplevel windows (histogram, stats, bit-plane grid, etc.)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


def show_histogram(parent: tk.Misc, img: np.ndarray, title: str = "Histogram") -> None:
    """Open a Toplevel window with the histogram of `img`."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("640x420")

    fig = Figure(figsize=(6.2, 4.0), dpi=100)
    ax = fig.add_subplot(111)

    if img.ndim == 2:
        ax.hist(img.ravel(), bins=256, range=(0, 256), color="black")
    else:
        colors = ("b", "g", "r")
        for c, col in enumerate(colors):
            ax.hist(img[:, :, c].ravel(), bins=256, range=(0, 256),
                    color=col, alpha=0.45, label=col.upper())
        ax.legend()

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 255)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)


def show_histogram_compare(parent: tk.Misc,
                           img_a: np.ndarray,
                           img_b: np.ndarray,
                           label_a: str = "Cover",
                           label_b: str = "Working") -> None:
    """Overlay luminance histograms of two images in a Toplevel window."""
    win = tk.Toplevel(parent)
    win.title(f"Histogram — {label_a} vs {label_b}")
    win.geometry("780x460")

    fig = Figure(figsize=(7.5, 4.2), dpi=100)
    ax = fig.add_subplot(111)

    def _lum(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ax.hist(_lum(img_a).ravel(), bins=256, range=(0, 256),
            alpha=0.55, label=label_a, color="#1976d2")
    ax.hist(_lum(img_b).ravel(), bins=256, range=(0, 256),
            alpha=0.55, label=label_b, color="#d32f2f")
    ax.legend()
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 255)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)


def show_line_plot(parent: tk.Misc, y: np.ndarray, title: str,
                   xlabel: str = "index", ylabel: str = "value") -> None:
    """Generic 1D line plot in a Toplevel (used for 1D FFT row/col)."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("680x420")

    fig = Figure(figsize=(6.5, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(y)), y, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(len(y) - 1, 1))
    ax.grid(True, alpha=0.3)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)


def show_text(parent: tk.Misc, title: str, text: str) -> None:
    """Generic read-only text window (used for stats output)."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("420x360")
    txt = tk.Text(win, wrap="none", font=("Consolas", 10))
    txt.insert("1.0", text)
    txt.config(state="disabled")
    txt.pack(fill="both", expand=True, padx=8, pady=8)
    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)


def show_image(parent: tk.Misc, img: np.ndarray, title: str = "Image",
               max_size: int = 900) -> None:
    """Generic image preview window (used for the bit-plane grid)."""
    win = tk.Toplevel(parent)
    win.title(title)

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
    photo = ImageTk.PhotoImage(pil)

    label = ttk.Label(win, image=photo)
    label.image = photo  # keep ref
    label.pack(padx=8, pady=8)
    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)
