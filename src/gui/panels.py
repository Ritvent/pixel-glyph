"""Tool panels — one tab per major workflow.

Tabs:
    Encode    — write text into the cover image's LSBs
    Decode    — extract text from the working image's LSBs
    Attack    — Exer 1, 3, 8, 9, 11 — apply DIP ops as attacks on the
                stego, then re-run Decode to see if message survives
    Analyze   — Exer 4, 7, 12 — cover-vs-working invisibility metrics

Hour 6-7 will add: Forensics.
Hour 8 will add JPEG-as-attack to the Attack tab + DPCM/Huffman stats.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import numpy as np

from gui.dialogs import (
    show_histogram, show_histogram_compare, show_image, show_text,
)
from ops import basic, contrast, filters, restore, slicing, stats, stego


class ToolPanel(ttk.Frame):
    def __init__(self, master, app) -> None:
        super().__init__(master)
        self.app = app

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        for label, builder in (
            ("Encode", self._build_encode),
            ("Decode", self._build_decode),
            ("Attack", self._build_attack),
            ("Analyze", self._build_analyze),
        ):
            tab = _scrollable(self.nb)
            self.nb.add(tab.outer, text=label)
            builder(tab.inner)

    # ==================================================================
    # Encode tab
    # ==================================================================
    def _build_encode(self, parent: ttk.Frame) -> None:
        _section(parent, "1. Open a cover image (File → Open Cover)")

        self.capacity_label = ttk.Label(parent, text="Capacity: —",
                                        foreground="#666")
        self.capacity_label.pack(anchor="w", padx=6, pady=(0, 8))

        _section(parent, "2. Type your secret message")
        msg_frame = ttk.Frame(parent)
        msg_frame.pack(fill="both", expand=True, padx=6, pady=2)
        self.msg_text = tk.Text(msg_frame, height=12, wrap="word",
                                font=("Consolas", 10))
        msg_scroll = ttk.Scrollbar(msg_frame, orient="vertical",
                                   command=self.msg_text.yview)
        self.msg_text.configure(yscrollcommand=msg_scroll.set)
        self.msg_text.pack(side="left", fill="both", expand=True)
        msg_scroll.pack(side="right", fill="y")
        self.msg_text.bind("<KeyRelease>", lambda _e: self._refresh_capacity())

        self.msg_len_label = ttk.Label(parent, text="0 bytes", foreground="#666")
        self.msg_len_label.pack(anchor="w", padx=6, pady=(2, 8))

        _section(parent, "3. Encode")
        _button(parent, "Encode → Working canvas", self._cmd_encode)

        ttk.Label(parent,
                  text="After encoding, use File → Save As (PNG).\n"
                       "Saving as JPEG will destroy the message.",
                  foreground="#a40", justify="left", wraplength=320
                  ).pack(anchor="w", padx=6, pady=(4, 0))

    def _refresh_capacity(self) -> None:
        msg = self.msg_text.get("1.0", "end-1c")
        msg_bytes = len(msg.encode("utf-8"))
        self.msg_len_label.config(text=f"{msg_bytes} bytes")
        if self.app.state_.cover is None:
            self.capacity_label.config(text="Capacity: — (load a cover)")
            return
        cap = stego.capacity_bytes(self.app.state_.cover)
        pct = (msg_bytes / cap * 100) if cap else 0.0
        color = "#666" if msg_bytes <= cap else "#c00"
        self.capacity_label.config(
            text=f"Capacity: {cap:,} bytes  ({pct:.2f}% used)",
            foreground=color,
        )

    def on_image_loaded(self) -> None:
        """Called by the app after a new cover image is loaded."""
        self._refresh_capacity()

    def _cmd_encode(self) -> None:
        if self.app.state_.cover is None:
            self.app.status.config(text="Open a cover image first.")
            return
        msg = self.msg_text.get("1.0", "end-1c")
        if not msg:
            self.app.status.config(text="Type a message before encoding.")
            return
        try:
            stego_img = stego.encode_text(self.app.state_.cover, msg)
        except ValueError as e:
            self.app.status.config(text=f"Encode failed: {e}")
            return
        self.app.state_.apply(stego_img)
        self.app.refresh()
        n = len(msg.encode("utf-8"))
        self.app.status.config(
            text=f"Encoded {n} bytes. Save as PNG to preserve."
        )

    # ==================================================================
    # Decode tab
    # ==================================================================
    def _build_decode(self, parent: ttk.Frame) -> None:
        _section(parent,
                 "Decode reads from the WORKING canvas.\n"
                 "Open a stego image (it loads into both canvases).")

        _button(parent, "Decode message", self._cmd_decode)

        _section(parent, "Decoded text")
        out_frame = ttk.Frame(parent)
        out_frame.pack(fill="both", expand=True, padx=6, pady=2)
        self.out_text = tk.Text(out_frame, height=14, wrap="word",
                                font=("Consolas", 10), state="disabled")
        out_scroll = ttk.Scrollbar(out_frame, orient="vertical",
                                   command=self.out_text.yview)
        self.out_text.configure(yscrollcommand=out_scroll.set)
        self.out_text.pack(side="left", fill="both", expand=True)
        out_scroll.pack(side="right", fill="y")

        _button(parent, "Copy to clipboard", self._cmd_copy_decoded)

    def _cmd_decode(self) -> None:
        if self.app.state_.working is None:
            self.app.status.config(text="Open an image first.")
            return
        try:
            text = stego.decode_text(self.app.state_.working)
        except ValueError as e:
            self._write_decoded("")
            self.app.status.config(
                text=f"No valid stego payload detected: {e}"
            )
            return
        self._write_decoded(text)
        self.app.status.config(text=f"Decoded {len(text.encode('utf-8'))} bytes.")

    def _write_decoded(self, text: str) -> None:
        self.out_text.config(state="normal")
        self.out_text.delete("1.0", "end")
        self.out_text.insert("1.0", text)
        self.out_text.config(state="disabled")

    def _cmd_copy_decoded(self) -> None:
        text = self.out_text.get("1.0", "end-1c")
        if not text:
            self.app.status.config(text="Nothing to copy.")
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(text)
        self.app.status.config(text="Decoded text copied to clipboard.")

    # ==================================================================
    # Attack tab — Exer 1, 3, 8, 9, 11
    # ==================================================================
    def _build_attack(self, parent: ttk.Frame) -> None:
        _intro(parent,
               "Each button applies a DIP operation to the WORKING image. "
               "Then switch to Decode to see if the message survives.")

        _section(parent, "Exer 1 — Negative")
        _button(parent, "Apply Negative",
                lambda: self._apply(basic.negative, "negative"))

        _separator(parent)
        _section(parent, "Exer 3 — Geometric / Intensity")
        self.atk_rot = _slider(parent, "Rotate (degrees)", -180, 180, 0)
        _button(parent, "Apply Rotation",
                lambda: self._apply(
                    lambda im: basic.rotate(im, self.atk_rot.get()),
                    "rotate",
                ))
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=2)
        ttk.Button(row, text="Flip H",
                   command=lambda: self._apply(
                       lambda im: basic.flip(im, "horizontal"), "flip-h"
                   )).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(row, text="Flip V",
                   command=lambda: self._apply(
                       lambda im: basic.flip(im, "vertical"), "flip-v"
                   )).pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.atk_scale = _slider(parent, "Scale", 0.25, 2.0, 1.0)
        _button(parent, "Apply Scale",
                lambda: self._apply(
                    lambda im: basic.resize(im, self.atk_scale.get()),
                    "scale",
                ))
        self.atk_gamma = _slider(parent, "Gamma", 0.1, 3.0, 1.0)
        _button(parent, "Apply Gamma",
                lambda: self._apply(
                    lambda im: basic.gamma_correct(im, self.atk_gamma.get()),
                    "gamma",
                ))

        _separator(parent)
        _section(parent, "Exer 8 — Smoothing")
        self.atk_smooth_k = _slider(parent, "Kernel size",
                                    1, 21, 3, integer=True)
        _button(parent, "Mean filter",
                lambda: self._apply(
                    lambda im: filters.mean_filter(im, self.atk_smooth_k.get()),
                    "mean",
                ))
        _button(parent, "Median filter",
                lambda: self._apply(
                    lambda im: filters.median_filter(im, self.atk_smooth_k.get()),
                    "median",
                ))
        _button(parent, "Gaussian filter",
                lambda: self._apply(
                    lambda im: filters.gaussian_filter(im, self.atk_smooth_k.get()),
                    "gaussian",
                ))

        _separator(parent)
        _section(parent, "Exer 9 — Sharpening")
        self.atk_sharp_amt = _slider(parent, "Sharpen amount", 0.1, 3.0, 1.0)
        _button(parent, "Laplacian sharpen",
                lambda: self._apply(
                    lambda im: filters.laplacian_sharpen(
                        im, self.atk_sharp_amt.get()),
                    "laplacian",
                ))
        _button(parent, "Unsharp mask",
                lambda: self._apply(
                    lambda im: filters.unsharp_mask(
                        im, self.atk_sharp_amt.get(), 1.5),
                    "unsharp",
                ))

        _separator(parent)
        _section(parent, "Exer 11 — Restoration (also acts as attack)")
        self.atk_nlm = _slider(parent, "NLM strength", 1, 30, 10)
        _button(parent, "Non-local Means denoise",
                lambda: self._apply(
                    lambda im: restore.denoise_nlm(im, self.atk_nlm.get()),
                    "nlm",
                ))
        _button(parent, "Median denoise (k=5)",
                lambda: self._apply(
                    lambda im: restore.denoise_median(im, 5),
                    "median-denoise",
                ))

    # ==================================================================
    # Analyze tab — Exer 4, 7, 12
    # ==================================================================
    def _build_analyze(self, parent: ttk.Frame) -> None:
        _intro(parent,
               "Compare Cover vs Working to measure stego invisibility and "
               "visualize what changed.")

        _section(parent, "Exer 7 — Invisibility metrics")
        _button(parent, "Show cover-vs-working stats",
                self._cmd_show_compare_stats)

        ttk.Label(parent, text="Pick ROI on Working canvas (drag):"
                  ).pack(anchor="w", padx=6, pady=(8, 0))
        self.roi_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="ROI mode on", variable=self.roi_on,
                        command=self._toggle_roi
                        ).pack(anchor="w", padx=6, pady=2)
        _button(parent, "Clear ROI", self._clear_roi)
        _button(parent, "Show stats (ROI of working)", self._cmd_show_roi_stats)
        _button(parent, "Correlation (cover vs working, ROI)",
                self._cmd_show_correlation)

        _separator(parent)
        _section(parent, "Exer 4 — Histogram comparison")
        _button(parent, "Histogram (cover)",
                lambda: self._cmd_histogram("cover"))
        _button(parent, "Histogram (working)",
                lambda: self._cmd_histogram("working"))
        _button(parent, "Histogram overlay (cover vs working)",
                self._cmd_histogram_compare)

        _separator(parent)
        _section(parent, "Exer 12 — Diff visualization")
        _button(parent, "Show diff mask (where pixels changed)",
                self._cmd_diff_mask)
        self.diff_gain = _slider(parent, "Amplify diff (gain)",
                                 1, 100, 50, integer=True)
        _button(parent, "Show amplified diff",
                self._cmd_diff_amplified)

    # ------------------------------------------------------------------
    # Analyze tab handlers
    # ------------------------------------------------------------------
    def _toggle_roi(self) -> None:
        on = self.roi_on.get()
        self.app.canvas_working.enable_roi(on)
        self.app.status.config(
            text="ROI mode ON — drag on Working canvas." if on
            else "ROI mode OFF."
        )

    def _clear_roi(self) -> None:
        self.app.canvas_working.clear_roi()
        self.app.status.config(text="ROI cleared.")

    def _cmd_show_compare_stats(self) -> None:
        s = self.app.state_
        if s.cover is None or s.working is None:
            self.app.status.config(text="Open a cover and encode first.")
            return
        out = stats.compare_stats(s.cover, s.working)
        show_text(self.app, "Cover vs Working — invisibility metrics",
                  stats.format_stats(out))

    def _cmd_show_roi_stats(self) -> None:
        if self.app.state_.working is None:
            self.app.status.config(text="Open an image first.")
            return
        roi = self.app.canvas_working.get_roi()
        out = stats.compute_stats(self.app.state_.working, roi)
        scope = f"ROI {roi}" if roi else "whole image"
        show_text(self.app, f"Stats — working ({scope})",
                  stats.format_stats(out))

    def _cmd_show_correlation(self) -> None:
        s = self.app.state_
        if s.cover is None or s.working is None:
            self.app.status.config(text="Need both cover and working.")
            return
        roi = self.app.canvas_working.get_roi()
        try:
            r = stats.correlation_coefficient(s.cover, s.working, roi)
        except Exception as e:
            self.app.status.config(text=f"Correlation failed: {e}")
            return
        scope = f"ROI {roi}" if roi else "whole image"
        show_text(self.app, "Pearson correlation",
                  f"Cover vs Working ({scope})\n\n  r = {r:.6f}")

    def _cmd_histogram(self, which: str) -> None:
        img = self.app.state_.cover if which == "cover" else self.app.state_.working
        if img is None:
            self.app.status.config(text="Open an image first.")
            return
        show_histogram(self.app, img, title=f"Histogram — {which}")

    def _cmd_histogram_compare(self) -> None:
        s = self.app.state_
        if s.cover is None or s.working is None:
            self.app.status.config(text="Need both cover and working.")
            return
        show_histogram_compare(self.app, s.cover, s.working,
                               label_a="Cover", label_b="Working")

    def _cmd_diff_mask(self) -> None:
        s = self.app.state_
        if s.cover is None or s.working is None:
            self.app.status.config(text="Need both cover and working.")
            return
        try:
            m = stats.diff_mask(s.cover, s.working)
        except ValueError as e:
            self.app.status.config(text=f"{e}")
            return
        show_image(self.app, m, title="Diff mask — white = pixel changed")

    def _cmd_diff_amplified(self) -> None:
        s = self.app.state_
        if s.cover is None or s.working is None:
            self.app.status.config(text="Need both cover and working.")
            return
        try:
            d = stats.diff_amplified(s.cover, s.working, self.diff_gain.get())
        except ValueError as e:
            self.app.status.config(text=f"{e}")
            return
        show_image(self.app, d,
                   title=f"|cover - working| × {int(self.diff_gain.get())}")

    # ==================================================================
    # Apply helper (used by Attack tab)
    # ==================================================================
    def _apply(self,
               fn: Callable[[np.ndarray], np.ndarray],
               name: Optional[str] = None) -> None:
        if self.app.state_.working is None:
            self.app.status.config(text="Open an image first.")
            return
        try:
            out = fn(self.app.state_.working)
        except Exception as e:
            self.app.status.config(text=f"Op failed: {e}")
            return
        self.app.state_.apply(out)
        self.app.refresh()
        self.app.status.config(
            text=f"Attack applied: {name or 'op'} — try Decode now."
        )


# ---------------------------------------------------------------------------
# Tiny ttk helpers (kept local to this module)
# ---------------------------------------------------------------------------
def _section(parent: ttk.Frame, text: str) -> None:
    ttk.Label(parent, text=text, font=("", 10, "bold"), justify="left",
              wraplength=300).pack(anchor="w", padx=6, pady=(8, 2))


def _intro(parent: ttk.Frame, text: str) -> None:
    ttk.Label(parent, text=text, foreground="#444", justify="left",
              wraplength=300).pack(anchor="w", padx=6, pady=(6, 6))


def _separator(parent: ttk.Frame) -> None:
    ttk.Separator(parent).pack(fill="x", pady=8, padx=6)


def _button(parent: ttk.Frame, text: str, command: Callable[[], None]) -> None:
    ttk.Button(parent, text=text, command=command).pack(fill="x", padx=6, pady=2)


def _slider(parent: ttk.Frame, label: str,
            lo: float, hi: float, init: float,
            integer: bool = False) -> tk.Variable:
    ttk.Label(parent, text=label).pack(anchor="w", padx=6, pady=(6, 0))
    var: tk.Variable = (
        tk.IntVar(value=int(init)) if integer else tk.DoubleVar(value=float(init))
    )
    ttk.Scale(parent, from_=lo, to=hi, variable=var,
              orient="horizontal").pack(fill="x", padx=6)
    return var


class _Scrollable:
    """Container for a vertically-scrollable Frame inside a Notebook tab."""
    def __init__(self, outer: ttk.Frame, inner: ttk.Frame) -> None:
        self.outer = outer
        self.inner = inner


def _scrollable(notebook: ttk.Notebook) -> _Scrollable:
    """Build a scrollable Frame to use as a Notebook tab. Returns a holder
    where `.outer` is the widget to .add() and `.inner` is where to pack
    your content."""
    outer = ttk.Frame(notebook)
    canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    canvas.configure(yscrollcommand=vsb.set)

    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_inner_configure(_e) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(e) -> None:
        # make inner match canvas width so children fill horizontally
        canvas.itemconfigure(window_id, width=e.width)

    inner.bind("<Configure>", _on_inner_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    # Mouse wheel only when pointer is over THIS canvas (not global)
    def _on_wheel(e) -> None:
        canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_wheel))
    canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

    return _Scrollable(outer, inner)
