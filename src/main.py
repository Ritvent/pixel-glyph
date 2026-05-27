"""PixelGlyph — Steganography Lab entry point.

Run from project root:
    python src/main.py
"""
from gui.app import PixelGlyphApp


def main() -> None:
    app = PixelGlyphApp()
    app.mainloop()


if __name__ == "__main__":
    main()
