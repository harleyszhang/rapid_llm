"""Shared visualization library for script/ directory GIF & PNG generation.

Centralises the dark-theme palette, font loading, title-bar chrome, and GIF
saving that every ``gen_*.py`` re-implements independently.  Every colour
constant and layout number lives here once; downstream files import what they
need and only define constants unique to that visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Terminal palette (dark theme)
# --------------------------------------------------------------------------- #

BG = (14, 16, 20)
PANEL_BG = (22, 25, 31)
BAR_BG = (38, 42, 52)
TITLE_BG = (32, 36, 44)
PANEL_EDGE = (52, 58, 68)
GRID_LINE = (52, 58, 68)
AXIS_FG = (52, 58, 68)

TITLE_FG = (222, 226, 232)
TEXT_FG = (222, 226, 232)
PROMPT_FG = (118, 214, 118)
DIM = (128, 136, 148)

GREEN = (94, 193, 117)
RED = (245, 99, 72)
YELLOW = (253, 188, 64)
BLUE = (88, 166, 255)
CYAN = (86, 198, 224)
AMBER = (226, 184, 92)
PURPLE = (192, 132, 252)
ORANGE = (253, 188, 64)

DATA_FG = (120, 190, 240)
CONTROL_FG = (200, 150, 240)
ALERT = (240, 140, 110)

RANK0_FG = (118, 214, 118)
RANK1_FG = (120, 190, 240)

RUNNING = (118, 214, 118)
QUEUED = (226, 184, 92)
DONE = (110, 160, 226)

PREFILL_FG = AMBER
DECODE_OK = GREEN
STALLED = RED

# --------------------------------------------------------------------------- #
# Common layout constants (most GIFs)
# --------------------------------------------------------------------------- #

TITLE_H = 36
PAD = 18
LINE_H = 25
LABEL_W = 130

# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

_DEJAVU_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_DEJAVU_MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_DEJAVU_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class FontPack:
    """Three-weight font set (body, bold, small) for terminal-style rendering."""

    body: ImageFont.FreeTypeFont
    bold: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont

    @classmethod
    def mono(
        cls, body_size: int = 16, bold_size: int = 16, small_size: int = 14
    ) -> FontPack:
        """DejaVuSansMono-based font triple."""
        try:
            return cls(
                ImageFont.truetype(_DEJAVU_MONO, body_size),
                ImageFont.truetype(_DEJAVU_MONO_BOLD, bold_size),
                ImageFont.truetype(_DEJAVU_MONO, small_size),
            )
        except OSError:
            d = ImageFont.load_default()
            return cls(d, d, d)

    @classmethod
    def sans(
        cls, body_size: int = 16, bold_size: int = 16, small_size: int = 14
    ) -> FontPack:
        """DejaVuSans-based font triple (for dense panels)."""
        try:
            return cls(
                ImageFont.truetype(_DEJAVU_SANS, body_size),
                ImageFont.truetype(_DEJAVU_SANS_BOLD, bold_size),
                ImageFont.truetype(_DEJAVU_SANS, small_size),
            )
        except OSError:
            d = ImageFont.load_default()
            return cls(d, d, d)


def default_font(body_size: int = 16, bold_size: int = 16, small_size: int = 14) -> FontPack:
    """Mono-spaced font triple; the common case for terminal GIFs."""
    return FontPack.mono(body_size, bold_size, small_size)


# --------------------------------------------------------------------------- #
# Title bar chrome
# --------------------------------------------------------------------------- #


def draw_title_bar(
    draw: ImageDraw.ImageDraw, w: int, text: str, *, small: ImageFont.FreeTypeFont | None = None
) -> None:
    """Dark title bar with three traffic-light circles on the right."""
    draw.rectangle([0, 0, w, TITLE_H], fill=TITLE_BG)
    f = small or ImageFont.load_default()
    draw.text((12, 9), text, fill=TITLE_FG, font=f)
    for i, colour in enumerate([RED, YELLOW, GREEN]):
        draw.ellipse([w - 78 + i * 18, 11, w - 68 + i * 18, 21], fill=colour)


# --------------------------------------------------------------------------- #
# GIF saving
# --------------------------------------------------------------------------- #


def save_gif(
    frames: list[Image.Image],
    path: Path,
    *,
    duration: int | list[int] = 500,
    loop: int = 0,
    optimize: bool = True,
    colors: int = 64,
) -> None:
    """Quantise to palette and write an animated GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=colors) for im in frames]
    palette[0].save(
        path,
        save_all=True,
        append_images=palette[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
    )


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    filled: int,
    total: int,
    colour,
    width: int = 260,
    height: int = 16,
) -> None:
    """A horizontal progress/occupancy bar."""
    draw.rectangle([x, y + 4, x + width, y + height - 4], fill=BAR_BG)
    bar_w = int(width * filled / total) if total else 0
    if bar_w > 0:
        draw.rectangle([x, y + 4, x + bar_w, y + height - 4], fill=colour)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str | None = None,
    *,
    radius: int = 8,
) -> None:
    """Rounded-rect panel with optional top-bar title."""
    draw.rounded_rectangle(box, radius=radius, fill=PANEL_BG, outline=PANEL_EDGE)
    if title:
        draw.rectangle([box[0], box[1], box[2], box[1] + 28], fill=TITLE_BG)
        draw.text((box[0] + 10, box[1] + 5), title, font=ImageFont.load_default(), fill=DIM)