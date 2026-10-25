"""The app icon (the approved design's mark: a relay line on a blue rounded square), drawn
with Pillow so no image files have to ship."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BLUE = (41, 82, 204, 255)  # #2952CC
WHITE = (255, 255, 255, 255)
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def make_icon(size: int = 256, *, badge: bool = False) -> Any:
    """A square RGBA image. ``badge`` adds an orange dot (something needs you)."""
    from PIL import Image, ImageDraw

    scale = 4  # draw large, then downsample: smooth edges at every size
    big = size * scale
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    unit = big / 28  # the design's 28-unit viewBox
    draw.rounded_rectangle((unit, unit, 27 * unit, 27 * unit), radius=7 * unit, fill=BLUE)
    points = [(7, 17), (12, 17), (15, 11), (18, 17), (21, 17)]
    draw.line([(x * unit, y * unit) for x, y in points], fill=WHITE, width=round(2.6 * unit),
              joint="curve")  # fmt: skip
    for x, y in (points[0], points[-1]):
        r = 1.3 * unit
        draw.ellipse((x * unit - r, y * unit - r, x * unit + r, y * unit + r), fill=WHITE)
    if badge:
        r = 5 * unit
        draw.ellipse((22 * unit - r, 6 * unit - r, 22 * unit + r, 6 * unit + r),
                     fill=(201, 138, 30, 255), outline=WHITE, width=round(1.2 * unit))  # fmt: skip
    return img.resize((size, size), Image.Resampling.LANCZOS)


def save_ico(path: Path) -> Path:
    """Write a multi-size Windows .ico (used by the packaged app)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    make_icon(256).save(path, format="ICO", sizes=ICO_SIZES)
    return path
