#!/usr/bin/env python3
"""Render the GitHub profile set from running.jpg.

Each entry produces .png (raster), .svg (crisp at any size, good for a README),
.txt (plain characters) and .ans (24-bit colour for a terminal).
"""

import os
from dataclasses import replace

from PIL import Image

import ascii_art as A

SRC = "running.jpg"
BASE = A.Config(
    look="sunlit", saturation=1.0, matte=True, matte_cache="running.matte.png"
)

OUTPUTS = [
    # name              aspect zoom  focus_y cols scale overrides
    ("avatar",          "1:1", 1.50, -0.05,   56,   4, {}),
    ("avatar_detail",   "1:1", 1.50, -0.05,   96,   3, {}),
    # tight portrait: skin fills the frame, so pull the chroma back and use the
    # cooler grade, otherwise it reads sunburnt
    ("avatar_portrait", "1:1", 2.80, -0.15,   60,   4,
     {"look": "film", "saturation": 0.8}),
    ("full",            None,  1.00,  0.00,   80,   3, {}),

    # --- wide set -------------------------------------------------------
    # zoom 1.0 keeps the whole runner in frame instead of cropping into the
    # torso. Column counts are chosen to hold one character per ~21px of the
    # source, so the glyphs stay the same physical size across the set.
    ("avatar_wide",         "1:1", 1.00,  0.00,   78,   4, {}),
    ("avatar_wide_detail",  "1:1", 1.00,  0.00,  110,   3, {}),
    ("avatar_wide_film",    "1:1", 1.00,  0.00,   78,   4,
     {"look": "film", "saturation": 0.85}),
    ("full_wide",           "4:5", 1.00, -0.02,   92,   3, {}),
]


def main() -> None:
    os.makedirs("out", exist_ok=True)
    src = Image.open(SRC)
    for name, aspect, zoom, fy, cols, scale, over in OUTPUTS:
        cfg = replace(BASE, aspect=aspect, zoom=zoom, focus_y=fy, cols=cols, **over)
        res = A.convert(src, cfg)
        img = A.composite(res, cfg, scale=scale)
        img.save(f"out/{name}.png")
        for ext, make in (
            ("txt", lambda: A.to_text(res)),
            ("svg", lambda: A.to_svg(res, cfg)),
            ("ans", lambda: A.to_ansi(res, cfg)),
        ):
            with open(f"out/{name}.{ext}", "w") as f:
                f.write(make())
        print(f"  ✓ {name:16s} {res['cols']}x{res['rows']} chars -> {img.size[0]}x{img.size[1]} px")


if __name__ == "__main__":
    main()
