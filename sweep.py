#!/usr/bin/env python3
"""Render a grid of parameter variants into one labelled contact sheet."""

from __future__ import annotations

import sys
from dataclasses import replace

from PIL import Image, ImageDraw, ImageFont

import ascii_art as A

FONT = ImageFont.truetype(A.DEFAULT_FONT, 20)


def sheet(src_path: str, base: A.Config, variants: list[dict], out: str, tile: int = 430,
          per_row: int = 3) -> None:
    src = Image.open(src_path)
    tiles = []
    for v in variants:
        cfg = replace(base, **{k: val for k, val in v.items() if k != "label"})
        res = A.convert(src, cfg)
        img = A.composite(res, cfg)
        img.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        tiles.append((v.get("label", ""), img))
        print(f"   · {v.get('label','')}")

    tw = max(t.width for _, t in tiles)
    th = max(t.height for _, t in tiles)
    pad, lab = 12, 30
    rows = (len(tiles) + per_row - 1) // per_row
    W = per_row * (tw + pad) + pad
    H = rows * (th + lab + pad) + pad
    sheet_img = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet_img)
    for i, (label, t) in enumerate(tiles):
        r, c = divmod(i, per_row)
        x = pad + c * (tw + pad)
        y = pad + r * (th + lab + pad)
        d.text((x, y + 4), label, font=FONT, fill=(235, 235, 240))
        sheet_img.paste(t, (x, y + lab))
    sheet_img.save(out)
    print(f"  ✓ {out} {sheet_img.size}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "charset"
    base = A.Config(
        cols=100, aspect="1:1", matte=True, matte_cache="running.matte.png", bloom=0.25
    )

    if which == "charset":
        variants = [
            {"label": "ascii fg1.2", "charset": "ascii", "fg_gain": 1.2},
            {"label": "ascii fg1.9", "charset": "ascii", "fg_gain": 1.9},
            {"label": "ascii fg2.6", "charset": "ascii", "fg_gain": 2.6},
            {"label": "mixed fg1.2", "charset": "mixed", "fg_gain": 1.2},
            {"label": "code fg1.9", "charset": "code", "fg_gain": 1.9},
            {"label": "minimal fg2.6", "charset": "minimal", "fg_gain": 2.6},
        ]
    elif which == "look":
        variants = [
            {"label": l, "look": l, "charset": "ascii", "fg_gain": 2.2}
            for l in ["neutral", "film", "sunlit", "neon", "cold", "mono"]
        ]
    elif which == "edges":
        variants = [
            {"label": f"edges {e}", "edges": e, "charset": "ascii", "fg_gain": 2.2}
            for e in [0.0, 0.35, 0.55, 0.8, 1.0, 1.3]
        ]
    elif which == "density":
        variants = [
            {"label": f"cols {c}", "cols": c, "charset": "ascii", "fg_gain": 2.2}
            for c in [60, 80, 100, 130, 170, 220]
        ]
    else:
        raise SystemExit(f"unknown sweep '{which}'")

    sheet("running.jpg", base, variants, f"out/sweep_{which}.png")
