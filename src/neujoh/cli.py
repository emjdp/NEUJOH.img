"""Command-line interface for NEUJOH.img."""

from __future__ import annotations

import argparse
import os

from PIL import Image

from . import __version__
from .renderer import CHARSETS, LOOKS, Config, composite, convert, to_ansi, to_svg, to_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neujoh",
        description="Render an image as structure-aware, color-preserving ASCII art.",
    )
    parser.add_argument("image", help="source image path")
    parser.add_argument("-o", "--out", default="out/ascii", help="output path without an extension")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--cols", type=int, default=Config.cols)
    parser.add_argument("--cell", default="12x17", help="cell size WxH in px")
    parser.add_argument("--charset", default=Config.charset, choices=list(CHARSETS))
    parser.add_argument("--look", default=Config.look, choices=list(LOOKS))
    parser.add_argument("--font", default=Config.font, help="custom TrueType/OpenType font path")
    parser.add_argument("--aspect", default=None, help='e.g. "1:1", "16:9"')
    parser.add_argument("--zoom", type=float, default=Config.zoom)
    parser.add_argument("--focus-y", type=float, default=Config.focus_y)
    parser.add_argument("--detail", type=float, default=Config.detail)
    parser.add_argument("--structure", type=float, default=Config.structure)
    parser.add_argument("--edges", type=float, default=Config.edges)
    parser.add_argument("--bloom", type=float, default=Config.bloom)
    parser.add_argument("--bg-gain", type=float, default=Config.bg_gain)
    parser.add_argument("--fg-gain", type=float, default=Config.fg_gain)
    parser.add_argument("--energy", type=float, default=Config.energy)
    parser.add_argument("--exposure", type=float, default=Config.exposure)
    parser.add_argument("--jitter", type=float, default=Config.jitter)
    parser.add_argument("--saturation", type=float, default=Config.saturation)
    parser.add_argument("--subject-boost", type=float, default=Config.subject_boost)
    parser.add_argument("--no-matte", action="store_true")
    parser.add_argument("--matte-model", default=Config.matte_model)
    parser.add_argument("--invert", action="store_true", help="dark glyphs on paper")
    parser.add_argument("--scale", type=int, default=1, help="upscale the PNG output")
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--ansi", action="store_true")
    return parser


def _cell_size(value: str, parser: argparse.ArgumentParser) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
        if width < 1 or height < 1:
            raise ValueError
        return width, height
    except ValueError:
        parser.error("--cell must be two positive integers formatted as WxH")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cell_w, cell_h = _cell_size(args.cell, parser)

    cfg = Config(
        cols=args.cols,
        cell_w=cell_w,
        cell_h=cell_h,
        charset=args.charset,
        font=args.font,
        look=args.look,
        aspect=args.aspect,
        zoom=args.zoom,
        focus_y=args.focus_y,
        detail=args.detail,
        structure=args.structure,
        edges=args.edges,
        bloom=args.bloom,
        bg_gain=args.bg_gain,
        fg_gain=args.fg_gain,
        energy=args.energy,
        exposure=args.exposure,
        jitter=args.jitter,
        saturation=args.saturation,
        subject_boost=args.subject_boost,
        matte=not args.no_matte,
        matte_model=args.matte_model,
        matte_cache=os.path.splitext(args.image)[0] + ".matte.png",
        invert=args.invert,
    )

    with Image.open(args.image) as source:
        src = source.convert("RGB")
    print(f"→ {args.image} {src.width}x{src.height}  look={cfg.look} charset={cfg.charset}")
    result = convert(src, cfg)
    print(f"  · grid {result['cols']}x{result['rows']} chars")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    image = composite(result, cfg, scale=args.scale)
    image.save(f"{args.out}.png")
    print(f"  ✓ {args.out}.png  {image.width}x{image.height}")

    with open(f"{args.out}.txt", "w", encoding="utf-8") as output:
        output.write(to_text(result))
    if args.svg:
        with open(f"{args.out}.svg", "w", encoding="utf-8") as output:
            output.write(to_svg(result, cfg))
        print(f"  ✓ {args.out}.svg")
    if args.ansi:
        with open(f"{args.out}.ans", "w", encoding="utf-8") as output:
            output.write(to_ansi(result, cfg))
        print(f"  ✓ {args.out}.ans")


if __name__ == "__main__":
    main()
