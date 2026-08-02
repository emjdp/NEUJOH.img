"""Public Python API for NEUJOH.img."""

from .renderer import (
    CHARSETS,
    DEFAULT_FONT,
    LOOKS,
    Config,
    GlyphAtlas,
    Look,
    apply_look,
    build_atlas,
    cell_colors,
    composite,
    convert,
    smart_crop,
    subject_matte,
    to_ansi,
    to_svg,
    to_text,
)

__version__ = "0.1.0"

__all__ = [
    "CHARSETS",
    "DEFAULT_FONT",
    "LOOKS",
    "Config",
    "GlyphAtlas",
    "Look",
    "apply_look",
    "build_atlas",
    "cell_colors",
    "composite",
    "convert",
    "smart_crop",
    "subject_matte",
    "to_ansi",
    "to_svg",
    "to_text",
    "__version__",
]
