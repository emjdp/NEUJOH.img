#!/usr/bin/env python3
r"""
Structure-aware, colour-preserving ASCII renderer.

The interesting parts, in order:

  1. glyph atlas       every candidate character is rasterised once and reduced
                       to a 4x8 "ink coverage" vector, so matching compares
                       *shape*, not just average brightness
  2. subject matte     optional BiRefNet/rembg alpha, used to grade and detail
                       the person and the background differently
  3. detail channel    CLAHE on luminance drives character choice while the
                       original luminance drives colour, so black clothing
                       still shows structure without turning grey
  4. edge pass         extended difference-of-gaussians + Sobel orientation
                       overrides cells that sit on a strong edge with a
                       directional glyph (| / - \) -- silhouettes read crisply
  5. colour            per-cell averaging in linear light, graded in OKLab
                       (filmic L curve + chroma boost + split tone)
  6. compositing       each cell is a dim colour wash with the glyph painted
                       brighter on top, then an optional bloom pass
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# character sets
# --------------------------------------------------------------------------

CHARSETS = {
    # dense, well graded ramp -- the default workhorse
    "ascii": (
        " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    ),
    # looks like source code from a distance
    "code": (
        " .,:;-_=+*!?/\\|()[]{}<>~^\"'`abcdefgijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&@"
    ),
    # unicode blocks: highest fidelity, least "ASCII"
    "blocks": " ▁▂▃▄▅▆▇█░▒▓▖▗▘▝▚▞▌▐▀▄",
    # ascii ramp + a few blocks to reach true white
    "mixed": (
        " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        "░▒▓█"
    ),
    # sparse and airy -- good for very large grids
    "minimal": " .:-=+*#%@",
}

EDGE_CHARS = ["-", "/", "|", "\\"]

# --------------------------------------------------------------------------
# colour: sRGB <-> linear <-> OKLab
# --------------------------------------------------------------------------


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_LMS_M = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LAB_M = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_LMS_M_INV = np.linalg.inv(_LMS_M)
_LAB_M_INV = np.linalg.inv(_LAB_M)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ _LMS_M.T
    lms = np.cbrt(np.maximum(lms, 0.0))
    return lms @ _LAB_M.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = lab @ _LAB_M_INV.T
    lms = lms**3
    return lms @ _LMS_M_INV.T


# --------------------------------------------------------------------------
# grading looks
# --------------------------------------------------------------------------


@dataclass
class Look:
    """A colour grade expressed in OKLab."""

    gamma: float = 1.0  # <1 lifts midtones, >1 deepens them
    lift: float = 0.0  # raises the black point
    gain: float = 1.0  # scales lightness after the curve
    contrast: float = 1.0  # S-curve strength around the pivot
    pivot: float = 0.5
    chroma: float = 1.0  # saturation multiplier
    chroma_knee: float = 0.18  # chroma above this rolls off instead of clipping
    shadow_tint: tuple = (0.0, 0.0)  # (a, b) pushed into the shadows
    highlight_tint: tuple = (0.0, 0.0)  # (a, b) pushed into the highlights


LOOKS = {
    "neutral": Look(),
    # soft lifted blacks, teal shadows, warm highlights -- the classic
    # "film emulation" feel
    "film": Look(
        gamma=0.92,
        lift=0.045,
        contrast=1.18,
        chroma=1.28,
        shadow_tint=(-0.012, -0.020),
        highlight_tint=(0.010, 0.018),
    ),
    # warm golden-hour push, good for outdoor daylight frames
    "sunlit": Look(
        gamma=0.88,
        lift=0.030,
        contrast=1.22,
        chroma=1.42,
        shadow_tint=(-0.008, -0.026),
        highlight_tint=(0.016, 0.030),
    ),
    # high chroma, cool shadows, magenta bias -- synthwave terminal
    "neon": Look(
        gamma=0.95,
        lift=0.060,
        contrast=1.30,
        chroma=1.85,
        shadow_tint=(0.020, -0.050),
        highlight_tint=(0.014, 0.020),
    ),
    # cold, clean, editorial
    "cold": Look(
        gamma=0.94,
        lift=0.035,
        contrast=1.20,
        chroma=1.15,
        shadow_tint=(-0.018, -0.030),
        highlight_tint=(-0.006, -0.010),
    ),
    "mono": Look(gamma=0.95, lift=0.04, contrast=1.25, chroma=0.10),
}


def apply_look(rgb_lin: np.ndarray, look: Look) -> np.ndarray:
    """Grade linear-light RGB (..., 3) through a Look, in OKLab."""
    lab = linear_to_oklab(rgb_lin)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    L = np.clip(L, 0.0, 1.0) ** look.gamma
    # smooth S-curve around the pivot
    if look.contrast != 1.0:
        d = L - look.pivot
        L = look.pivot + np.sign(d) * (np.abs(d) ** (1.0 / look.contrast)) * (
            look.pivot ** (1.0 - 1.0 / look.contrast)
        )
    L = np.clip(look.lift + L * look.gain * (1.0 - look.lift), 0.0, 1.0)

    C = np.hypot(a, b)
    h = np.arctan2(b, a)
    C = C * look.chroma
    # soft knee so boosted saturation compresses instead of clipping hard
    k = look.chroma_knee
    C = np.where(C > k, k + (C - k) / (1.0 + (C - k) / k), C)
    a, b = C * np.cos(h), C * np.sin(h)

    w_lo = (1.0 - L) ** 2
    w_hi = L**2
    a = a + look.shadow_tint[0] * w_lo + look.highlight_tint[0] * w_hi
    b = b + look.shadow_tint[1] * w_lo + look.highlight_tint[1] * w_hi

    out = oklab_to_linear(np.stack([L, a, b], axis=-1))
    return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------
# glyph atlas
# --------------------------------------------------------------------------


@dataclass
class GlyphAtlas:
    chars: list
    masks: np.ndarray  # (n, ch, cw) float 0..1 ink coverage
    feats: np.ndarray  # (n, fy*fx) coverage vectors used for matching
    coverage: np.ndarray  # (n,) mean ink
    cw: int
    ch: int
    index: dict = field(default_factory=dict)
    tree: cKDTree | None = None


def build_atlas(
    font_path: str,
    charset: str,
    cw: int,
    ch: int,
    feat_cols: int = 4,
    feat_rows: int = 8,
    supersample: int = 4,
    lum_weight: float = 1.6,
) -> GlyphAtlas:
    """Rasterise every character and reduce it to a coverage vector."""
    chars = list(dict.fromkeys(charset))  # de-dupe, keep order

    # pick the pixel size whose advance width matches the requested cell width
    size = cw * supersample
    font = ImageFont.truetype(font_path, size)
    adv = font.getlength("M")
    if adv > 0:
        size = max(4, int(round(size * (cw * supersample) / adv)))
        font = ImageFont.truetype(font_path, size)

    W, H = cw * supersample, ch * supersample
    ascent, descent = font.getmetrics()
    # centre the em box vertically inside the cell
    baseline = (H - (ascent + descent)) // 2 + ascent

    masks = np.zeros((len(chars), ch, cw), dtype=np.float32)
    for i, chr_ in enumerate(chars):
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        try:
            d.text((0, baseline), chr_, font=font, fill=255, anchor="ls")
        except Exception:
            continue
        small = np.asarray(
            img.resize((cw, ch), Image.Resampling.LANCZOS), dtype=np.float32
        )
        masks[i] = np.clip(small / 255.0, 0.0, 1.0)

    # coverage vector: mean ink over a feat_rows x feat_cols grid
    fy, fx = feat_rows, feat_cols
    ry, rx = ch // fy, cw // fx
    crop = masks[:, : ry * fy, : rx * fx]
    feats = crop.reshape(len(chars), fy, ry, fx, rx).mean(axis=(2, 4))
    feats = feats.reshape(len(chars), fy * fx)

    coverage = masks.reshape(len(chars), -1).mean(axis=1)
    # append overall brightness so the match also respects global tone
    feats_w = np.concatenate(
        [feats, (coverage * lum_weight * math.sqrt(fy * fx))[:, None]], axis=1
    )

    atlas = GlyphAtlas(
        chars=chars,
        masks=masks,
        feats=feats_w,
        coverage=coverage,
        cw=cw,
        ch=ch,
        index={c: i for i, c in enumerate(chars)},
    )
    atlas.tree = cKDTree(feats_w)
    atlas._feat_shape = (fy, fx)  # type: ignore[attr-defined]
    atlas._lum_weight = lum_weight  # type: ignore[attr-defined]
    return atlas


# --------------------------------------------------------------------------
# subject matte
# --------------------------------------------------------------------------


def subject_matte(
    img: Image.Image, model: str = "birefnet-general", cache: str | None = None
) -> np.ndarray | None:
    """Alpha matte of the main subject, or None if rembg is unavailable.

    Inference costs ~30s on CPU, so the result is cached next to the source
    image -- parameter sweeps then cost nothing.
    """
    if cache and os.path.exists(cache):
        m = np.asarray(Image.open(cache).convert("L"), dtype=np.float32) / 255.0
        if m.shape == (img.height, img.width):
            print(f"  · matte from cache {cache}")
            return m

    try:
        from rembg import new_session, remove
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  ! rembg unavailable ({exc}); continuing without a matte")
        return None
    try:
        session = new_session(model)
    except Exception as exc:
        print(f"  ! model '{model}' unavailable ({exc}); falling back to u2net")
        session = new_session("u2net")
    out = remove(img.convert("RGB"), session=session, only_mask=True)
    if cache:
        out.save(cache)
    return np.asarray(out, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------
# main conversion
# --------------------------------------------------------------------------


@dataclass
class Config:
    cols: int = 150
    cell_w: int = 12
    cell_h: int = 17
    charset: str = "ascii"
    font: str = "fonts/JetBrainsMono-Bold.ttf"
    look: str = "film"
    aspect: str | None = None  # "1:1", "16:9", ...
    zoom: float = 1.0  # >1 tightens the crop around the subject
    focus_y: float = 0.0  # nudge the crop centre up (-) or down (+)
    detail: float = 0.55  # CLAHE strength on the character channel
    local_contrast: float = 0.6  # unsharp on the character channel
    structure: float = 0.8  # local contrast normalisation of the char channel
    structure_bg: float = 0.3  # how much of that reaches the background
    lum_weight: float = 0.45  # tone vs shape in the glyph match; high = repetitive
    jitter: float = 0.05  # breaks up glyph repetition in flat areas
    edges: float = 0.55  # 0 disables the directional edge pass
    edge_sigma: float = 1.1
    edge_bg: float = 0.55  # how much background edges are suppressed
    silhouette: float = 0.9  # weight of the matte outline in the edge pass
    bg_smooth: float = 0.7  # calms background texture into tonal fields
    bg_gain: float = 0.30  # how much colour the cell wash keeps
    fg_gain: float = 1.4  # glyph brightness when energy solving is off
    energy: float = 0.8  # 0 = flat scaling, 1 = full energy compensation
    exposure: float = 1.35  # overall brightness target
    ink_floor: float = 0.22  # glyphs never go fully black -> shadow detail
    ink_gamma: float = 1.6  # how fast the floor fades out of the highlights
    bloom: float = 0.28
    bloom_radius: float = 9.0
    matte: bool = True
    matte_model: str = "birefnet-general"
    saturation: float = 0.78  # scales the look's chroma
    subject_boost: float = 1.0  # extra chroma/detail on the subject
    bg_recede: float = 0.82  # background chroma multiplier
    invert: bool = False  # dark glyphs on a light wash
    matte_cache: str | None = None
    seed: int = 0


def smart_crop(
    img: Image.Image,
    aspect: str | None,
    matte: np.ndarray | None,
    zoom: float = 1.0,
    focus_y: float = 0.0,
):
    """Crop to `aspect`, centred on the subject, tightened by `zoom`."""
    if not aspect and zoom <= 1.0:
        return img, matte
    w, h = img.size
    target = (
        (lambda a, b: a / b)(*(float(v) for v in aspect.replace(":", "/").split("/")))
        if aspect
        else w / h
    )

    if matte is not None and matte.size and (matte > 0.5).any():
        ys, xs = np.nonzero(matte > 0.5)
        cx, cy = float(xs.mean()) / w, float(ys.mean()) / h
    else:
        cx, cy = 0.5, 0.45
    cy = float(np.clip(cy + focus_y, 0.0, 1.0))

    if w / h > target:  # too wide -> trim the sides
        nw, nh = h * target, float(h)
    else:  # too tall -> trim top/bottom
        nw, nh = float(w), w / target
    nw, nh = nw / max(zoom, 1e-3), nh / max(zoom, 1e-3)
    nw, nh = int(round(min(nw, w))), int(round(min(nh, h)))

    x0 = int(round(np.clip(cx * w - nw / 2, 0, w - nw)))
    y0 = int(round(np.clip(cy * h - nh / 2, 0, h - nh)))
    return img.crop((x0, y0, x0 + nw, y0 + nh)), (
        matte[y0 : y0 + nh, x0 : x0 + nw] if matte is not None else None
    )


def cell_reduce(arr: np.ndarray, rows: int, cols: int, fy: int, fx: int) -> np.ndarray:
    """Mean-pool arr (rows*fy, cols*fx) -> (rows, cols, fy*fx)."""
    a = arr[: rows * fy, : cols * fx]
    a = a.reshape(rows, fy, cols, fx).transpose(0, 2, 1, 3)
    return a.reshape(rows, cols, fy * fx)


def convert(src: Image.Image, cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    atlas = build_atlas(
        cfg.font, CHARSETS[cfg.charset], cfg.cell_w, cfg.cell_h, lum_weight=cfg.lum_weight
    )
    fy, fx = atlas._feat_shape  # type: ignore[attr-defined]

    # ---- matte ---------------------------------------------------------
    matte = None
    if cfg.matte:
        matte = subject_matte(src, cfg.matte_model, cfg.matte_cache)

    src, matte = smart_crop(src, cfg.aspect, matte, cfg.zoom, cfg.focus_y)

    # ---- resample to the character grid --------------------------------
    cols = cfg.cols
    cell_ar = cfg.cell_h / cfg.cell_w
    rows = max(1, int(round(src.height / src.width * cols / cell_ar)))

    # sample each cell at fx x fy so structure survives the downscale
    sw, sh = cols * fx, rows * fy
    small = src.convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    lin = srgb_to_linear(rgb)

    if matte is not None:
        m_small = cv2.resize(matte, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        m_small = np.zeros((sh, sw), dtype=np.float32)

    # ---- character channel ---------------------------------------------
    # perceptual lightness, then CLAHE so dark regions keep their structure
    lum = linear_to_oklab(lin)[..., 0].astype(np.float32)
    det = lum.copy()

    # unsharp: puts back the fine texture the downscale to the grid ate
    if cfg.local_contrast > 0:
        blur = cv2.GaussianBlur(det, (0, 0), 6.0)
        det = np.clip(det + cfg.local_contrast * (det - blur), 0.0, 1.0)

    # CLAHE for mid-scale structure
    if cfg.detail > 0:
        eq = (
            cv2.createCLAHE(clipLimit=1.0 + 3.0 * cfg.detail, tileGridSize=(8, 8))
            .apply(np.clip(det * 255, 0, 255).astype(np.uint8))
            .astype(np.float32)
            / 255.0
        )
        det = det * (1 - cfg.detail) + eq * cfg.detail

    # Local contrast normalisation: re-centre every neighbourhood on mid
    # density using its own mean and spread. This is what decouples character
    # density from brightness -- black clothing comes out *full* of characters
    # (which the colour layer then paints dark) instead of as an empty void.
    # The variance floor stops flat regions like sky from amplifying noise.
    if cfg.structure > 0:
        sig = max(3.0, min(sw, sh) / 22.0)
        mu = cv2.GaussianBlur(det, (0, 0), sig)
        var = cv2.GaussianBlur((det - mu) ** 2, (0, 0), sig)
        sd = np.sqrt(np.maximum(var, 1e-8))
        loc = np.clip(0.5 + (det - mu) / np.maximum(sd * 3.0, 0.055), 0.0, 1.0)
        # applied hard on the subject, gently on the background (whose
        # global tone already reads well)
        w = cfg.structure * (cfg.structure_bg + (1 - cfg.structure_bg) * m_small)
        det = det * (1 - w) + loc * w

    # the background is mostly foliage and gravel: left alone it turns into a
    # field of random glyphs. Smoothing it there keeps it as tonal shape while
    # the subject stays crisp.
    det = np.ascontiguousarray(det, dtype=np.float32)
    if cfg.bg_smooth > 0:
        smooth = cv2.bilateralFilter(det, 9, 0.10, 7)
        w_bg = (1.0 - m_small) * cfg.bg_smooth
        det = det * (1 - w_bg) + smooth * w_bg

    # background recedes slightly so the subject reads first
    det = det * (cfg.bg_recede + (1 - cfg.bg_recede) * (0.5 + 0.5 * m_small))

    if cfg.invert:
        det = 1.0 - det

    # map the detail range onto the charset's achievable coverage range
    lo, hi = float(atlas.coverage.min()), float(atlas.coverage.max())
    p1, p99 = np.percentile(det, [1.0, 99.0])
    det = np.clip((det - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
    det = lo + det * (hi - lo)

    # ---- match glyphs ---------------------------------------------------
    target = cell_reduce(det, rows, cols, fy, fx)
    mean_ink = target.mean(axis=2, keepdims=True)
    lw = atlas._lum_weight  # type: ignore[attr-defined]
    query = np.concatenate(
        [target.reshape(-1, fy * fx), (mean_ink.reshape(-1, 1) * lw * math.sqrt(fy * fx))],
        axis=1,
    )
    # a little noise before the lookup: in flat regions many glyphs are nearly
    # equidistant, and always taking the same winner produces visible bands of
    # repeated characters. Jitter picks among the near-ties instead.
    if cfg.jitter > 0:
        query = query + rng.normal(0.0, cfg.jitter, query.shape) * np.array(
            [1.0] * (fy * fx) + [0.35]
        )
    _, idx = atlas.tree.query(query, workers=-1)
    idx = idx.reshape(rows, cols)

    # ---- directional edge pass -----------------------------------------
    if cfg.edges > 0:
        g = cv2.GaussianBlur(lum, (0, 0), cfg.edge_sigma)
        g2 = cv2.GaussianBlur(lum, (0, 0), cfg.edge_sigma * 1.6)
        dog = g - 0.985 * g2  # extended DoG
        mag = np.abs(dog)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        norm = lambda v: v / (np.percentile(np.abs(v), 99.0) + 1e-6)
        gx, gy = norm(gx), norm(gy)

        # the matte outline is a guaranteed, noise-free edge: folding it in
        # makes the runner's silhouette read even where luminance contrast
        # against the background is weak
        if cfg.silhouette > 0 and m_small.any():
            ms = cv2.GaussianBlur(m_small, (0, 0), 1.2)
            sx = norm(cv2.Sobel(ms, cv2.CV_32F, 1, 0, ksize=3))
            sy = norm(cv2.Sobel(ms, cv2.CV_32F, 0, 1, ksize=3))
            gx = gx + sx * cfg.silhouette
            gy = gy + sy * cfg.silhouette
            mag = np.maximum(mag / (np.percentile(mag, 99) + 1e-6),
                             np.hypot(sx, sy) * cfg.silhouette)
        grad = np.hypot(gx, gy)

        # orientation of the *edge* is perpendicular to the gradient;
        # average with doubled angles so opposite directions agree
        ang2 = 2.0 * np.arctan2(gy, gx)
        wsum = cell_reduce(grad, rows, cols, fy, fx).sum(axis=2)
        cs = cell_reduce(grad * np.cos(ang2), rows, cols, fy, fx).sum(axis=2)
        sn = cell_reduce(grad * np.sin(ang2), rows, cols, fy, fx).sum(axis=2)
        theta = 0.5 * np.arctan2(sn, cs) + math.pi / 2  # -> edge direction

        energy = cell_reduce(mag, rows, cols, fy, fx).mean(axis=2)
        coher = np.hypot(cs, sn) / np.maximum(wsum, 1e-6)  # 1 = one clean direction

        # damp background edges so gravel and foliage stop drawing dashes
        cm = cell_reduce(m_small, rows, cols, fy, fx).mean(axis=2)
        energy = energy * (cfg.edge_bg + (1.0 - cfg.edge_bg) * cm)

        thr = np.percentile(energy, 100 - 34 * cfg.edges)
        strong = (energy >= thr) & (coher > 0.5)

        bin_ = np.mod(np.round(theta / (math.pi / 4)).astype(int), 4)
        edge_idx = np.array(
            [atlas.index.get(c, atlas.index[" "]) for c in EDGE_CHARS]
        )
        # 0 = horizontal edge, 1 = "/", 2 = vertical, 3 = "\"
        idx = np.where(strong, edge_idx[bin_], idx)

    # ---- colour ---------------------------------------------------------
    cell_rgb = np.stack(
        [cell_reduce(lin[..., c], rows, cols, fy, fx).mean(2) for c in range(3)], axis=-1
    )
    cell_m = cell_reduce(m_small, rows, cols, fy, fx).mean(2)

    look = LOOKS[cfg.look]
    if cfg.saturation != 1.0:
        look = replace(look, chroma=look.chroma * cfg.saturation)
    col = apply_look(cell_rgb, look)

    # subject a touch richer, background a touch calmer -> depth
    lab = linear_to_oklab(col)
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.arctan2(lab[..., 2], lab[..., 1])
    scale = cfg.bg_recede + (cfg.subject_boost - cfg.bg_recede) * cell_m
    C = C * scale
    lab[..., 1], lab[..., 2] = C * np.cos(h), C * np.sin(h)
    col = np.clip(oklab_to_linear(lab), 0.0, None)

    chars = np.array(atlas.chars)[idx]
    return {
        "idx": idx,
        "chars": chars,
        "color": col,
        "atlas": atlas,
        "rows": rows,
        "cols": cols,
        "matte": cell_m,
    }


# --------------------------------------------------------------------------
# compositing
# --------------------------------------------------------------------------


def cell_colors(res: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (ink, wash) in linear light.

    A glyph only inks a fraction of its cell -- even '@' covers under half of
    it. Painting the glyph at the cell's own colour therefore lands the cell
    average far below the photo, which is why naive colour ASCII always comes
    out muddy. So instead solve for the ink that satisfies

        coverage * ink + (1 - coverage) * wash  =  cell colour

    and each cell carries the right amount of light no matter which character
    the structure matcher happened to pick.
    """
    col = res["color"]
    cov = res["atlas"].coverage[res["idx"]][..., None]

    if cfg.invert:
        paper = srgb_to_linear(np.array([0.96, 0.955, 0.94], dtype=np.float32))
        bg = paper * (1.0 - cfg.bg_gain) + col * cfg.bg_gain
    else:
        bg = col * cfg.bg_gain

    solved = (col * cfg.exposure - (1.0 - cov) * bg) / np.maximum(cov, 0.04)
    flat = col * (0.45 if cfg.invert else cfg.fg_gain)
    ink = flat * (1.0 - cfg.energy) + solved * cfg.energy

    lab = linear_to_oklab(np.clip(ink, 0.0, None))
    L = np.maximum(lab[..., 0], 0.0)
    tl = np.clip(linear_to_oklab(col)[..., 0], 0.0, 1.0)
    if cfg.invert:
        # the solve already drives dark cells to black ink; all that is needed
        # is a ceiling so ink in the highlights stays darker than the paper and
        # the characters remain readable
        L = np.minimum(np.clip(L, 0.0, 1.0), 1.0 - cfg.ink_floor * tl**cfg.ink_gamma)
    else:
        # soft shoulder rather than a hard clip, so full compensation in bright
        # cells rolls off instead of blowing out to flat white
        k = 0.78
        L = np.where(L <= k, L, k + (1 - k) * (1.0 - np.exp(-(L - k) / (1 - k))))
        # and a floor so shadow structure never vanishes entirely
        L = np.maximum(L, cfg.ink_floor * (1.0 - tl) ** cfg.ink_gamma)

    fg = np.clip(oklab_to_linear(np.stack([L, lab[..., 1], lab[..., 2]], -1)), 0.0, None)
    return fg, bg


def composite(res: dict, cfg: Config, scale: int = 1) -> Image.Image:
    atlas: GlyphAtlas = res["atlas"]
    rows, cols = res["rows"], res["cols"]
    cw, ch = atlas.cw * scale, atlas.ch * scale
    masks = atlas.masks
    if scale != 1:
        masks = np.clip(
            np.stack(
                [
                    cv2.resize(m, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
                    for m in atlas.masks
                ]
            ),
            0,
            1,
        )

    idx = res["idx"]
    fg, bg = cell_colors(res, cfg)
    canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.float32)

    for r in range(rows):
        y0 = r * ch
        for c in range(cols):
            m = masks[idx[r, c]][..., None]
            x0 = c * cw
            canvas[y0 : y0 + ch, x0 : x0 + cw] = bg[r, c] * (1 - m) + fg[r, c] * m

    if cfg.bloom > 0 and not cfg.invert:
        thr = float(np.percentile(canvas, 88))
        bright = np.clip(canvas - thr, 0, None)
        blur = cv2.GaussianBlur(bright, (0, 0), cfg.bloom_radius * scale)
        canvas = canvas + blur * cfg.bloom

    out = (linear_to_srgb(canvas) * 255.0).round().astype(np.uint8)
    return Image.fromarray(out, "RGB")


def to_text(res: dict) -> str:
    return "\n".join("".join(row) for row in res["chars"])


def to_ansi(res: dict, cfg: Config) -> str:
    fg, _ = cell_colors(res, cfg)
    srgb = np.clip(linear_to_srgb(fg) * 255, 0, 255).astype(int)
    lines = []
    for r, row in enumerate(res["chars"]):
        buf, last = [], None
        for c, chr_ in enumerate(row):
            rgb = tuple(srgb[r, c])
            if rgb != last:
                buf.append("\033[38;2;%d;%d;%dm" % rgb)
                last = rgb
            buf.append(chr_)
        lines.append("".join(buf) + "\033[0m")
    return "\n".join(lines)


SVG_FONTS = (
    "'JetBrains Mono','DejaVu Sans Mono','SFMono-Regular',Menlo,Consolas,monospace"
)


def to_svg(res: dict, cfg: Config, font_family: str = SVG_FONTS) -> str:
    """Resolution-independent version of the same image.

    One <text> per row with per-character <tspan> fills, and textLength pinned
    to the row width so the grid stays exact whichever monospace font the
    viewer actually resolves. Setting textLength per *character* instead would
    stretch every glyph to the cell width and mangle them.
    """
    atlas: GlyphAtlas = res["atlas"]
    rows, cols = res["rows"], res["cols"]
    cw, ch = atlas.cw, atlas.ch
    W, H = cols * cw, rows * ch
    fg, bg = cell_colors(res, cfg)
    srgb = np.clip(linear_to_srgb(fg) * 255, 0, 255).astype(int)
    bgc = np.clip(linear_to_srgb(bg) * 255, 0, 255).astype(int)

    esc = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="#000"/>',
        '<g shape-rendering="crispEdges">',
    ]
    # colour wash: merge horizontal runs of equal colour into one rect
    for r in range(rows):
        c = 0
        while c < cols:
            c2 = c
            while c2 + 1 < cols and (bgc[r, c2 + 1] == bgc[r, c]).all():
                c2 += 1
            parts.append(
                '<rect x="%d" y="%d" width="%d" height="%d" fill="#%02x%02x%02x"/>'
                % (c * cw, r * ch, (c2 - c + 1) * cw, ch, *bgc[r, c])
            )
            c = c2 + 1
    parts.append("</g>")

    # a monospace advance is ~0.6em, so this font-size makes one glyph one cell
    parts.append(
        f'<g font-family="{font_family}" font-size="{cw / 0.6:.2f}" '
        f'xml:space="preserve">'
    )
    for r in range(rows):
        row = res["chars"][r]
        if all(c == " " for c in row):
            continue
        spans = []
        c = 0
        while c < cols:
            c2 = c
            while c2 + 1 < cols and (srgb[r, c2 + 1] == srgb[r, c]).all():
                c2 += 1
            text = "".join(esc.get(ch_, ch_) for ch_ in row[c : c2 + 1])
            spans.append('<tspan fill="#%02x%02x%02x">%s</tspan>' % (*srgb[r, c], text))
            c = c2 + 1
        parts.append(
            '<text x="0" y="%.2f" textLength="%d" lengthAdjust="spacing">%s</text>'
            % (r * ch + ch * 0.76, W, "".join(spans))
        )
    parts.append("</g></svg>")
    return "".join(parts)


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("-o", "--out", default="out/ascii")
    p.add_argument("--cols", type=int, default=Config.cols)
    p.add_argument("--cell", default="12x17", help="cell size WxH in px")
    p.add_argument("--charset", default=Config.charset, choices=list(CHARSETS))
    p.add_argument("--look", default=Config.look, choices=list(LOOKS))
    p.add_argument("--font", default=Config.font)
    p.add_argument("--aspect", default=None, help='e.g. "1:1", "16:9"')
    p.add_argument("--zoom", type=float, default=Config.zoom)
    p.add_argument("--focus-y", type=float, default=Config.focus_y)
    p.add_argument("--detail", type=float, default=Config.detail)
    p.add_argument("--structure", type=float, default=Config.structure)
    p.add_argument("--edges", type=float, default=Config.edges)
    p.add_argument("--bloom", type=float, default=Config.bloom)
    p.add_argument("--bg-gain", type=float, default=Config.bg_gain)
    p.add_argument("--fg-gain", type=float, default=Config.fg_gain)
    p.add_argument("--energy", type=float, default=Config.energy)
    p.add_argument("--exposure", type=float, default=Config.exposure)
    p.add_argument("--jitter", type=float, default=Config.jitter)
    p.add_argument("--saturation", type=float, default=Config.saturation)
    p.add_argument("--subject-boost", type=float, default=Config.subject_boost)
    p.add_argument("--no-matte", action="store_true")
    p.add_argument("--matte-model", default=Config.matte_model)
    p.add_argument("--invert", action="store_true", help="dark glyphs on paper")
    p.add_argument("--scale", type=int, default=1, help="upscale the PNG output")
    p.add_argument("--svg", action="store_true")
    p.add_argument("--ansi", action="store_true")
    args = p.parse_args()

    cw, chh = (int(v) for v in args.cell.lower().split("x"))
    cfg = Config(
        cols=args.cols,
        cell_w=cw,
        cell_h=chh,
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

    src = Image.open(args.image)
    print(f"→ {args.image} {src.size[0]}x{src.size[1]}  look={cfg.look} charset={cfg.charset}")
    res = convert(src, cfg)
    print(f"  · grid {res['cols']}x{res['rows']} chars")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img = composite(res, cfg, scale=args.scale)
    img.save(f"{args.out}.png")
    print(f"  ✓ {args.out}.png  {img.size[0]}x{img.size[1]}")

    with open(f"{args.out}.txt", "w") as f:
        f.write(to_text(res))
    if args.svg:
        with open(f"{args.out}.svg", "w") as f:
            f.write(to_svg(res, cfg))
        print(f"  ✓ {args.out}.svg")
    if args.ansi:
        with open(f"{args.out}.ans", "w") as f:
            f.write(to_ansi(res, cfg))


if __name__ == "__main__":
    main()
