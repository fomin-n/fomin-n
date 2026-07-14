#!/usr/bin/env python3
"""Generate a fixed-width, edge-aware ASCII portrait from a photograph."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "portrait.txt"
DEFAULT_CROP = (720, 650, 2120, 2200)
DEFAULT_RAMP = " .:-=+*#%@"
FONT_STACK = "SFMono-Regular, 'SF Mono', Menlo, Monaco, Consolas, 'Liberation Mono', monospace"


@dataclass(frozen=True)
class Settings:
    crop: tuple[int, int, int, int]
    width: int
    height: int
    contrast: float
    gamma: float
    edge_strength: float
    edge_threshold: float
    background_threshold: float
    ramp: str


def parse_crop(value: str) -> tuple[int, int, int, int]:
    try:
        crop = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must contain four pixel integers") from exc
    if len(crop) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    left, top, right, bottom = crop
    if min(crop) < 0 or left >= right or top >= bottom:
        raise argparse.ArgumentTypeError("crop coordinates must be non-negative and ordered")
    return left, top, right, bottom


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def clahe_like(gray: Image.Image, tiles: int = 8, clip_limit: float = 2.2) -> Image.Image:
    """Apply deterministic tiled histogram equalization with clipped histograms."""

    source = np.asarray(gray, dtype=np.uint8)
    height, width = source.shape
    y_edges = np.linspace(0, height, tiles + 1, dtype=int)
    x_edges = np.linspace(0, width, tiles + 1, dtype=int)
    luts = np.empty((tiles, tiles, 256), dtype=np.float32)

    for tile_y in range(tiles):
        for tile_x in range(tiles):
            tile = source[
                y_edges[tile_y] : y_edges[tile_y + 1],
                x_edges[tile_x] : x_edges[tile_x + 1],
            ]
            histogram = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
            limit = max(1.0, clip_limit * tile.size / 256.0)
            excess = np.maximum(histogram - limit, 0.0).sum()
            histogram = np.minimum(histogram, limit) + excess / 256.0
            cumulative = np.cumsum(histogram)
            luts[tile_y, tile_x] = 255.0 * cumulative / cumulative[-1]

    tile_x = (np.arange(width) + 0.5) * tiles / width - 0.5
    tile_y = (np.arange(height) + 0.5) * tiles / height - 0.5
    x0 = np.clip(np.floor(tile_x).astype(int), 0, tiles - 1)
    y0 = np.clip(np.floor(tile_y).astype(int), 0, tiles - 1)
    x1 = np.minimum(x0 + 1, tiles - 1)
    y1 = np.minimum(y0 + 1, tiles - 1)
    wx = np.clip(tile_x - np.floor(tile_x), 0.0, 1.0)[None, :]
    wy = np.clip(tile_y - np.floor(tile_y), 0.0, 1.0)[:, None]
    pixels = source

    top = (
        luts[y0[:, None], x0[None, :], pixels] * (1.0 - wx)
        + luts[y0[:, None], x1[None, :], pixels] * wx
    )
    bottom = (
        luts[y1[:, None], x0[None, :], pixels] * (1.0 - wx)
        + luts[y1[:, None], x1[None, :], pixels] * wx
    )
    result = top * (1.0 - wy) + bottom * wy
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "L")


def subject_matte(width: int, height: int) -> np.ndarray:
    """Create a soft head-and-upper-neck matte without retaining the torso."""

    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)

    head_distance = ((x - 0.50) / 0.455) ** 2 + ((y - 0.49) / 0.54) ** 2
    head = smoothstep((1.04 - head_distance) / 0.10)

    neck_half_width = 0.23 - 0.04 * np.clip((y - 0.84) / 0.16, 0.0, 1.0)
    neck_x = smoothstep((neck_half_width - np.abs(x - 0.50)) / 0.04)
    neck_y = smoothstep((y - 0.82) / 0.05)
    neck = neck_x * neck_y

    matte = np.maximum(head, neck)
    matte *= smoothstep((1.015 - y) / 0.035)
    return matte.astype(np.float32)


def resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), "F")
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)


def load_source(source: Path) -> Image.Image:
    with Image.open(source) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def prepare_crop(
    source: Path | Image.Image, crop: tuple[int, int, int, int]
) -> Image.Image:
    image = load_source(source) if isinstance(source, Path) else source
    left, top, right, bottom = crop
    if right > image.width or bottom > image.height:
        raise ValueError(f"crop {crop} exceeds source dimensions {image.size}")
    return image.crop(crop)


def portrait_rows(
    source: Path | Image.Image, settings: Settings
) -> tuple[list[str], Image.Image]:
    crop = prepare_crop(source, settings.crop)
    analysis_width = 560
    analysis_height = max(420, round(analysis_width * crop.height / crop.width))
    crop = crop.resize((analysis_width, analysis_height), Image.Resampling.LANCZOS)

    gray = crop.convert("L").filter(ImageFilter.MedianFilter(size=3))
    base = np.asarray(gray, dtype=np.float32)
    illumination = np.asarray(gray.filter(ImageFilter.BoxBlur(radius=42)), dtype=np.float32)
    balanced = np.clip(base - 0.58 * (illumination - 128.0), 0.0, 255.0)
    balanced_image = Image.fromarray(balanced.astype(np.uint8), "L")
    balanced_image = ImageOps.autocontrast(balanced_image, cutoff=(0.4, 0.4))
    local = clahe_like(balanced_image)
    local = Image.blend(balanced_image, local, 0.48)
    local = ImageEnhance.Contrast(local).enhance(settings.contrast)
    pixels = np.asarray(local, dtype=np.float32) / 255.0
    pixels = np.power(np.clip(pixels, 0.0, 1.0), settings.gamma)

    edge_source = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius=0.75)), dtype=np.float32
    ) / 255.0
    grad_y, grad_x = np.gradient(edge_source)
    edge = np.hypot(grad_x, grad_y)
    scale = max(float(np.percentile(edge, 98.5)), 1e-6)
    edge = np.clip(edge / scale, 0.0, 1.0)
    darkness = 1.0 - pixels
    matte = subject_matte(analysis_width, analysis_height)

    size = (settings.width, settings.height)
    small_darkness = np.clip(resize_float(darkness, size), 0.0, 1.0)
    small_edge = np.clip(resize_float(edge, size), 0.0, 1.0)
    small_x = resize_float(grad_x, size)
    small_y = resize_float(grad_y, size)
    small_matte = np.clip(resize_float(matte, size), 0.0, 1.0)

    score = np.clip(
        small_darkness * (1.0 - 0.34 * settings.edge_strength)
        + small_edge * (0.58 * settings.edge_strength),
        0.0,
        1.0,
    )
    ramp = settings.ramp
    if len(ramp) < 3 or ramp[0] != " ":
        raise ValueError("ramp must start with a space and contain at least three characters")

    rows: list[str] = []
    for row_index in range(settings.height):
        characters: list[str] = []
        for column in range(settings.width):
            mask_value = small_matte[row_index, column]
            tone = score[row_index, column]
            edge_value = small_edge[row_index, column]
            relative_y = row_index / max(settings.height - 1, 1)
            local_edge_threshold = settings.edge_threshold
            if 0.44 <= relative_y <= 0.64:
                local_edge_threshold *= 0.72
            if mask_value < 0.36 or tone * mask_value < settings.background_threshold:
                character = " "
            elif edge_value >= local_edge_threshold and (
                small_darkness[row_index, column] >= 0.16
                or edge_value >= min(1.0, local_edge_threshold * 1.55)
            ):
                gx = small_x[row_index, column]
                gy = small_y[row_index, column]
                if abs(gx) > abs(gy) * 1.65:
                    character = "|"
                elif abs(gy) > abs(gx) * 1.65:
                    character = "-"
                elif gx * gy >= 0:
                    character = "/"
                else:
                    character = "\\"
            else:
                normalized = np.clip(
                    (tone - settings.background_threshold)
                    / max(1.0 - settings.background_threshold, 1e-6),
                    0.0,
                    1.0,
                )
                character = ramp[round(normalized * (len(ramp) - 1))]
            characters.append(character)
        rows.append("".join(characters).ljust(settings.width))
    return rows, crop


def candidate_settings() -> list[Settings]:
    crops = [
        (720, 650, 2120, 2200),
        (650, 620, 2200, 2250),
        (700, 700, 2150, 2280),
    ]
    sizes = [(44, 34), (48, 36), (52, 38)]
    variants = [
        dict(contrast=1.12, gamma=0.92, edge_strength=0.74, edge_threshold=0.24, background_threshold=0.13),
        dict(contrast=1.28, gamma=0.82, edge_strength=0.60, edge_threshold=0.19, background_threshold=0.16),
    ]
    return [
        Settings(crop, width, height, ramp=DEFAULT_RAMP, **variant)
        for crop in crops
        for width, height in sizes
        for variant in variants
    ]


def write_rows(rows: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_contact_sheet(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = load_source(source)
    cards: list[str] = []
    manifest: list[dict[str, object]] = []
    for index, settings in enumerate(candidate_settings(), start=1):
        rows, crop = portrait_rows(source_image, settings)
        candidate_id = f"candidate-{index:02d}"
        write_rows(rows, output_dir / f"{candidate_id}.txt")
        if index == 1:
            crop.save(output_dir / "source-crop.png")
        parameters = asdict(settings)
        parameters["crop"] = list(settings.crop)
        manifest.append({"id": candidate_id, **parameters})
        summary = (
            f"{settings.width}×{settings.height} · crop {settings.crop}<br>"
            f"contrast {settings.contrast} · gamma {settings.gamma} · "
            f"edge {settings.edge_strength}/{settings.edge_threshold} · "
            f"background {settings.background_threshold}"
        )
        cards.append(
            f'<article><h2>{candidate_id}</h2><p>{summary}</p>'
            f'<pre>{html.escape(chr(10).join(rows))}</pre></article>'
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ASCII portrait candidates</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #0b0b0b; color: #d0d0d0; font-family: {FONT_STACK}; }}
header {{ display: flex; gap: 28px; align-items: center; padding: 28px; border-bottom: 1px solid #333; }}
header img {{ width: 260px; max-height: 300px; object-fit: cover; object-position: center; }}
h1 {{ color: #f2f2f2; font-size: 24px; }}
main {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; padding: 18px; }}
article {{ min-width: 0; border: 1px solid #333; border-radius: 8px; padding: 14px; background: #000; }}
h2 {{ margin: 0 0 6px; color: #ffd75f; font-size: 16px; }}
p {{ min-height: 48px; margin: 0 0 10px; color: #858585; font: 11px/1.35 {FONT_STACK}; }}
pre {{ margin: 0; overflow: hidden; color: #d0d0d0; font: 10px/1 {FONT_STACK}; white-space: pre; }}
</style></head><body>
<header><img src="source-crop.png" alt="Tight source crop"><div><h1>Face-only ASCII candidates</h1>
<p>Compare glasses, hairstyle, jawline, whitespace, and 50% readability.</p></div></header>
<main>{''.join(cards)}</main></body></html>
"""
    (output_dir / "contact-sheet.html").write_text(document, encoding="utf-8")


def validate_settings(parser: argparse.ArgumentParser, settings: Settings) -> None:
    if not 40 <= settings.width <= 52:
        parser.error("width must be between 40 and 52 characters")
    if not 30 <= settings.height <= 40:
        parser.error("height must be between 30 and 40 lines")
    if settings.contrast <= 0 or settings.gamma <= 0:
        parser.error("contrast and gamma must be positive")
    for name, value in (
        ("edge strength", settings.edge_strength),
        ("edge threshold", settings.edge_threshold),
        ("background threshold", settings.background_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source portrait photograph")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="selected ASCII output")
    parser.add_argument("--crop", type=parse_crop, default=DEFAULT_CROP, help="pixel crop: left,top,right,bottom")
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--height", type=int, default=38)
    parser.add_argument("--contrast", type=float, default=1.15)
    parser.add_argument("--gamma", type=float, default=0.88)
    parser.add_argument("--edge-strength", type=float, default=0.45)
    parser.add_argument("--edge-threshold", type=float, default=0.34)
    parser.add_argument("--background-threshold", type=float, default=0.21)
    parser.add_argument("--ramp", default=DEFAULT_RAMP)
    parser.add_argument("--candidates-dir", type=Path, help="write 18 candidates and a temporary contact sheet")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source image not found: {args.source}")
    settings = Settings(
        crop=args.crop,
        width=args.width,
        height=args.height,
        contrast=args.contrast,
        gamma=args.gamma,
        edge_strength=args.edge_strength,
        edge_threshold=args.edge_threshold,
        background_threshold=args.background_threshold,
        ramp=args.ramp,
    )
    validate_settings(parser, settings)

    if args.candidates_dir:
        write_contact_sheet(args.source, args.candidates_dir)
        print(f"wrote 18 candidates and contact sheet to {args.candidates_dir}")
        return

    rows, _ = portrait_rows(args.source, settings)
    write_rows(rows, args.output)
    print(f"wrote {len(rows)} rows × {len(rows[0])} columns to {args.output}")


if __name__ == "__main__":
    main()
