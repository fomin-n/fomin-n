#!/usr/bin/env python3
"""Generate a sparse, structure-aware ASCII portrait from a photograph."""

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
DEFAULT_CROP = (620, 600, 2400, 2920)
DEFAULT_RAMP = " .:+#@"
FONT_STACK = "SFMono-Regular, 'SF Mono', Menlo, Monaco, Consolas, 'Liberation Mono', monospace"


@dataclass(frozen=True)
class Settings:
    crop: tuple[int, int, int, int]
    width: int
    height: int
    contrast: float
    gamma: float
    edge_threshold: float
    shade_threshold: float
    silhouette_threshold: float
    ramp: str


@dataclass(frozen=True)
class PreparedPortrait:
    crop: Image.Image
    darkness: np.ndarray
    peak_darkness: np.ndarray
    edge: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    matte: np.ndarray
    matte_edge: np.ndarray
    matte_x: np.ndarray
    matte_y: np.ndarray


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


def clahe_like(gray: Image.Image, tiles: int = 8, clip_limit: float = 2.0) -> Image.Image:
    """Apply clipped local histogram equalization without external CV dependencies."""

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

    top = (
        luts[y0[:, None], x0[None, :], source] * (1.0 - wx)
        + luts[y0[:, None], x1[None, :], source] * wx
    )
    bottom = (
        luts[y1[:, None], x0[None, :], source] * (1.0 - wx)
        + luts[y1[:, None], x1[None, :], source] * wx
    )
    result = top * (1.0 - wy) + bottom * wy
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "L")


def subject_matte(width: int, height: int) -> np.ndarray:
    """Create a clean hair, face, neck, and shoulder silhouette."""

    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)

    face_distance = ((x - 0.50) / 0.285) ** 2 + ((y - 0.385) / 0.335) ** 2
    face = smoothstep((1.035 - face_distance) / 0.075)

    hair_distance = ((x - 0.485) / 0.325) ** 2 + ((y - 0.205) / 0.205) ** 2
    hair = smoothstep((1.02 - hair_distance) / 0.085)

    neck_half_width = 0.14 + 0.03 * np.clip((y - 0.65) / 0.16, 0.0, 1.0)
    neck = smoothstep((neck_half_width - np.abs(x - 0.50)) / 0.025)
    neck *= smoothstep((y - 0.61) / 0.045) * smoothstep((0.86 - y) / 0.055)

    shoulder_progress = np.clip((y - 0.735) / 0.265, 0.0, 1.0)
    shoulder_half_width = 0.16 + 0.44 * np.sqrt(shoulder_progress)
    shoulders = smoothstep((shoulder_half_width - np.abs(x - 0.50)) / 0.035)
    shoulders *= smoothstep((y - 0.72) / 0.045)

    matte = np.maximum.reduce([face, hair, neck, shoulders])
    matte *= smoothstep((1.01 - y) / 0.025)
    return matte.astype(np.float32)


def load_source(source: Path) -> Image.Image:
    with Image.open(source) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def crop_source(
    source: Image.Image, crop: tuple[int, int, int, int]
) -> Image.Image:
    left, top, right, bottom = crop
    if right > source.width or bottom > source.height:
        raise ValueError(f"crop {crop} exceeds source dimensions {source.size}")
    return source.crop(crop)


def prepare_portrait(source: Image.Image, settings: Settings) -> PreparedPortrait:
    crop = crop_source(source, settings.crop)
    analysis_width = 520
    analysis_height = max(560, round(analysis_width * crop.height / crop.width))
    crop = crop.resize((analysis_width, analysis_height), Image.Resampling.LANCZOS)

    gray = crop.convert("L").filter(ImageFilter.MedianFilter(size=3))
    raw = np.asarray(gray, dtype=np.float32)
    illumination = np.asarray(gray.filter(ImageFilter.BoxBlur(radius=38)), dtype=np.float32)
    balanced = np.clip(raw - 0.50 * (illumination - 128.0), 0.0, 255.0)
    balanced_image = Image.fromarray(balanced.astype(np.uint8), "L")
    local = clahe_like(balanced_image)
    local = Image.blend(balanced_image, local, 0.38)
    local = ImageEnhance.Contrast(local).enhance(settings.contrast)
    tones = np.asarray(local, dtype=np.float32) / 255.0
    tones = np.power(np.clip(tones, 0.0, 1.0), settings.gamma)
    darkness = 1.0 - tones

    edge_source = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius=0.65)), dtype=np.float32
    ) / 255.0
    grad_y, grad_x = np.gradient(edge_source)
    edge = np.hypot(grad_x, grad_y)
    edge_scale = max(float(np.percentile(edge, 98.8)), 1e-6)
    edge = np.clip(edge / edge_scale, 0.0, 1.0)

    matte = subject_matte(analysis_width, analysis_height)
    matte_y, matte_x = np.gradient(matte)
    matte_edge = np.hypot(matte_x, matte_y)
    matte_scale = max(float(np.percentile(matte_edge, 99.0)), 1e-6)
    matte_edge = np.clip(matte_edge / matte_scale, 0.0, 1.0)

    raw_darkness = 1.0 - np.asarray(gray, dtype=np.float32) / 255.0
    return PreparedPortrait(
        crop=crop,
        darkness=darkness,
        peak_darkness=raw_darkness,
        edge=edge,
        grad_x=grad_x,
        grad_y=grad_y,
        matte=matte,
        matte_edge=matte_edge,
        matte_x=matte_x,
        matte_y=matte_y,
    )


def resize_mean(values: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), "F")
    return np.asarray(
        image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32
    )


def pool_peak(
    strength: np.ndarray,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_edges = np.linspace(0, strength.shape[0], height + 1, dtype=int)
    x_edges = np.linspace(0, strength.shape[1], width + 1, dtype=int)
    peaks = np.zeros((height, width), dtype=np.float32)
    pooled_x = np.zeros_like(peaks)
    pooled_y = np.zeros_like(peaks)
    for row in range(height):
        for column in range(width):
            y0, y1 = y_edges[row], y_edges[row + 1]
            x0, x1 = x_edges[column], x_edges[column + 1]
            patch = strength[y0:y1, x0:x1]
            flat_index = int(np.argmax(patch))
            local_y, local_x = np.unravel_index(flat_index, patch.shape)
            sample_y, sample_x = y0 + local_y, x0 + local_x
            peaks[row, column] = strength[sample_y, sample_x]
            pooled_x[row, column] = direction_x[sample_y, sample_x]
            pooled_y[row, column] = direction_y[sample_y, sample_x]
    return peaks, pooled_x, pooled_y


def directional_character(gx: float, gy: float) -> str:
    if abs(gx) > abs(gy) * 1.5:
        return "|"
    if abs(gy) > abs(gx) * 1.5:
        return "-"
    return "/" if gx * gy >= 0 else "\\"


def portrait_rows(prepared: PreparedPortrait, settings: Settings) -> list[str]:
    """Render a sparse semantic portrait, using the photograph for local tone.

    At the deliberately low output resolution, thin eyeglass rims and facial
    features cannot survive ordinary averaging.  The silhouette and feature
    guides below keep those structures continuous, while hair and garment tone
    still come directly from the processed source crop.
    """

    width, height = settings.width, settings.height
    mean_darkness = np.clip(resize_mean(prepared.darkness, width, height), 0.0, 1.0)
    mean_edge = np.clip(resize_mean(prepared.edge, width, height), 0.0, 1.0)
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * (width - 1)), round(y * (height - 1))

    def put(column: int, row: int, character: str, *, overwrite: bool = True) -> None:
        if 0 <= column < width and 0 <= row < height:
            if overwrite or grid[row][column] == " ":
                grid[row][column] = character

    def line(
        start: tuple[float, float],
        end: tuple[float, float],
        character: str,
        *,
        overwrite: bool = True,
    ) -> None:
        x0, y0 = point(*start)
        x1, y1 = point(*end)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            fraction = step / steps
            put(
                round(x0 + (x1 - x0) * fraction),
                round(y0 + (y1 - y0) * fraction),
                character,
                overwrite=overwrite,
            )

    def horizontal(x0: float, x1: float, y: float, character: str) -> None:
        start_x, row = point(x0, y)
        end_x, _ = point(x1, y)
        for column in range(start_x, end_x + 1):
            put(column, row, character)

    # Sparse photograph-derived tone: dense enough to retain the swept hair and
    # T-shirt collar, but never allowed to become a rectangular background.
    for row in range(height):
        y = row / max(height - 1, 1)
        for column in range(width):
            x = column / max(width - 1, 1)
            hair_distance = ((x - 0.49) / 0.36) ** 2 + ((y - 0.18) / 0.23) ** 2
            shoulder_progress = np.clip((y - 0.78) / 0.22, 0.0, 1.0)
            shoulder_half_width = 0.12 + 0.43 * np.sqrt(shoulder_progress)
            in_hair = hair_distance <= 1.0 and y <= 0.36
            in_shirt = y >= 0.79 and abs(x - 0.50) <= shoulder_half_width
            if not (in_hair or in_shirt):
                continue
            tone = mean_darkness[row, column]
            detail = mean_edge[row, column]
            threshold = settings.shade_threshold + (0.01 if in_hair else 0.08)
            if tone < threshold or (row + 2 * column) % 3 == 0:
                continue
            if in_hair and detail >= settings.edge_threshold * 0.75:
                character = "\\" if column >= width * 0.47 else "/"
            else:
                normalized = np.clip((tone - threshold) / max(1.0 - threshold, 1e-6), 0, 1)
                character = settings.ramp[round(normalized * (len(settings.ramp) - 1))]
            put(column, row, character)

    # Hair outline and characteristic asymmetrical fringe.
    line((0.15, 0.23), (0.23, 0.07), "/")
    line((0.23, 0.07), (0.42, 0.00), "-")
    line((0.42, 0.00), (0.60, 0.04), "\\")
    line((0.60, 0.04), (0.79, 0.20), "\\")
    line((0.15, 0.23), (0.17, 0.38), "|")
    line((0.79, 0.20), (0.82, 0.39), "|")
    for offset in (0.00, 0.07, 0.14, 0.21):
        line((0.25 + offset, 0.06), (0.42 + offset * 0.55, 0.27), "\\")
    line((0.21, 0.17), (0.43, 0.28), "/")
    line((0.29, 0.13), (0.52, 0.30), "/")
    line((0.38, 0.09), (0.61, 0.27), "\\")

    # Clean face contour, ears, neck, and shoulders.
    face_center_y, face_radius_x, face_radius_y = 0.43, 0.31, 0.34
    for row in range(round(height * 0.17), round(height * 0.77)):
        y = row / max(height - 1, 1)
        vertical = (y - face_center_y) / face_radius_y
        if abs(vertical) > 1.0:
            continue
        half_width = face_radius_x * np.sqrt(max(0.0, 1.0 - vertical * vertical))
        left = round((0.50 - half_width) * (width - 1))
        right = round((0.50 + half_width) * (width - 1))
        put(left, row, "/" if y < face_center_y else "\\")
        put(right, row, "\\" if y < face_center_y else "/")
    line((0.16, 0.39), (0.15, 0.53), "(")
    line((0.83, 0.39), (0.84, 0.53), ")")
    line((0.40, 0.72), (0.40, 0.82), "|")
    line((0.60, 0.72), (0.60, 0.82), "|")
    line((0.40, 0.80), (0.08, 0.98), "/")
    line((0.60, 0.80), (0.92, 0.98), "\\")
    line((0.08, 0.98), (0.01, 1.00), "-")
    line((0.92, 0.98), (0.99, 1.00), "-")
    line((0.34, 0.83), (0.50, 0.92), "\\")
    line((0.50, 0.92), (0.66, 0.83), "/")

    # Rectangular frames are drawn as continuous primitives so they remain
    # legible after GitHub scales the SVG to typical profile width.
    glasses_top = round(0.34 * (height - 1))
    glasses_bottom = round(0.45 * (height - 1))
    left_outer = round(0.19 * (width - 1))
    left_inner = round(0.45 * (width - 1))
    right_inner = round(0.55 * (width - 1))
    right_outer = round(0.81 * (width - 1))
    for x0, x1 in ((left_outer, left_inner), (right_inner, right_outer)):
        for column in range(x0 + 1, x1):
            put(column, glasses_top, "-")
            put(column, glasses_bottom, "-")
        for row in range(glasses_top + 1, glasses_bottom):
            put(x0, row, "|")
            put(x1, row, "|")
        for column in (x0, x1):
            put(column, glasses_top, "+")
            put(column, glasses_bottom, "+")
    bridge_row = round(0.38 * (height - 1))
    for column in range(left_inner + 1, right_inner):
        put(column, bridge_row, "-")
    eye_row = round(0.39 * (height - 1))
    for eye_x in (0.33, 0.67):
        eye_column, _ = point(eye_x, 0.39)
        put(eye_column - 1, eye_row, ".")
        put(eye_column, eye_row, "o")
        put(eye_column + 1, eye_row, ".")

    # Nose, mouth, and chin use a few stable strokes instead of noisy shading.
    center = round(0.50 * (width - 1))
    nose_top = glasses_bottom + 1
    put(center - 1, nose_top, "\\")
    put(center, nose_top + 1, "|")
    put(center, nose_top + 2, "|")
    put(center - 1, nose_top + 3, "\\")
    put(center, nose_top + 3, "_")
    put(center + 1, nose_top + 3, "/")
    mouth_row = round(0.64 * (height - 1))
    mouth_left, _ = point(0.40, 0.64)
    mouth_right, _ = point(0.60, 0.64)
    put(mouth_left, mouth_row, "\\")
    for column in range(mouth_left + 1, mouth_right):
        put(column, mouth_row, "-")
    put(mouth_right, mouth_row, "/")
    horizontal(0.45, 0.55, 0.72, "_")

    return ["".join(row).ljust(width) for row in grid]


def candidate_settings() -> list[Settings]:
    crops = [
        (620, 600, 2400, 2920),
        (520, 520, 2500, 3050),
        (700, 650, 2320, 2860),
    ]
    sizes = [(38, 29), (40, 30), (42, 31)]
    variants = [
        dict(
            contrast=1.08,
            gamma=0.94,
            edge_threshold=0.30,
            shade_threshold=0.31,
            silhouette_threshold=0.22,
        ),
        dict(
            contrast=1.22,
            gamma=0.86,
            edge_threshold=0.24,
            shade_threshold=0.36,
            silhouette_threshold=0.18,
        ),
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
    prepared_cache: dict[tuple[int, int, int, int], PreparedPortrait] = {}
    crop_paths: dict[tuple[int, int, int, int], str] = {}

    for index, settings in enumerate(candidate_settings(), start=1):
        prepared = prepared_cache.get(settings.crop)
        if prepared is None:
            prepared = prepare_portrait(source_image, settings)
            prepared_cache[settings.crop] = prepared
            crop_name = f"crop-{len(prepared_cache):02d}.png"
            prepared.crop.save(output_dir / crop_name)
            crop_paths[settings.crop] = crop_name
        rows = portrait_rows(prepared, settings)
        candidate_id = f"candidate-{index:02d}"
        write_rows(rows, output_dir / f"{candidate_id}.txt")
        parameters = asdict(settings)
        parameters["crop"] = list(settings.crop)
        manifest.append(
            {"id": candidate_id, "source_crop": crop_paths[settings.crop], **parameters}
        )
        summary = (
            f"{settings.width}×{settings.height} · crop {settings.crop}<br>"
            f"contrast {settings.contrast} · gamma {settings.gamma} · "
            f"edge {settings.edge_threshold} · shade {settings.shade_threshold} · "
            f"silhouette {settings.silhouette_threshold}"
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
body {{ margin: 0; background: #0d1117; color: #d0d0d0; font-family: {FONT_STACK}; }}
header {{ display: flex; gap: 22px; align-items: center; padding: 24px; border-bottom: 1px solid #333; }}
header img {{ width: 180px; max-height: 240px; object-fit: cover; object-position: center; }}
h1 {{ color: #f2f2f2; font-size: 24px; }}
main {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; padding: 16px; }}
article {{ min-width: 0; border: 1px solid #333; border-radius: 8px; padding: 14px; background: #000; }}
h2 {{ margin: 0 0 6px; color: #ffd75f; font-size: 18px; }}
p {{ min-height: 48px; margin: 0 0 10px; color: #858585; font: 11px/1.35 {FONT_STACK}; }}
pre {{ margin: 0; overflow: hidden; color: #d0d0d0; font: 21.5px/1.02 {FONT_STACK}; white-space: pre; }}
</style></head><body>
<header><img src="crop-01.png" alt="Source portrait crop"><div><h1>New face-and-shoulders candidates</h1>
<p>Choose by hair, rectangular glasses, facial features, outline, and clean empty background.</p></div></header>
<main>{''.join(cards)}</main></body></html>
"""
    (output_dir / "contact-sheet.html").write_text(document, encoding="utf-8")


def validate_settings(parser: argparse.ArgumentParser, settings: Settings) -> None:
    if not 36 <= settings.width <= 44:
        parser.error("width must be between 36 and 44 characters")
    if not 27 <= settings.height <= 33:
        parser.error("height must be between 27 and 33 lines")
    if settings.contrast <= 0 or settings.gamma <= 0:
        parser.error("contrast and gamma must be positive")
    for name, value in (
        ("edge threshold", settings.edge_threshold),
        ("shade threshold", settings.shade_threshold),
        ("silhouette threshold", settings.silhouette_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    if len(settings.ramp) < 3 or settings.ramp[0] != " ":
        parser.error("ramp must start with a space and contain at least three characters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source portrait photograph")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop", type=parse_crop, default=DEFAULT_CROP)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--contrast", type=float, default=1.22)
    parser.add_argument("--gamma", type=float, default=0.86)
    parser.add_argument("--edge-threshold", type=float, default=0.24)
    parser.add_argument("--shade-threshold", type=float, default=0.36)
    parser.add_argument("--silhouette-threshold", type=float, default=0.18)
    parser.add_argument("--ramp", default=DEFAULT_RAMP)
    parser.add_argument("--candidates-dir", type=Path, help="write 18 candidates and a contact sheet")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source image not found: {args.source}")
    settings = Settings(
        crop=args.crop,
        width=args.width,
        height=args.height,
        contrast=args.contrast,
        gamma=args.gamma,
        edge_threshold=args.edge_threshold,
        shade_threshold=args.shade_threshold,
        silhouette_threshold=args.silhouette_threshold,
        ramp=args.ramp,
    )
    validate_settings(parser, settings)

    if args.candidates_dir:
        write_contact_sheet(args.source, args.candidates_dir)
        print(f"wrote 18 new portrait candidates to {args.candidates_dir}")
        return

    source = load_source(args.source)
    prepared = prepare_portrait(source, settings)
    rows = portrait_rows(prepared, settings)
    write_rows(rows, args.output)
    print(f"wrote {len(rows)} rows × {len(rows[0])} columns to {args.output}")


if __name__ == "__main__":
    main()
