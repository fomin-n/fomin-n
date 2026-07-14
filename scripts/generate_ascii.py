#!/usr/bin/env python3
"""Generate a fixed-width ASCII portrait from a source photograph.

The defaults are tuned for the profile photograph used by this repository.
All processing is deterministic; pass explicit options to experiment with crops
or tonal mapping without changing the implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "portrait.txt"
DEFAULT_RAMP = "@#*+=-:. "


def parse_crop(value: str) -> tuple[float, float, float, float]:
    try:
        crop = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must contain four decimals") from exc
    if len(crop) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    left, top, right, bottom = crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise argparse.ArgumentTypeError("crop coordinates must be ordered within 0..1")
    return left, top, right, bottom


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def suppress_background(gray: Image.Image, strength: float) -> Image.Image:
    """Fade peripheral background while retaining the head and shoulders."""

    pixels = np.asarray(gray, dtype=np.float32)
    height, width = pixels.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)

    # A compact head ellipse plus a torso mask that widens toward the bottom.
    head_distance = ((x - 0.5) / 0.34) ** 2 + ((y - 0.28) / 0.35) ** 2
    head_mask = smoothstep((1.18 - head_distance) / 0.38)

    torso_radius = 0.26 + 0.34 * np.clip((y - 0.38) / 0.62, 0.0, 1.0)
    torso_mask = smoothstep((torso_radius - np.abs(x - 0.5)) / 0.10)
    torso_mask *= smoothstep((y - 0.30) / 0.14)

    subject_mask = np.maximum(head_mask, torso_mask)
    subject_mask = (1.0 - strength) + strength * subject_mask
    flattened = pixels * subject_mask + 244.0 * (1.0 - subject_mask)
    return Image.fromarray(np.clip(flattened, 0, 255).astype(np.uint8), "L")


def process_image(
    source: Path,
    crop: tuple[float, float, float, float],
    width: int,
    height: int,
    contrast: float,
    gamma: float,
    edge_strength: float,
    background_strength: float,
    invert: bool,
) -> Image.Image:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    left, top, right, bottom = crop
    crop_box = (
        round(image.width * left),
        round(image.height * top),
        round(image.width * right),
        round(image.height * bottom),
    )
    gray = image.crop(crop_box).convert("L")
    gray = suppress_background(gray, background_strength)
    gray = ImageOps.autocontrast(gray, cutoff=(1.0, 1.0))
    gray = ImageEnhance.Contrast(gray).enhance(contrast)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2.0, percent=190, threshold=3))

    oversampled = gray.resize((width * 4, height * 4), Image.Resampling.LANCZOS)
    edges = oversampled.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.65))
    base_small = oversampled.resize((width, height), Image.Resampling.LANCZOS)
    edge_small = edges.resize((width, height), Image.Resampling.LANCZOS)

    base = np.asarray(base_small, dtype=np.float32)
    edge = np.asarray(edge_small, dtype=np.float32)
    combined = np.clip(base - edge_strength * edge, 0.0, 255.0)
    combined = np.power(combined / 255.0, gamma) * 255.0
    if invert:
        combined = 255.0 - combined
    return Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8), "L")


def image_to_ascii(image: Image.Image, ramp: str) -> list[str]:
    if len(ramp) < 2:
        raise ValueError("character ramp must contain at least two characters")
    values = np.asarray(image, dtype=np.float32)
    indices = np.rint(values / 255.0 * (len(ramp) - 1)).astype(np.int16)
    rows = ["".join(ramp[index] for index in row) for row in indices]
    width = image.width
    return [row[:width].ljust(width) for row in rows]


def write_preview(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = image.resize((image.width * 12, image.height * 20), Image.Resampling.NEAREST)
    preview.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source portrait photograph")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, help="optional processed grayscale preview")
    parser.add_argument("--width", type=int, default=42)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--crop", type=parse_crop, default=parse_crop("0.21,0.11,0.79,0.75"))
    parser.add_argument("--contrast", type=float, default=1.20)
    parser.add_argument("--gamma", type=float, default=0.82)
    parser.add_argument("--edge-strength", type=float, default=0.36)
    parser.add_argument("--background-strength", type=float, default=0.88)
    parser.add_argument("--ramp", default=DEFAULT_RAMP)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source image not found: {args.source}")
    if not 36 <= args.width <= 44:
        parser.error("width must be between 36 and 44 characters")
    if not 27 <= args.height <= 34:
        parser.error("height must be between 27 and 34 lines")
    for name, value in (
        ("edge strength", args.edge_strength),
        ("background strength", args.background_strength),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    if args.gamma <= 0 or args.contrast <= 0:
        parser.error("gamma and contrast must be positive")

    processed = process_image(
        source=args.source,
        crop=args.crop,
        width=args.width,
        height=args.height,
        contrast=args.contrast,
        gamma=args.gamma,
        edge_strength=args.edge_strength,
        background_strength=args.background_strength,
        invert=args.invert,
    )
    rows = image_to_ascii(processed, args.ramp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    if args.preview:
        write_preview(processed, args.preview)

    print(f"wrote {len(rows)} rows × {len(rows[0])} columns to {args.output}")


if __name__ == "__main__":
    main()
