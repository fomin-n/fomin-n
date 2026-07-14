#!/usr/bin/env python3
"""Render responsive, self-contained SVG terminal profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO_ROOT / "profile.json"
DEFAULT_PORTRAIT = REPO_ROOT / "assets" / "portrait.txt"
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "profile-terminal.svg"
DEFAULT_MOBILE_OUTPUT = REPO_ROOT / "assets" / "profile-terminal-mobile.svg"
FONT_STACK = "SFMono-Regular, 'SF Mono', Menlo, Monaco, Consolas, 'Liberation Mono', monospace"

COLORS = {
    "border": "#4a4a4a",
    "title": "#1c1c1c",
    "background": "#000000",
    "primary": "#d0d0d0",
    "bright": "#f2f2f2",
    "muted": "#6c6c6c",
    "yellow": "#ffd75f",
    "orange": "#ffaf00",
    "blue": "#5fafff",
    "green": "#00af87",
}


def load_profile(path: Path) -> dict[str, object]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {"username", "name", "roles", "location", "focus", "contacts", "statement"}
    missing = required - profile.keys()
    if missing:
        raise ValueError(f"profile is missing: {', '.join(sorted(missing))}")
    if len(profile["roles"]) != 2 or len(profile["focus"]) != 2:
        raise ValueError("profile requires exactly two role and two focus lines")
    if len(profile["contacts"]) != 7:
        raise ValueError("profile requires exactly seven contacts")
    return profile


def load_portrait(path: Path) -> list[str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    widths = {len(row) for row in rows}
    if not rows or len(widths) != 1:
        raise ValueError("portrait must contain equally sized rows")
    width = widths.pop()
    if not 36 <= width <= 44 or not 27 <= len(rows) <= 33:
        raise ValueError(f"portrait dimensions {width}×{len(rows)} are outside the supported range")
    return rows


def text_line(
    x: float,
    y: float,
    parts: list[tuple[str, str]],
    *,
    size: float,
    weight: int = 400,
    anchor: str | None = None,
    attributes: dict[str, str] | None = None,
) -> str:
    svg_attributes = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'font-size="{size:g}"',
        f'font-weight="{weight}"',
        'xml:space="preserve"',
    ]
    if anchor:
        svg_attributes.append(f'text-anchor="{anchor}"')
    if attributes:
        svg_attributes.extend(
            f'{escape(name)}="{escape(value)}"' for name, value in attributes.items()
        )
    spans = "".join(
        f'<tspan fill="{COLORS[color]}">{escape(value)}</tspan>' for value, color in parts
    )
    return f"    <text {' '.join(svg_attributes)}>{spans}</text>"


def portrait_lines(
    rows: list[str], *, x: float, y: float, size: float, line_height: float
) -> list[str]:
    return [
        text_line(
            x,
            y + index * line_height,
            [(row, "primary")],
            size=size,
            attributes={"data-kind": "portrait", "data-line": str(index + 1)},
        )
        for index, row in enumerate(rows)
    ]


def dotted_row(
    *,
    label_x: float,
    leader_x: float,
    value_x: float,
    y: float,
    label: str,
    value: str,
    size: float,
    section: str,
    value_color: str,
) -> str:
    """Draw one fixed-coordinate key, leader, and right-aligned value row."""

    character_width = size * 0.60
    value_start = value_x - len(value) * character_width
    dot_count = max(1, math.floor((value_start - leader_x - character_width * 0.7) / character_width))
    row_attributes = f'data-row="{escape(label)}" data-section="{escape(section)}"'
    return "\n".join(
        [
            f"    <g {row_attributes}>",
            text_line(
                label_x,
                y,
                [(label, "yellow")],
                size=size,
                weight=600,
                attributes={"data-part": "label"},
            ),
            text_line(
                leader_x,
                y,
                [("." * dot_count, "muted")],
                size=size,
                attributes={"data-part": "leader"},
            ),
            text_line(
                value_x,
                y,
                [(value, value_color)],
                size=size,
                anchor="end",
                attributes={"data-part": "value"},
            ),
            "    </g>",
        ]
    )


def window_chrome(width: int, height: int, title_height: int, title_size: float) -> list[str]:
    return [
        f'    <rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14" fill="{COLORS["background"]}" stroke="{COLORS["border"]}" stroke-width="2"/>',
        f'    <path d="M15 1 H{width - 15} Q{width - 1} 1 {width - 1} 15 V{title_height} H1 V15 Q1 1 15 1 Z" fill="{COLORS["title"]}"/>',
        f'    <path d="M1 {title_height} H{width - 1}" stroke="{COLORS["border"]}" stroke-width="1"/>',
        '    <circle cx="25" cy="27" r="7" fill="#ff5f57"/>',
        '    <circle cx="49" cy="27" r="7" fill="#febc2e"/>',
        '    <circle cx="73" cy="27" r="7" fill="#28c840"/>',
        text_line(width / 2, 35, [("~ | whoami", "muted")], size=title_size, anchor="middle"),
    ]


def svg_shell(width: int, height: int, title: str, description: str, body: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">',
            f'  <title id="title">{escape(title)}</title>',
            f'  <desc id="desc">{escape(description)}</desc>',
            f'  <g font-family="{FONT_STACK}">',
            *body,
            "  </g>",
            "</svg>",
            "",
        ]
    )


def build_desktop(profile: dict[str, object], rows: list[str]) -> str:
    width, height = 1200, 900
    username = str(profile["username"])
    roles = [str(value) for value in profile["roles"]]
    focus = [str(value) for value in profile["focus"]]
    contacts = [(str(label), str(value)) for label, value in profile["contacts"]]

    label_x, leader_x, value_x = 590, 725, 1165
    focus_parts = ["Production ML", "LLM applications", "AI agents · NLP", "deep learning"]
    if " · ".join(focus_parts[:2]) != focus[0] or " · ".join(focus_parts[2:]) != focus[1]:
        raise ValueError("desktop focus segmentation no longer matches profile.json")
    profile_rows = [
        ("Name", str(profile["name"])),
        ("Role", roles[0]),
        ("Role", roles[1]),
        ("Location", str(profile["location"])),
        *(("Focus", value) for value in focus_parts),
    ]
    content: list[str] = [
        *window_chrome(width, height, 54, 24),
        text_line(34, 92, [("^^", "orange"), (" >>> ", "muted"), ("whoami", "bright")], size=26),
        '    <path d="M555 112 V796" stroke="#4a4a4a" stroke-width="1" opacity="0.6"/>',
        *portrait_lines(rows, x=24, y=126, size=21.5, line_height=20.45),
        text_line(590, 130, [(f"{username}@github", "bright")], size=30, weight=600),
        text_line(590, 158, [("PROFILE", "orange"), (" ", "muted"), ("-" * 30, "muted")], size=22),
    ]
    for index, (label, value) in enumerate(profile_rows):
        content.append(
            dotted_row(
                label_x=label_x,
                leader_x=leader_x,
                value_x=value_x,
                y=198 + index * 36,
                label=label,
                value=value,
                size=25,
                section="profile",
                value_color="primary",
            )
        )
    content.append(
        text_line(590, 502, [("CONTACT", "orange"), (" ", "muted"), ("-" * 30, "muted")], size=22)
    )
    for index, (label, value) in enumerate(contacts):
        content.append(
            dotted_row(
                label_x=label_x,
                leader_x=leader_x,
                value_x=value_x,
                y=542 + index * 36,
                label=label,
                value=value,
                size=25,
                section="contact",
                value_color="blue",
            )
        )
    content.extend(
        [
            '    <path d="M26 810 H1174" stroke="#4a4a4a" stroke-width="1" opacity="0.6"/>',
            text_line(
                32,
                844,
                [("# ", "green"), ("Building production ML systems, LLM applications, and AI agents across", "muted")],
                size=21.2,
                attributes={"data-kind": "statement"},
            ),
            text_line(
                58,
                870,
                [("modeling, data pipelines, deployment, and monitoring.", "muted")],
                size=21.2,
                attributes={"data-kind": "statement"},
            ),
        ]
    )
    return svg_shell(
        width,
        height,
        "Terminal profile for Nikita Fomin",
        "A macOS terminal window with an ASCII portrait and aligned public profile information.",
        content,
    )


def mobile_profile_rows(profile: dict[str, object]) -> list[tuple[str, str]]:
    roles = [str(value) for value in profile["roles"]]
    focus = [str(value) for value in profile["focus"]]
    focus_parts = [
        "Production ML",
        "LLM applications",
        "AI agents · NLP",
        "deep learning",
    ]
    if " · ".join(focus_parts[:2]) != focus[0] or " · ".join(focus_parts[2:]) != focus[1]:
        raise ValueError("mobile focus segmentation no longer matches profile.json")
    return [
        ("Name", str(profile["name"])),
        ("Role", roles[0]),
        ("Role", roles[1]),
        ("Location", str(profile["location"])),
        *(("Focus", value) for value in focus_parts),
    ]


def build_mobile(profile: dict[str, object], rows: list[str]) -> str:
    width, height = 600, 1490
    username = str(profile["username"])
    contacts = [(str(label), str(value)) for label, value in profile["contacts"]]
    row_size = 23.6
    portrait_size = 18.5
    portrait_width = len(rows[0]) * portrait_size * 0.60
    portrait_x = (width - portrait_width) / 2
    label_x, leader_x, value_x = 30, 150, 570

    content: list[str] = [
        *window_chrome(width, height, 54, 23),
        text_line(28, 92, [("^^", "orange"), (" >>> ", "muted"), ("whoami", "bright")], size=25),
        *portrait_lines(rows, x=portrait_x, y=122, size=portrait_size, line_height=18.5),
        '    <path d="M28 682 H572" stroke="#4a4a4a" stroke-width="1" opacity="0.6"/>',
        text_line(30, 724, [(f"{username}@github", "bright")], size=27, weight=600),
        text_line(30, 754, [("PROFILE", "orange"), (" ", "muted"), ("-" * 22, "muted")], size=21),
    ]
    profile_rows = mobile_profile_rows(profile)
    for index, (label, value) in enumerate(profile_rows):
        content.append(
            dotted_row(
                label_x=label_x,
                leader_x=leader_x,
                value_x=value_x,
                y=794 + index * 34,
                label=label,
                value=value,
                size=row_size,
                section="profile",
                value_color="primary",
            )
        )
    contact_header_y = 794 + len(profile_rows) * 34 + 22
    content.append(
        text_line(30, contact_header_y, [("CONTACT", "orange"), (" ", "muted"), ("-" * 22, "muted")], size=21)
    )
    for index, (label, value) in enumerate(contacts):
        content.append(
            dotted_row(
                label_x=label_x,
                leader_x=leader_x,
                value_x=value_x,
                y=contact_header_y + 40 + index * 34,
                label=label,
                value=value,
                size=row_size,
                section="contact",
                value_color="blue",
            )
        )
    content.extend(
        [
            '    <path d="M28 1372 H572" stroke="#4a4a4a" stroke-width="1" opacity="0.6"/>',
            text_line(30, 1406, [("# ", "green"), ("Building production ML systems,", "muted")], size=19, attributes={"data-kind": "statement"}),
            text_line(52, 1433, [("LLM applications, and AI agents across modeling,", "muted")], size=19, attributes={"data-kind": "statement"}),
            text_line(52, 1460, [("data pipelines, deployment, and monitoring.", "muted")], size=19, attributes={"data-kind": "statement"}),
        ]
    )
    return svg_shell(
        width,
        height,
        "Terminal profile for Nikita Fomin",
        "A compact macOS terminal window with an ASCII portrait and aligned public profile information.",
        content,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--portrait", type=Path, default=DEFAULT_PORTRAIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mobile-output", type=Path, default=DEFAULT_MOBILE_OUTPUT)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    rows = load_portrait(args.portrait)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mobile_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_desktop(profile, rows), encoding="utf-8")
    args.mobile_output.write_text(build_mobile(profile, rows), encoding="utf-8")
    print(f"wrote 1200×900 SVG to {args.output}")
    print(f"wrote 600×1490 SVG to {args.mobile_output}")


if __name__ == "__main__":
    main()
