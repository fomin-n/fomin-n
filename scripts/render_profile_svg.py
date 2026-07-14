#!/usr/bin/env python3
"""Render responsive, self-contained SVG terminal profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO_ROOT / "profile.json"
DEFAULT_PORTRAIT = REPO_ROOT / "assets" / "portrait.txt"
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "profile-terminal.svg"
DEFAULT_MOBILE_OUTPUT = REPO_ROOT / "assets" / "profile-terminal-mobile.svg"
FONT_STACK = "SFMono-Regular, SF Mono, Menlo, Monaco, Consolas, Liberation Mono, monospace"

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
    if not 40 <= width <= 52 or not 30 <= len(rows) <= 40:
        raise ValueError(f"portrait dimensions {width}×{len(rows)} are outside the supported range")
    return rows


def text_line(
    x: float,
    y: float,
    parts: list[tuple[str, str]],
    *,
    size: float = 12.5,
    weight: int = 400,
    anchor: str | None = None,
) -> str:
    attributes = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'font-size="{size:g}"',
        f'font-weight="{weight}"',
        'xml:space="preserve"',
    ]
    if anchor:
        attributes.append(f'text-anchor="{anchor}"')
    spans = "".join(
        f'<tspan fill="{COLORS[color]}">{escape(value)}</tspan>' for value, color in parts
    )
    return f"    <text {' '.join(attributes)}>{spans}</text>"


def portrait_lines(
    rows: list[str], *, x: float, y: float, size: float, line_height: float
) -> list[str]:
    return [
        text_line(x, y + index * line_height, [(row, "primary")], size=size)
        for index, row in enumerate(rows)
    ]


def field_line(
    label_x: float,
    value_x: float,
    y: float,
    label: str,
    value: str,
    *,
    value_color: str = "primary",
    size: float = 12.5,
) -> str:
    return "\n".join(
        [
            text_line(label_x, y, [(f"{label}:", "yellow")], size=size, weight=600),
            text_line(value_x, y, [(value, value_color)], size=size),
        ]
    )


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
    width, height = 1120, 540
    username = str(profile["username"])
    roles = [str(value) for value in profile["roles"]]
    focus = [str(value) for value in profile["focus"]]
    contacts = [(str(label), str(value)) for label, value in profile["contacts"]]
    content: list[str] = [
        f'    <rect x="1" y="1" width="1118" height="538" rx="12" fill="{COLORS["background"]}" stroke="{COLORS["border"]}" stroke-width="2"/>',
        f'    <path d="M13 1 H1107 Q1119 1 1119 13 V46 H1 V13 Q1 1 13 1 Z" fill="{COLORS["title"]}"/>',
        f'    <path d="M1 46 H1119" stroke="{COLORS["border"]}" stroke-width="1"/>',
        '    <circle cx="23" cy="23" r="6" fill="#ff5f57"/>',
        '    <circle cx="43" cy="23" r="6" fill="#febc2e"/>',
        '    <circle cx="63" cy="23" r="6" fill="#28c840"/>',
        text_line(560, 28, [("~ | whoami", "muted")], size=12, anchor="middle"),
        text_line(32, 75, [("^^", "orange"), (" >>> ", "muted"), ("whoami", "bright")], size=13),
        f'    <path d="M454 92 V477" stroke="{COLORS["border"]}" stroke-width="1" opacity="0.55"/>',
        *portrait_lines(rows, x=50, y=105, size=10.6, line_height=10.55),
        text_line(490, 105, [(f"{username}@github", "bright")], size=15, weight=600),
        text_line(490, 124, [("--------------", "muted")], size=12),
        field_line(490, 592, 150, "Name", str(profile["name"])),
        field_line(490, 592, 174, "Role", roles[0]),
        text_line(592, 193, [(roles[1], "primary")], size=12.5),
        field_line(490, 592, 218, "Location", str(profile["location"])),
        field_line(490, 592, 242, "Focus", focus[0]),
        text_line(592, 261, [(focus[1], "primary")], size=12.5),
        text_line(490, 294, [("Contact", "bright")], size=13, weight=600),
        text_line(490, 312, [("-------", "muted")], size=12),
    ]
    for index, (label, value) in enumerate(contacts):
        content.append(field_line(490, 592, 336 + index * 20, label, value, value_color="blue"))
    content.extend(
        [
            f'    <path d="M28 482 H1092" stroke="{COLORS["border"]}" stroke-width="1" opacity="0.55"/>',
            text_line(32, 511, [("# ", "green"), (str(profile["statement"]), "muted")], size=10.5),
        ]
    )
    return svg_shell(
        width,
        height,
        "Terminal profile for Nikita Fomin",
        "A macOS terminal window with an ASCII portrait and concise public profile information.",
        content,
    )


def build_mobile(profile: dict[str, object], rows: list[str]) -> str:
    width, height = 600, 840
    username = str(profile["username"])
    roles = [str(value) for value in profile["roles"]]
    focus = [str(value) for value in profile["focus"]]
    contacts = [(str(label), str(value)) for label, value in profile["contacts"]]
    portrait_size = 9.2
    character_width = portrait_size * 0.60
    portrait_x = (width - len(rows[0]) * character_width) / 2
    content: list[str] = [
        f'    <rect x="1" y="1" width="598" height="838" rx="12" fill="{COLORS["background"]}" stroke="{COLORS["border"]}" stroke-width="2"/>',
        f'    <path d="M13 1 H587 Q599 1 599 13 V44 H1 V13 Q1 1 13 1 Z" fill="{COLORS["title"]}"/>',
        f'    <path d="M1 44 H599" stroke="{COLORS["border"]}" stroke-width="1"/>',
        '    <circle cx="21" cy="22" r="5.5" fill="#ff5f57"/>',
        '    <circle cx="39" cy="22" r="5.5" fill="#febc2e"/>',
        '    <circle cx="57" cy="22" r="5.5" fill="#28c840"/>',
        text_line(300, 27, [("~ | whoami", "muted")], size=11.5, anchor="middle"),
        text_line(26, 69, [("^^", "orange"), (" >>> ", "muted"), ("whoami", "bright")], size=12.5),
        *portrait_lines(rows, x=portrait_x, y=89, size=portrait_size, line_height=9.45),
    ]
    information_y = 89 + len(rows) * 9.45 + 10
    content.extend(
        [
            f'    <path d="M26 {information_y - 22:g} H574" stroke="{COLORS["border"]}" stroke-width="1" opacity="0.55"/>',
            text_line(30, information_y, [(f"{username}@github", "bright")], size=13.5, weight=600),
            text_line(30, information_y + 17, [("--------------", "muted")], size=11.5),
            field_line(30, 126, information_y + 39, "Name", str(profile["name"]), size=11.8),
            field_line(30, 126, information_y + 59, "Role", roles[0], size=11.8),
            text_line(126, information_y + 76, [(roles[1], "primary")], size=11.8),
            field_line(30, 126, information_y + 98, "Location", str(profile["location"]), size=11.8),
            field_line(30, 126, information_y + 118, "Focus", focus[0], size=11.8),
            text_line(126, information_y + 135, [(focus[1], "primary")], size=11.8),
            text_line(30, information_y + 161, [("Contact", "bright")], size=12.2, weight=600),
            text_line(30, information_y + 177, [("-------", "muted")], size=11.2),
        ]
    )
    for index, (label, value) in enumerate(contacts):
        content.append(
            field_line(30, 126, information_y + 197 + index * 18, label, value, value_color="blue", size=11.2)
        )
    content.extend(
        [
            f'    <path d="M26 786 H574" stroke="{COLORS["border"]}" stroke-width="1" opacity="0.55"/>',
            text_line(30, 807, [("# ", "green"), ("Building production ML systems, LLM applications, and AI agents", "muted")], size=9.4),
            text_line(44, 823, [("across modeling, data pipelines, deployment, and monitoring.", "muted")], size=9.4),
        ]
    )
    return svg_shell(
        width,
        height,
        "Terminal profile for Nikita Fomin",
        "A compact macOS terminal window with an ASCII portrait and public profile information.",
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
    print(f"wrote 1120×540 SVG to {args.output}")
    print(f"wrote 600×840 SVG to {args.mobile_output}")


if __name__ == "__main__":
    main()
