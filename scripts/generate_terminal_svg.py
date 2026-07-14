#!/usr/bin/env python3
"""Build the self-contained SVG terminal card from assets/portrait.txt."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTRAIT = REPO_ROOT / "assets" / "portrait.txt"
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "profile-terminal.svg"
DEFAULT_MOBILE_OUTPUT = REPO_ROOT / "assets" / "profile-terminal-mobile.svg"

WIDTH = 1100
HEIGHT = 510
MOBILE_WIDTH = 600
MOBILE_HEIGHT = 920
FONT_STACK = (
    "ui-monospace, SFMono-Regular, SF Mono, Menlo, Monaco, Consolas, "
    "Liberation Mono, monospace"
)


def text_node(
    x: int,
    y: int,
    value: str,
    *,
    fill: str = "#e6edf3",
    size: float = 12.5,
    weight: int = 400,
    anchor: str | None = None,
    opacity: float | None = None,
) -> str:
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'fill="{fill}"',
        f'font-size="{size}"',
        f'font-weight="{weight}"',
    ]
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"    <text {' '.join(attrs)}>{escape(value)}</text>"


def label(y: int, value: str) -> str:
    return text_node(474, y, value, fill="#d6a06c", size=11.5, weight=600)


def value(y: int, content: str) -> str:
    return text_node(582, y, content, fill="#e6edf3", size=12.5)


def read_portrait(path: Path) -> list[str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError("portrait is empty")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"portrait rows have inconsistent widths: {sorted(widths)}")
    width = widths.pop()
    if not 36 <= width <= 44:
        raise ValueError(f"portrait width {width} is outside 36..44")
    if not 27 <= len(rows) <= 34:
        raise ValueError(f"portrait height {len(rows)} is outside 27..34")
    return rows


def portrait_text(
    rows: list[str],
    *,
    x: int = 84,
    y: int = 92,
    size: float = 10.5,
    line_height: float = 11.15,
) -> str:
    tspans: list[str] = []
    for index, row in enumerate(rows):
        dy = "0" if index == 0 else str(line_height)
        tspans.append(f'      <tspan x="{x}" dy="{dy}">{escape(row)}</tspan>')
    return "\n".join(
        [
            f'    <text x="{x}" y="{y}" fill="#b9c4d0" font-size="{size}" '
            'font-weight="500" letter-spacing="0.15" xml:space="preserve">',
            *tspans,
            "    </text>",
        ]
    )


def build_svg(rows: list[str]) -> str:
    information = [
        text_node(474, 98, "$", fill="#10b981", size=16, weight=700),
        text_node(494, 98, "nikita@github", size=16, weight=650),
        text_node(1048, 98, "profile", fill="#7f8b99", size=10.5, anchor="end"),
        '    <path d="M474 116 H1050" stroke="#2a3543" stroke-width="1"/>',
        label(141, "Name"),
        value(141, "Nikita Fomin"),
        label(165, "Role"),
        value(165, "AI / ML Engineer"),
        value(183, "Data Scientist"),
        label(207, "Location"),
        value(207, "Paris, France"),
        label(231, "Focus"),
        value(231, "Production ML · LLM applications"),
        value(249, "AI agents · NLP · deep learning"),
        label(273, "Experience"),
        value(273, "Appodeal · inDrive"),
        value(291, "Yandex · Medialogia"),
        '    <path d="M474 307 H1050" stroke="#2a3543" stroke-width="1"/>',
        label(329, "Languages"),
        value(329, "Python · SQL"),
        label(350, "ML / DL"),
        value(350, "PyTorch · scikit-learn · CatBoost"),
        label(371, "AI"),
        value(371, "LLMs · LangGraph · RAG · AI agents"),
        label(392, "Data"),
        value(392, "Spark · Databricks · Kafka"),
        label(413, "MLOps"),
        value(413, "MLflow · Airflow · Kubeflow"),
    ]

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title desc">',
            "  <title id=\"title\">Nikita Fomin — AI and Machine Learning Engineer</title>",
            "  <desc id=\"desc\">A compact terminal profile card with an ASCII portrait and professional summary.</desc>",
            "  <defs>",
            '    <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">',
            '      <stop offset="0" stop-color="#0b0f14"/>',
            '      <stop offset="1" stop-color="#101722"/>',
            "    </linearGradient>",
            '    <filter id="shadow" x="-10%" y="-10%" width="120%" height="125%">',
            '      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000000" flood-opacity="0.30"/>',
            "    </filter>",
            "  </defs>",
            f'  <g font-family="{FONT_STACK}">',
            '    <rect x="10" y="10" width="1080" height="490" rx="18" fill="url(#background)" '
            'stroke="#2a3543" stroke-width="1.5" filter="url(#shadow)"/>',
            '    <path d="M10 58 H1090" stroke="#2a3543" stroke-width="1"/>',
            '    <circle cx="34" cy="34" r="5" fill="#d6a06c"/>',
            '    <circle cx="52" cy="34" r="5" fill="#7f8b99"/>',
            '    <circle cx="70" cy="34" r="5" fill="#10b981"/>',
            text_node(92, 39, "fomin-n / profile", fill="#9aa7b5", size=12, weight=600),
            text_node(1066, 39, "AI · ML · DATA", fill="#667281", size=10, anchor="end"),
            '    <rect x="24" y="72" width="414" height="360" rx="14" fill="#121820" '
            'stroke="#202b38" stroke-width="1"/>',
            '    <rect x="452" y="72" width="624" height="360" rx="14" fill="#151d27" '
            'stroke="#2a3543" stroke-width="1"/>',
            portrait_text(rows),
            *information,
            '    <rect x="24" y="448" width="1052" height="34" rx="10" fill="#121820" '
            'stroke="#202b38" stroke-width="1"/>',
            text_node(40, 470, "›", fill="#10b981", size=14, weight=700),
            text_node(
                58,
                470,
                "Building production ML systems, LLM applications, and AI agents across modeling, data pipelines, deployment, and monitoring.",
                fill="#9aa7b5",
                size=10.7,
            ),
            "  </g>",
            "</svg>",
            "",
        ]
    )


def build_mobile_svg(rows: list[str]) -> str:
    def mobile_label(y: int, content: str) -> str:
        return text_node(48, y, content, fill="#d6a06c", size=12, weight=600)

    def mobile_value(y: int, content: str) -> str:
        return text_node(158, y, content, fill="#e6edf3", size=13.2)

    information = [
        text_node(48, 454, "$", fill="#10b981", size=16, weight=700),
        text_node(68, 454, "nikita@github", size=16, weight=650),
        text_node(552, 454, "profile", fill="#7f8b99", size=10.5, anchor="end"),
        '    <path d="M48 472 H552" stroke="#2a3543" stroke-width="1"/>',
        mobile_label(498, "Name"),
        mobile_value(498, "Nikita Fomin"),
        mobile_label(523, "Role"),
        mobile_value(523, "AI / ML Engineer"),
        mobile_value(542, "Data Scientist"),
        mobile_label(567, "Location"),
        mobile_value(567, "Paris, France"),
        mobile_label(592, "Focus"),
        mobile_value(592, "Production ML · LLM applications"),
        mobile_value(611, "AI agents · NLP · deep learning"),
        mobile_label(636, "Experience"),
        mobile_value(636, "Appodeal · inDrive"),
        mobile_value(655, "Yandex · Medialogia"),
        '    <path d="M48 674 H552" stroke="#2a3543" stroke-width="1"/>',
        mobile_label(700, "Languages"),
        mobile_value(700, "Python · SQL"),
        mobile_label(724, "ML / DL"),
        mobile_value(724, "PyTorch · scikit-learn · CatBoost"),
        mobile_label(748, "AI"),
        mobile_value(748, "LLMs · LangGraph · RAG · AI agents"),
        mobile_label(772, "Data"),
        mobile_value(772, "Spark · Databricks · Kafka"),
        mobile_label(796, "MLOps"),
        mobile_value(796, "MLflow · Airflow · Kubeflow"),
    ]

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MOBILE_WIDTH} {MOBILE_HEIGHT}" '
            f'width="{MOBILE_WIDTH}" height="{MOBILE_HEIGHT}" role="img" aria-labelledby="title desc">',
            "  <title id=\"title\">Nikita Fomin — AI and Machine Learning Engineer</title>",
            "  <desc id=\"desc\">A stacked terminal profile card with an ASCII portrait and professional summary.</desc>",
            "  <defs>",
            '    <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">',
            '      <stop offset="0" stop-color="#0b0f14"/>',
            '      <stop offset="1" stop-color="#101722"/>',
            "    </linearGradient>",
            '    <filter id="shadow" x="-10%" y="-10%" width="120%" height="115%">',
            '      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000000" flood-opacity="0.30"/>',
            "    </filter>",
            "  </defs>",
            f'  <g font-family="{FONT_STACK}">',
            '    <rect x="10" y="10" width="580" height="900" rx="18" fill="url(#background)" '
            'stroke="#2a3543" stroke-width="1.5" filter="url(#shadow)"/>',
            '    <path d="M10 58 H590" stroke="#2a3543" stroke-width="1"/>',
            '    <circle cx="34" cy="34" r="5" fill="#d6a06c"/>',
            '    <circle cx="52" cy="34" r="5" fill="#7f8b99"/>',
            '    <circle cx="70" cy="34" r="5" fill="#10b981"/>',
            text_node(92, 39, "fomin-n / profile", fill="#9aa7b5", size=12, weight=600),
            text_node(566, 39, "AI · ML · DATA", fill="#667281", size=10, anchor="end"),
            '    <rect x="24" y="72" width="552" height="336" rx="14" fill="#121820" '
            'stroke="#202b38" stroke-width="1"/>',
            portrait_text(rows, x=166, y=87, size=10.2, line_height=10.35),
            '    <rect x="24" y="424" width="552" height="392" rx="14" fill="#151d27" '
            'stroke="#2a3543" stroke-width="1"/>',
            *information,
            '    <rect x="24" y="832" width="552" height="60" rx="10" fill="#121820" '
            'stroke="#202b38" stroke-width="1"/>',
            text_node(40, 857, "›", fill="#10b981", size=14, weight=700),
            text_node(
                58,
                856,
                "Building production ML systems, LLM applications, and AI agents",
                fill="#9aa7b5",
                size=10.4,
            ),
            text_node(
                58,
                875,
                "across modeling, data pipelines, deployment, and monitoring.",
                fill="#9aa7b5",
                size=10.4,
            ),
            "  </g>",
            "</svg>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portrait", type=Path, default=DEFAULT_PORTRAIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mobile-output", type=Path, default=DEFAULT_MOBILE_OUTPUT)
    args = parser.parse_args()

    rows = read_portrait(args.portrait)
    svg = build_svg(rows)
    mobile_svg = build_mobile_svg(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    args.mobile_output.parent.mkdir(parents=True, exist_ok=True)
    args.mobile_output.write_text(mobile_svg, encoding="utf-8")
    print(f"wrote {WIDTH}×{HEIGHT} SVG to {args.output}")
    print(f"wrote {MOBILE_WIDTH}×{MOBILE_HEIGHT} SVG to {args.mobile_output}")


if __name__ == "__main__":
    main()
