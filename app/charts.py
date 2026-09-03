# Copyright 2026 chatgpt-to-openai-api contributors.
"""Render ChatGPT genui chart specs to PNG bytes (PIL-drawn bar charts).

The genui widget payload looks like:
    {"chart": {"content": {
        "chartType": "bar",
        "meta": {"title": ..., "description": ..., "footer": ...},
        "xKey": "city",
        "series": [{"dataKey": "rate", "label": "...", "axisLabel": "...",
                    "valueFormat": "raw"}],
        "data": [{"city": "Fresno, CA", "rate": 735.5}, ...]}}}

Only bar/column charts are rendered -- anything else raises ChartError so the
caller can degrade to stripping instead of shipping a misleading picture.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from typing import TypeGuard

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("charts")

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_PALETTE = ["#10a37f", "#4f83cc", "#e67e22", "#8e6bbf", "#d9534f", "#2eb88a"]
_INK = "#111827"
_MUTE = "#6b7280"
_FAINT = "#9ca3af"
_GRID = "#e5e7eb"
_X_LABEL_COLOR = "#4b5563"

_ChartFont = ImageFont.FreeTypeFont | ImageFont.ImageFont

_LEFT_MARGIN = 96
_RIGHT_MARGIN = 40
_PROBE_SIZE = 8
_PLOT_HEIGHT = 420
_BAR_WIDTH_CAP = 140
_TITLE_MAX_LINES = 2
_DESC_MAX_LINES = 2
_FOOT_MAX_LINES = 2
_LABEL_MAX_LINES = 2
_LABEL_MIN_WIDTH = 60
_FOOT_RIGHT_PAD = 24
_VTEXT_PADDING = 4

_TICK_TARGET_COUNT = 4
_NICE_MULTIPLIERS = (1.0, 2.0, 2.5, 5.0, 10.0)
_FALLBACK_TICK_MAX = 1.0
_FALLBACK_TICK_STEP = 0.25


class ChartError(ValueError):
    """Un-renderable chart spec."""

    def __init__(self, message: str = "un-renderable chart spec") -> None:
        """Store the reason the spec cannot be rendered."""
        super().__init__(message)


# Beyond a few hundred bars the picture is unreadable anyway; larger payloads
# only burn render time (PIL drawing is O(rows)).
MAX_CHART_ROWS = 400


def _is_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Check whether the value is a string-keyed mapping."""
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    """Check whether the value is a list."""
    return isinstance(value, list)


def chart_from_payload(payload: dict[str, object]) -> dict[str, object] | None:
    """Build a normalized chart spec from a genui payload, if it holds one."""
    chart_raw = payload.get("chart")
    if not _is_dict(chart_raw):
        return None
    content = chart_raw.get("content")
    if not _is_dict(content):
        return None
    series_raw = content.get("series")
    series_items = series_raw if _is_list(series_raw) else []
    series = [
        entry for entry in series_items if _is_dict(entry) and entry.get("dataKey")
    ]
    data_raw = content.get("data")
    data_items = data_raw if _is_list(data_raw) else []
    data = [row for row in data_items if _is_dict(row)]
    if not series or not data or not content.get("xKey"):
        return None
    if len(data) > MAX_CHART_ROWS:
        return None
    chart_type = content.get("chartType") or "bar"
    return {
        "type": str(chart_type).lower(),
        "meta": content.get("meta") or {},
        "x_key": content["xKey"],
        "series": series,
        "data": data,
    }


def _font(size: int, *, bold: bool = False) -> _ChartFont:
    """Load a DejaVu font, falling back to PIL's default bitmap font."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"{FONT_DIR}/{name}", size)
    except OSError:
        return ImageFont.load_default()


def _wrap(
    draw: ImageDraw.ImageDraw,
    text: object,
    font: _ChartFont,
    max_w: int,
    limit: int,
) -> list[str]:
    """Wrap text to at most limit lines fitting max_w pixels."""
    lines: list[str] = []
    current = ""
    for word in str(text or "").split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_w:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) == limit:
                break
    if current and len(lines) < limit:
        lines.append(current)
    if len(lines) == limit and lines:
        # mark that content did not fit
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


def _fmt_value(value: float, fmt: str | None) -> str:
    """Format one chart value, honoring the series value format."""
    if fmt == "percent":
        return f"{value:g}%"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _nice_ticks(ymax: float) -> tuple[float, float]:
    """Return (nice_max, step) covering [0, ymax] with ~4-5 ticks."""
    if ymax <= 0:
        return _FALLBACK_TICK_MAX, _FALLBACK_TICK_STEP
    raw = ymax / _TICK_TARGET_COUNT
    magnitude = 10.0 ** math.floor(math.log10(raw))
    scaled = raw / magnitude
    multiplier = _NICE_MULTIPLIERS[-1]
    for candidate in _NICE_MULTIPLIERS:
        if scaled <= candidate:
            multiplier = candidate
            break
    step = multiplier * magnitude
    nice = math.ceil(ymax / step) * step
    return float(nice), float(step)


def _vtext(text: str, font: _ChartFont, fill: str) -> Image.Image:
    """Render text rotated 90 degrees for the vertical axis label."""
    tmp = Image.new("RGB", (_PROBE_SIZE, _PROBE_SIZE))
    probe = ImageDraw.Draw(tmp)
    box = probe.textbbox((0, 0), text, font=font)
    size = (
        int(box[2] - box[0] + _VTEXT_PADDING),
        int(box[3] - box[1] + _VTEXT_PADDING),
    )
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, font=font, fill=fill)
    return img.rotate(90, expand=True)


@dataclass
class _ChartFonts:
    """Loaded typefaces for one chart render."""

    title: _ChartFont
    description: _ChartFont
    tick: _ChartFont
    value: _ChartFont
    footer: _ChartFont
    axis: _ChartFont


@dataclass
class _ChartSpec:
    """Validated chart render inputs."""

    chart_type: str
    meta: dict[str, object]
    x_key: str
    series: list[dict[str, object]]
    data: list[dict[str, object]]


@dataclass
class _ChartLayout:
    """Pixel geometry and wrapped text for one chart render."""

    width: int
    left_margin: int
    right_margin: int
    plot_width: int
    top: int
    bottom: int
    height: int
    title_lines: list[str]
    desc_lines: list[str]
    foot_lines: list[str]
    labels: list[str]
    label_lines: list[list[str]]
    max_label_lines: int


def _load_chart_fonts() -> _ChartFonts:
    """Load every chart typeface."""
    return _ChartFonts(
        title=_font(31, bold=True),
        description=_font(19),
        tick=_font(17),
        value=_font(17, bold=True),
        footer=_font(15),
        axis=_font(16),
    )


def _coerce_chart_spec(spec: dict[str, object]) -> _ChartSpec:
    """Validate a normalized chart spec, raising ChartError when unusable."""
    chart_type_raw = spec.get("type")
    chart_type = chart_type_raw if isinstance(chart_type_raw, str) else ""
    if chart_type not in ("bar", "column"):
        msg = f"unsupported chartType {chart_type!r}"
        raise ChartError(msg)
    meta_raw = spec.get("meta")
    if not _is_dict(meta_raw):
        msg = "chart spec meta must be an object"
        raise ChartError(msg)
    x_key_raw = spec.get("x_key")
    if not isinstance(x_key_raw, str) or not x_key_raw:
        msg = "chart spec x_key must be a non-empty string"
        raise ChartError(msg)
    series_raw = spec.get("series")
    if not _is_list(series_raw):
        msg = "chart spec series must be a list"
        raise ChartError(msg)
    series = [entry for entry in series_raw if _is_dict(entry)]
    for entry in series:
        key_raw = entry.get("dataKey")
        if not isinstance(key_raw, str) or not key_raw:
            msg = "chart spec series entries must carry a non-empty dataKey"
            raise ChartError(msg)
    data_raw = spec.get("data")
    if not _is_list(data_raw):
        msg = "chart spec data must be a list"
        raise ChartError(msg)
    data = [row for row in data_raw if _is_dict(row)]
    if not data:
        msg = "chart spec data must not be empty"
        raise ChartError(msg)
    return _ChartSpec(
        chart_type=chart_type,
        meta=meta_raw,
        x_key=x_key_raw,
        series=series,
        data=data,
    )


def _prepare_layout(
    meta: dict[str, object],
    x_key: str,
    data: list[dict[str, object]],
    width: int,
    fonts: _ChartFonts,
) -> _ChartLayout:
    """Wrap chart text and compute pixel geometry."""
    probe = ImageDraw.Draw(Image.new("RGB", (_PROBE_SIZE, _PROBE_SIZE)))
    left_margin = _LEFT_MARGIN
    right_margin = _RIGHT_MARGIN
    plot_width = width - left_margin - right_margin
    title_lines = _wrap(
        probe, meta.get("title") or "", fonts.title, plot_width, _TITLE_MAX_LINES
    )
    desc_lines = _wrap(
        probe,
        meta.get("description") or "",
        fonts.description,
        plot_width,
        _DESC_MAX_LINES,
    )
    foot_lines = _wrap(
        probe,
        meta.get("footer") or "",
        fonts.footer,
        width - right_margin - _FOOT_RIGHT_PAD,
        _FOOT_MAX_LINES,
    )
    labels = [" ".join(str(row.get(x_key, "")).split()) or "—" for row in data]
    label_width = max(plot_width // max(len(data), 1) - 8, _LABEL_MIN_WIDTH)
    label_lines = [
        _wrap(probe, label, fonts.tick, label_width, _LABEL_MAX_LINES) or ["—"]
        for label in labels
    ]
    max_label_lines = max(len(lines) for lines in label_lines)
    title_height = len(title_lines) * 40
    desc_height = len(desc_lines) * 25 + 12 if desc_lines else 0
    top = 26 + title_height + desc_height
    bottom = max_label_lines * 21 + 10 + (26 if foot_lines else 14)
    height = top + _PLOT_HEIGHT + bottom
    return _ChartLayout(
        width=width,
        left_margin=left_margin,
        right_margin=right_margin,
        plot_width=plot_width,
        top=top,
        bottom=bottom,
        height=height,
        title_lines=title_lines,
        desc_lines=desc_lines,
        foot_lines=foot_lines,
        labels=labels,
        label_lines=label_lines,
        max_label_lines=max_label_lines,
    )


def _chart_values(
    data: list[dict[str, object]],
    series: list[dict[str, object]],
) -> tuple[list[list[float]], float]:
    """Extract per-series float rows and the overall maximum."""
    values: list[list[float]] = []
    vmax = 0.0
    for row_data in data:
        row: list[float] = []
        for entry in series:
            key_raw = entry.get("dataKey", "")
            key = key_raw if isinstance(key_raw, str) else ""
            raw_value = row_data.get(key, 0)
            number = float(raw_value) if isinstance(raw_value, (int, float)) else 0.0
            row.append(number)
            vmax = max(vmax, number)
        values.append(row)
    return values, vmax


class _ChartRenderer:
    """Draw one validated chart spec onto a PIL image."""

    def __init__(self, spec: _ChartSpec, width: int) -> None:
        """Load fonts, wrap text, and scale values for the render."""
        self._fonts = _load_chart_fonts()
        self._layout = _prepare_layout(
            spec.meta, spec.x_key, spec.data, width, self._fonts
        )
        self._series = spec.series
        self._values, vmax = _chart_values(spec.data, spec.series)
        nice_max, step = _nice_ticks(vmax)
        if nice_max <= 0:
            nice_max, step = _FALLBACK_TICK_MAX, _FALLBACK_TICK_STEP
        self._nice_max = nice_max
        self._step = step
        self._plot_top = self._layout.top + 16
        self._plot_bottom = self._layout.height - self._layout.bottom
        self._plot_height = self._plot_bottom - self._plot_top
        self._img = Image.new("RGB", (width, self._layout.height), "white")
        self._draw = ImageDraw.Draw(self._img)

    def render(self) -> bytes:
        """Draw every chart element, returning PNG image bytes."""
        self._draw_header()
        self._draw_grid()
        self._draw_bars()
        self._draw_footer()
        out = io.BytesIO()
        self._img.save(out, format="PNG", optimize=True)
        return out.getvalue()

    def _value_y(self, value: float) -> int:
        """Map a data value to its vertical pixel position."""
        ratio = value / self._nice_max
        return int(self._plot_bottom - ratio * self._plot_height)

    def _draw_header(self) -> None:
        """Draw the title and description lines."""
        layout = self._layout
        fonts = self._fonts
        draw = self._draw
        y = 26
        for line in layout.title_lines:
            draw.text((layout.left_margin, y), line, font=fonts.title, fill=_INK)
            y += 40
        y += 4
        for line in layout.desc_lines:
            draw.text((layout.left_margin, y), line, font=fonts.description, fill=_MUTE)
            y += 25

    def _draw_grid(self) -> None:
        """Draw gridlines, y tick labels, and the rotated axis label."""
        layout = self._layout
        fonts = self._fonts
        draw = self._draw
        tick = 0.0
        while tick <= self._nice_max + self._step / 2:
            yy = self._value_y(tick)
            draw.line(
                [(layout.left_margin, yy), (layout.width - layout.right_margin, yy)],
                fill=_GRID,
                width=1,
            )
            lbl = _fmt_value(
                int(tick) if float(tick).is_integer() else round(tick, 4), None
            )
            label_w = draw.textlength(lbl, font=fonts.tick)
            draw.text(
                (layout.left_margin - label_w - 8, yy - 9),
                lbl,
                font=fonts.tick,
                fill=_MUTE,
            )
            tick += self._step
        axis_raw = next(
            (
                entry.get("axisLabel")
                for entry in self._series
                if entry.get("axisLabel")
            ),
            "",
        )
        axis_label = axis_raw if isinstance(axis_raw, str) else ""
        if axis_label:
            vertical = _vtext(axis_label, fonts.axis, _MUTE)
            draw_x = max(
                layout.left_margin - vertical.width - 34,
                4,
            )
            draw_y = self._plot_top + (self._plot_height - vertical.height) // 2
            self._img.paste(vertical, (draw_x, draw_y))

    def _draw_bars(self) -> None:
        """Draw grouped bars with value tags, x labels, and the legend."""
        layout = self._layout
        fonts = self._fonts
        draw = self._draw
        n_groups = len(self._values)
        n_series = max(len(self._series), 1)
        slot = layout.plot_width / n_groups
        bar_w = min(slot * 0.62 / n_series, _BAR_WIDTH_CAP)
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_series)]
        for gi, (_label, row) in enumerate(
            zip(layout.labels, self._values, strict=True)
        ):
            group_x = layout.left_margin + gi * slot + (slot - bar_w * n_series) / 2
            for si, value in enumerate(row):
                x0 = group_x + si * bar_w
                bar_h = max(self._plot_bottom - self._value_y(value), 2)
                draw.rectangle(
                    [x0, self._plot_bottom - bar_h, x0 + bar_w, self._plot_bottom],
                    fill=colors[si],
                )
                tag = _fmt_value(value, self._series_value_format(si))
                tag_w = draw.textlength(tag, font=fonts.value)
                tag_y = max(self._plot_bottom - bar_h - 21, 2)
                draw.text(
                    (x0 + (bar_w - tag_w) / 2, tag_y),
                    tag,
                    font=fonts.value,
                    fill=_INK,
                )
            center_x = layout.left_margin + gi * slot + slot / 2
            text_y = self._plot_bottom + 8
            for line in layout.label_lines[gi]:
                line_w = draw.textlength(line, font=fonts.tick)
                draw.text(
                    (center_x - line_w / 2, text_y),
                    line,
                    font=fonts.tick,
                    fill=_X_LABEL_COLOR,
                )
                text_y += 21
        if n_series > 1:
            legend_y = self._plot_bottom + layout.max_label_lines * 21 + 14
            legend_x = layout.left_margin
            for entry, color in zip(self._series, colors, strict=True):
                draw.rectangle(
                    [legend_x, legend_y + 3, legend_x + 14, legend_y + 17],
                    fill=color,
                )
                name_raw = entry.get("label") or entry.get("dataKey")
                name = " ".join(str(name_raw).split())
                draw.text((legend_x + 20, legend_y), name, font=fonts.tick, fill=_MUTE)
                legend_x += 34 + draw.textlength(name, font=fonts.tick)
                if legend_x > layout.width - layout.right_margin - 80:
                    break

    def _series_value_format(self, index: int) -> str | None:
        """Return the value format name for one series, if configured."""
        raw = self._series[index].get("valueFormat")
        return raw if isinstance(raw, str) else None

    def _draw_footer(self) -> None:
        """Draw the footer lines."""
        layout = self._layout
        draw = self._draw
        y = layout.height - 14 - len(layout.foot_lines) * 20
        for line in layout.foot_lines:
            draw.text((24, y), line, font=self._fonts.footer, fill=_FAINT)
            y += 20


def render_chart_png(spec: dict[str, object], width: int = 1080) -> bytes:
    """Render a normalized chart spec to PNG image bytes."""
    return _ChartRenderer(_coerce_chart_spec(spec), width).render()
