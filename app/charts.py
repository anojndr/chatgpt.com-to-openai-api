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
import math
import logging

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("charts")

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_PALETTE = ["#10a37f", "#4f83cc", "#e67e22", "#8e6bbf", "#d9534f", "#2eb88a"]
_INK = "#111827"
_MUTE = "#6b7280"
_FAINT = "#9ca3af"
_GRID = "#e5e7eb"


class ChartError(ValueError):
    """Un-renderable chart spec."""


# Beyond a few hundred bars the picture is unreadable anyway; larger payloads
# only burn render time (PIL drawing is O(rows)).
MAX_CHART_ROWS = 400


def chart_from_payload(payload: dict) -> dict | None:
    """Normalized chart spec from a genui payload, or None if not a chart."""
    content = ((payload or {}).get("chart") or {}).get("content")
    if not isinstance(content, dict):
        return None
    series = [s for s in content.get("series") or [] if isinstance(s, dict) and s.get("dataKey")]
    data = [d for d in content.get("data") or [] if isinstance(d, dict)]
    if not series or not data or not content.get("xKey"):
        return None
    if len(data) > MAX_CHART_ROWS:
        return None
    return {
        "type": str(content.get("chartType") or "bar").lower(),
        "meta": content.get("meta") or {},
        "x_key": content["xKey"],
        "series": series,
        "data": data,
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"{FONT_DIR}/{name}", size)
    except OSError:  # pragma: no cover - font dir missing on odd systems
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, limit: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in str(text or "").split():
        cand = f"{cur} {word}".strip()
        if not cur or draw.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
            if len(lines) == limit:
                break
    if cur and len(lines) < limit:
        lines.append(cur)
    if len(lines) == limit and lines:
        # mark that content did not fit
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


def _fmt_value(v, fmt: str | None) -> str:
    if fmt == "percent":
        return f"{v:g}%"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return f"{v:g}"


def _nice_ticks(ymax: float) -> tuple[float, float]:
    """Return (nice_max, step) covering [0, ymax] with ~4-5 ticks."""
    if ymax <= 0:
        return 1.0, 0.25
    raw = ymax / 4
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm <= 1 else 2 if norm <= 2 else 2.5 if norm <= 2.5
            else 5 if norm <= 5 else 10) * mag
    nice = math.ceil(ymax / step) * step
    return nice, step


def _vtext(text: str, font, fill) -> Image.Image:
    tmp = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(tmp)
    box = d.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (box[2] - box[0] + 4, box[3] - box[1] + 4), "white")
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, font=font, fill=fill)
    return img.rotate(90, expand=True)


def render_chart_png(spec: dict, width: int = 1080) -> bytes:
    if spec["type"] not in ("bar", "column"):
        raise ChartError(f"unsupported chartType {spec['type']!r}")

    meta = spec["meta"]
    x_key, series, data = spec["x_key"], spec["series"], spec["data"]

    title_f, desc_f, tick_f, val_f, foot_f, axis_f = (
        _font(31, True), _font(19), _font(17), _font(17, True), _font(15), _font(16))

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    ML, MR, gap = 96, 40, 14
    plot_w = width - ML - MR

    title_lines = _wrap(probe, meta.get("title") or "", title_f, width - ML - MR, 2)
    desc_lines = _wrap(probe, meta.get("description") or "", desc_f, width - ML - MR, 2)
    foot_lines = _wrap(probe, meta.get("footer") or "", foot_f, width - MR - 24, 2)

    labels = [" ".join(str(d.get(x_key, "")).split()) or "—" for d in data]
    lab_limit = max(plot_w // max(len(data), 1) - 8, 60)
    lab_lines = [_wrap(probe, lb, tick_f, lab_limit, 2) or ["—"] for lb in labels]

    values: list[list[float]] = []
    vmax = 0.0
    for d in data:
        row = []
        for s in series:
            v = d.get(s["dataKey"], 0)
            v = float(v) if isinstance(v, (int, float)) else 0.0
            row.append(v)
            vmax = max(vmax, v)
        values.append(row)

    top = 26 + len(title_lines) * 40 + (len(desc_lines) * 25 + 12 if desc_lines else 0)
    bottom = max(len(l) for l in lab_lines) * 21 + 10 + (26 if foot_lines else 14)
    height = top + 420 + bottom

    img = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(img)

    y = 26
    for line in title_lines:
        dr.text((ML, y), line, font=title_f, fill=_INK)
        y += 40
    y += 4
    for line in desc_lines:
        dr.text((ML, y), line, font=desc_f, fill=_MUTE)
        y += 25

    p_top, p_bot = top + 16, height - bottom
    p_h = p_bot - p_top
    nice_max, step = _nice_ticks(vmax)
    if nice_max <= 0:
        nice_max, step = 1.0, 0.25

    def vy(v: float) -> int:
        return int(p_bot - (v / nice_max) * p_h)

    # gridlines + y tick labels (+ rotated series axis label, first series wins)
    t = 0.0
    while t <= nice_max + step / 2:
        yy = vy(t)
        dr.line([(ML, yy), (width - MR, yy)], fill=_GRID, width=1)
        lbl = _fmt_value(int(t) if float(t).is_integer() else round(t, 4), None)
        tw = dr.textlength(lbl, font=tick_f)
        dr.text((ML - tw - 8, yy - 9), lbl, font=tick_f, fill=_MUTE)
        t += step
    axis_label = next((s.get("axisLabel") for s in series if s.get("axisLabel")), "")
    if axis_label:
        vt = _vtext(axis_label, axis_f, _MUTE)
        img.paste(vt, (max(ML - vt.width - 34, 4), p_top + (p_h - vt.height) // 2))

    # grouped bars + value tags + x labels
    n_groups, n_series = len(data), max(len(series), 1)
    slot = plot_w / n_groups
    bar_w = min(slot * 0.62 / n_series, 140)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_series)]
    for gi, (label, row) in enumerate(zip(labels, values)):
        gx = ML + gi * slot + (slot - bar_w * n_series) / 2
        for si, v in enumerate(row):
            x0 = gx + si * bar_w
            bh = max(p_bot - vy(v), 2)
            dr.rectangle([x0, p_bot - bh, x0 + bar_w, p_bot], fill=colors[si])
            tag = _fmt_value(v, series[si].get("valueFormat"))
            tw = dr.textlength(tag, font=val_f)
            ty = max(p_bot - bh - 21, 2)
            dr.text((x0 + (bar_w - tw) / 2, ty), tag, font=val_f, fill=_INK)
        lx = ML + gi * slot + slot / 2
        lines = lab_lines[gi]
        ly = p_bot + 8
        for line in lines:
            lw = dr.textlength(line, font=tick_f)
            dr.text((lx - lw / 2, ly), line, font=tick_f, fill="#4b5563")
            ly += 21

    # legend under the x labels (only when multiple series)
    if n_series > 1:
        ly = p_bot + max(len(l) for l in lab_lines) * 21 + 14
        lx = ML
        for s, color in zip(series, colors):
            dr.rectangle([lx, ly + 3, lx + 14, ly + 17], fill=color)
            name = " ".join(str(s.get("label") or s["dataKey"]).split())
            dr.text((lx + 20, ly), name, font=tick_f, fill=_MUTE)
            lx += 34 + dr.textlength(name, font=tick_f)
            if lx > width - MR - 80:
                break

    fy = height - 14 - len(foot_lines) * 20
    for line in foot_lines:
        dr.text((24, fy), line, font=foot_f, fill=_FAINT)
        fy += 20

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
