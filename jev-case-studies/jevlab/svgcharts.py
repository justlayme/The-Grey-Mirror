"""Tiny dependency-free SVG chart helpers for the reports.

Charts are themed through CSS custom properties (see report CSS), carry a native hover tooltip
(<title>) on every mark, direct labels where there are few series, and ship with an HTML table view.
Colour follows the entity, never its rank: the ENTITY map below is used by every chart.
"""
from __future__ import annotations

import html
import math

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 64, 24, 28, 56

ENTITY = {
    "jev": "var(--series-1)",
    "lexical": "var(--series-2)",
    "silence": "var(--series-3)",
    "behavioral": "var(--series-3)",
    "llm": "var(--series-4)",
    "stack": "var(--series-5)",
    "published": "var(--series-5)",
    "neutral": "var(--text-muted)",
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def _frame(title, x_label, y_label, body, legend, aria):
    return (f'<figure class="chart"><figcaption>{esc(title)}</figcaption>{legend}'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" preserveAspectRatio="xMidYMid meet">'
            f'{body}'
            f'<text class="axis-label" x="{PAD_L + (W - PAD_L - PAD_R) / 2}" y="{H - 12}" text-anchor="middle">{esc(x_label)}</text>'
            f'<text class="axis-label" transform="translate(16,{PAD_T + (H - PAD_T - PAD_B) / 2}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>'
            f'</svg></figure>')


def _axes(xs, ys, fx, fy, xfmt, yfmt, x_ticks=None, y_ticks=None):
    parts = []
    for t in (y_ticks if y_ticks is not None else _ticks(*ys)):
        y = fy(t)
        parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="tick" x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
    if xs is not None:
        for t in (x_ticks if x_ticks is not None else _ticks(*xs)):
            x = fx(t)
            parts.append(f'<text class="tick" x="{x:.1f}" y="{H - PAD_B + 18}" text-anchor="middle">{esc(xfmt(t))}</text>')
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{H - PAD_B}" y2="{H - PAD_B}"/>')
    return "".join(parts)


def _legend(items):
    """HTML legend row above the plot (wraps on narrow screens, never truncates)."""
    keys = "".join(f'<span class="lg"><span class="key{" dashed" if dashed else ""}" style="--c:{color}"></span>{esc(name)}</span>'
                   for name, color, dashed in items)
    return f'<div class="legend-row">{keys}</div>'


def line_chart(title, series, x_label, y_label, x_range, y_range, xfmt=str, yfmt=str, diagonal=False,
               vlines=(), hlines=(), markers=True, x_ticks=None, y_ticks=None, aria=None):
    """series: list of dicts {name, color, points:[(x,y,tooltip)], dashed?, label_last?}"""
    (x0, x1), (y0, y1) = x_range, y_range
    fx = lambda v: PAD_L + (v - x0) / (x1 - x0) * (W - PAD_L - PAD_R)  # noqa: E731
    fy = lambda v: H - PAD_B - (v - y0) / (y1 - y0) * (H - PAD_T - PAD_B)  # noqa: E731
    body = [_axes(x_range, y_range, fx, fy, xfmt, yfmt, x_ticks, y_ticks)]
    if diagonal:
        body.append(f'<line class="ref" x1="{fx(x0):.1f}" y1="{fy(y0):.1f}" x2="{fx(x1):.1f}" y2="{fy(y1):.1f}"/>')
    for xv, lab in vlines:
        body.append(f'<line class="ref" x1="{fx(xv):.1f}" x2="{fx(xv):.1f}" y1="{PAD_T}" y2="{H - PAD_B}"/>'
                    f'<text class="note" x="{fx(xv) + 4:.1f}" y="{PAD_T + 12}">{esc(lab)}</text>')
    for yv, lab in hlines:
        body.append(f'<line class="ref" x1="{PAD_L}" x2="{W - PAD_R}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>'
                    f'<text class="note" x="{PAD_L + 4}" y="{fy(yv) - 4:.1f}">{esc(lab)}</text>')
    for s in series:
        pts = [(x, y, t) for x, y, t in s["points"] if x is not None and y is not None
               and x0 <= x <= x1 and not (isinstance(y, float) and math.isnan(y))]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(min(max(y, y0), y1)):.1f}" for i, (x, y, _) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        body.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2" stroke-linejoin="round"{dash}/>')
        if markers:
            for x, y, tip in pts:
                body.append(f'<g class="pt"><circle cx="{fx(x):.1f}" cy="{fy(min(max(y, y0), y1)):.1f}" r="4" '
                            f'fill="{s["color"]}" stroke="var(--surface-1)" stroke-width="2"/>'
                            f'<circle cx="{fx(x):.1f}" cy="{fy(min(max(y, y0), y1)):.1f}" r="10" fill="transparent">'
                            f'<title>{esc(s["name"])}: {esc(tip)}</title></circle></g>')
    legend = _legend([(s["name"], s["color"], s.get("dashed")) for s in series])
    return _frame(title, x_label, y_label, "".join(body), legend, aria or title)


def bar_chart(title, categories, series, y_label, y_max, yfmt=str, value_fmt=str, aria=None, note=None):
    """Grouped vertical bars. series: list of {name, color, values:[..], tips:[..]} (None value = pending)."""
    n_cat, n_ser = len(categories), len(series)
    fy = lambda v: H - PAD_B - v / y_max * (H - PAD_T - PAD_B)  # noqa: E731
    body = [_axes(None, (0, y_max), None, fy, str, yfmt)]
    band = (W - PAD_L - PAD_R) / n_cat
    bw = min(34, (band - 24) / n_ser - 2)
    for ci, cat in enumerate(categories):
        cx = PAD_L + band * ci + band / 2
        start = cx - (n_ser * (bw + 2) - 2) / 2
        for si, s in enumerate(series):
            v = s["values"][ci]
            x = start + si * (bw + 2)
            tip = (s.get("tips") or [None] * n_cat)[ci]
            if v is None:
                body.append(f'<rect x="{x:.1f}" y="{fy(y_max * 0.04):.1f}" width="{bw:.1f}" height="{fy(0) - fy(y_max * 0.04):.1f}" '
                            f'class="pending"><title>{esc(s["name"])}: pending</title></rect>'
                            f'<text class="value" x="{x + bw / 2:.1f}" y="{fy(y_max * 0.04) - 4:.1f}" text-anchor="middle">…</text>')
                continue
            y = fy(v)
            h = max(fy(0) - y, 0.5)
            r = min(4, h / 2, bw / 2)
            # rounded data-end, square baseline
            path = (f"M{x:.1f},{fy(0):.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                    f"L{x + bw - r:.1f},{y:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} L{x + bw:.1f},{fy(0):.1f} Z")
            body.append(f'<path d="{path}" fill="{s["color"]}"><title>{esc(s["name"])} · {esc(cat)}: '
                        f'{esc(tip or value_fmt(v))}</title></path>'
                        f'<text class="value" x="{x + bw / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{esc(value_fmt(v))}</text>')
        body.append(f'<text class="tick" x="{cx:.1f}" y="{H - PAD_B + 18}" text-anchor="middle">{esc(cat)}</text>')
    if note:
        body.append(f'<text class="note" x="{PAD_L}" y="{PAD_T - 14}">{esc(note)}</text>')
    legend = _legend([(s["name"], s["color"], False) for s in series])
    return _frame(title, "", y_label, "".join(body), legend, aria or title)


def hbar_chart(title, rows, x_label, x_max, value_fmt=str, aria=None, ref=None, tick_fmt=None):
    """rows: list of (label, value|None, color, tooltip)."""
    h_row = 30
    height = PAD_T + len(rows) * h_row + 40
    lab_w = 220
    fx = lambda v: lab_w + v / x_max * (W - lab_w - 90)  # noqa: E731
    body = []
    for t in _ticks(0, x_max):
        body.append(f'<line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{PAD_T - 6}" y2="{height - 34}"/>'
                    f'<text class="tick" x="{fx(t):.1f}" y="{height - 18}" text-anchor="middle">{esc((tick_fmt or value_fmt)(t))}</text>')
    if ref is not None:
        body.append(f'<line class="ref" x1="{fx(ref[0]):.1f}" x2="{fx(ref[0]):.1f}" y1="{PAD_T - 10}" y2="{height - 34}"/>'
                    f'<text class="note" x="{fx(ref[0]) + 4:.1f}" y="{PAD_T - 14}">{esc(ref[1])}</text>')
    for i, (label, v, color, tip) in enumerate(rows):
        y = PAD_T + i * h_row
        body.append(f'<text class="tick" x="{lab_w - 10}" y="{y + 16}" text-anchor="end">{esc(label)}</text>')
        if v is None:
            body.append(f'<text class="value" x="{lab_w + 4}" y="{y + 16}">pending</text>')
            continue
        x2 = fx(v)
        r = min(4, (x2 - lab_w) / 2) if x2 > lab_w else 0
        path = (f"M{lab_w},{y + 4} L{x2 - r:.1f},{y + 4} Q{x2:.1f},{y + 4} {x2:.1f},{y + 4 + r:.1f} "
                f"L{x2:.1f},{y + 18 - r:.1f} Q{x2:.1f},{y + 18} {x2 - r:.1f},{y + 18} L{lab_w},{y + 18} Z")
        body.append(f'<path d="{path}" fill="{color}"><title>{esc(label)}: {esc(tip or value_fmt(v))}</title></path>'
                    f'<text class="value" x="{x2 + 6:.1f}" y="{y + 16}">{esc(value_fmt(v))}</text>')
    svg = (f'<figure class="chart"><figcaption>{esc(title)}</figcaption>'
           f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{esc(aria or title)}">{"".join(body)}'
           f'<text class="axis-label" x="{lab_w + (W - lab_w - 90) / 2}" y="{height - 2}" text-anchor="middle">{esc(x_label)}</text>'
           f'</svg></figure>')
    return svg


def table(headers, rows, caption=None) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c if isinstance(c, Raw) else esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def data_table(headers, rows, summary="Data table") -> str:
    return f'<details class="data"><summary>{esc(summary)}</summary>{table(headers, rows)}</details>'


class Raw(str):
    """Marks pre-escaped HTML for table cells."""
