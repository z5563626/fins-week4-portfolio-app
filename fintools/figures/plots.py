"""Reusable chart builders for finance coursework."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter

from .theme import FINS_COLORS, FT_COLOR_CYCLE, FT_COLORS, figure_style


@dataclass(frozen=True)
class RecessionWindow:
    """One NBER recession shading window."""

    label: str
    start: str
    end: str


NBER_RECESSIONS = [
    RecessionWindow("1929-1933", "1929-09-01", "1933-03-31"),
    RecessionWindow("1937-1938", "1937-06-01", "1938-06-30"),
    RecessionWindow("1945", "1945-03-01", "1945-10-31"),
    RecessionWindow("1948-1949", "1948-12-01", "1949-10-31"),
    RecessionWindow("1953-1954", "1953-08-01", "1954-05-31"),
    RecessionWindow("1957-1958", "1957-09-01", "1958-04-30"),
    RecessionWindow("1960-1961", "1960-05-01", "1961-02-28"),
    RecessionWindow("1969-1970", "1970-01-01", "1970-11-30"),
    RecessionWindow("1973-1975", "1973-12-01", "1975-03-31"),
    RecessionWindow("1980", "1980-02-01", "1980-07-31"),
    RecessionWindow("1981-1982", "1981-08-01", "1982-11-30"),
    RecessionWindow("1990-1991", "1990-08-01", "1991-03-31"),
    RecessionWindow("2001", "2001-04-01", "2001-11-30"),
    RecessionWindow("2007-2009", "2008-01-01", "2009-06-30"),
    RecessionWindow("2020", "2020-03-01", "2020-04-30"),
]


def _frame_with_date(data: pd.DataFrame, date: str | None) -> pd.DataFrame:
    frame = data.copy()
    if date:
        frame[date] = pd.to_datetime(frame[date])
        return frame.set_index(date)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    return frame


def _column_list(columns: str | Sequence[str]) -> list[str]:
    """Return one or more column names as a list."""

    return [columns] if isinstance(columns, str) else list(columns)


def _observed_date_range(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the plotted date range after dropping all-missing rows."""

    observed = frame[list(columns)].dropna(how="all")
    if observed.empty:
        return None
    return pd.Timestamp(observed.index.min()), pd.Timestamp(observed.index.max())


def _style_colors(style: str) -> dict[str, str]:
    if style == "fins":
        return FINS_COLORS
    if style == "ft":
        return FT_COLORS
    raise ValueError("style must be one of: fins, ft")


def _palette_for_style(style: str) -> list[str]:
    if style == "fins":
        return COLOR_SAFE_SEABORN
    if style == "ft":
        return FT_COLOR_CYCLE
    raise ValueError("style must be one of: fins, ft")


def _line_color(style: str, key: str = "primary") -> str:
    colors = _style_colors(style)
    if style == "ft":
        return {
            "primary": colors["maroon"],
            "secondary": colors["blue"],
            "positive": colors["teal"],
            "negative": colors["pink"],
            "text": colors["muted"],
            "zero": colors["charcoal"],
            "grid": colors["grid"],
            "recession": colors["recession"],
        }[key]
    return {
        "primary": colors["navy"],
        "secondary": colors["crimson"],
        "positive": colors["forest"],
        "negative": colors["crimson"],
        "text": colors["steel"],
        "zero": "#111827",
        "grid": "#D8DDE6",
        "recession": "#DDE1E7",
    }[key]


DISPLAY_LABEL_ACRONYMS = {
    "bamlh0a0hym2": "HY OAS",
    "bm": "BM",
    "cape": "CAPE",
    "cpi": "CPI",
    "cpiaucsl": "CPI",
    "dgs10": "10Y Treasury",
    "dgs2": "2Y Treasury",
    "dtb3": "3M T-bill",
    "fedfunds": "Federal funds rate",
    "gdp": "GDP",
    "hml": "HML",
    "hy": "HY",
    "indpro": "Industrial production",
    "mkt": "Mkt",
    "payems": "Payroll employment",
    "rf": "RF",
    "smb": "SMB",
    "t10y2y": "10Y-2Y",
    "t10y3m": "10Y-3M",
    "us": "U.S.",
    "usd": "USD",
    "vix": "VIX",
    "vixcls": "VIX",
}

DISPLAY_LABEL_PHRASES = {
    "hy spread rolling 21d": "HY spread 21-day avg",
    "ten year minus three month": "10Y-3M",
    "ten year minus two year": "10Y-2Y",
    "vix rolling 21d": "VIX 21-day avg",
}


def _display_label(value: object) -> str:
    """Return a presentation-ready label for a dataframe field name."""

    text = str(value).strip()
    if not text:
        return text

    normalized = re.sub(r"_+", " ", text)
    phrase_key = re.sub(r"\s+", " ", normalized).lower()
    if phrase_key in DISPLAY_LABEL_PHRASES:
        return DISPLAY_LABEL_PHRASES[phrase_key]

    acronym_key = re.sub(r"[^A-Za-z0-9]", "", text).lower()
    if acronym_key in DISPLAY_LABEL_ACRONYMS:
        return DISPLAY_LABEL_ACRONYMS[acronym_key]

    if normalized != text or (len(normalized.split()) == 1 and normalized.islower()):
        parts = []
        for token in normalized.split():
            key = re.sub(r"[^A-Za-z0-9]", "", token).lower()
            if key in DISPLAY_LABEL_ACRONYMS:
                parts.append(DISPLAY_LABEL_ACRONYMS[key])
            elif token.isupper() or any(char.isupper() for char in token[1:]):
                parts.append(token)
            else:
                parts.append(token[:1].upper() + token[1:])
        return " ".join(parts)

    if normalized[0].islower():
        return normalized[:1].upper() + normalized[1:]
    return normalized


def _unique_display_labels(values: Sequence[object]) -> list[str]:
    """Return display labels with suffixes if labels would otherwise collide."""

    labels: list[str] = []
    used: set[str] = set()
    for value in values:
        label = _display_label(value)
        candidate = label
        suffix = 2
        while candidate in used:
            candidate = f"{label} {suffix}"
            suffix += 1
        labels.append(candidate)
        used.add(candidate)
    return labels


def _legend_title(value: str | None) -> str | None:
    return _display_label(value) if value else None


def _categorical_colors(style: str, count: int) -> list[str]:
    """Return enough distinct category colors without silently repeating."""

    if count <= 0:
        return []
    base = _palette_for_style(style)
    if count <= len(base):
        return base[:count]
    candidates = list(base)
    candidates.extend(sns.color_palette("tab20", count).as_hex())
    candidates.extend(sns.color_palette("husl", count).as_hex())
    colors: list[str] = []
    for color in candidates:
        if color not in colors:
            colors.append(color)
        if len(colors) == count:
            break
    return colors


def _resolve_plot_colors(
    style: str,
    labels: Sequence[object],
    colors: Mapping[object, str] | Sequence[str] | None = None,
) -> list[str]:
    """Return explicit or style-default colors for a plotted label sequence."""

    label_list = list(labels)
    if colors is None:
        return _categorical_colors(style, len(label_list))
    if isinstance(colors, Mapping):
        defaults = _categorical_colors(style, len(label_list))
        resolved: list[str] = []
        for index, label in enumerate(label_list):
            resolved.append(
                colors.get(label)
                or colors.get(str(label))
                or defaults[index]
            )
        return resolved
    palette = list(colors)
    if len(palette) < len(label_list):
        raise ValueError("explicit colors must cover every plotted label")
    return palette[: len(label_list)]


def _neutral_mark_color(style: str) -> str:
    if style == "ft":
        return FT_COLORS["axis"]
    return FINS_COLORS["gray"]


def _dense_line_width(style: str, line_count: int) -> float | None:
    if style == "ft":
        return 1.25 if line_count > 1 else 1.55
    return None


def _dense_line_alpha(style: str, line_count: int) -> float:
    if style == "ft":
        return 0.78 if line_count > 1 else 0.92
    return 1.0


def _apply_horizontal_grid(ax: plt.Axes, *, style: str = "fins") -> None:
    """Apply the house style for report grids: horizontal only and unobtrusive."""

    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    ax.grid(
        True,
        axis="y",
        color=_line_color(style, "grid"),
        linewidth=0.7 if style == "ft" else 0.6,
        alpha=0.75 if style == "ft" else 0.55,
    )


def _nice_year_step(year_span: int, max_ticks: int) -> int:
    """Return a readable year step that keeps labels under the tick cap."""

    for step in [1, 2, 5, 10, 20, 25, 50, 100]:
        if int(np.ceil(year_span / step)) + 2 <= max_ticks:
            return step
    return 100


def _format_date_axis(
    ax: plt.Axes,
    *,
    date_start: object | None = None,
    date_end: object | None = None,
    max_ticks: int = 8,
) -> None:
    """Apply sparse date ticks that remain readable in Word-sized figures."""

    if date_start is None or date_end is None:
        date_start, date_end = _axis_date_range(ax)
    start = _clean_timestamp(date_start)
    end = _clean_timestamp(date_end)
    span_days = max((end - start).days, 1)

    if span_days >= 365 * 2:
        year_span = max(end.year - start.year, 1)
        step = _nice_year_step(year_span, max_ticks)
        first_internal_year = ((start.year // step) + 1) * step
        min_endpoint_gap = max(2.0, step / 2)
        tick_dates = [start]
        tick_dates.extend(
            pd.Timestamp(year=year, month=1, day=1)
            for year in range(first_internal_year, end.year + 1, step)
            if (
                start < pd.Timestamp(year=year, month=1, day=1) < end
                and year - start.year >= min_endpoint_gap
                and end.year - year >= min_endpoint_gap
            )
        )
        tick_dates.append(end)
        formatter = mdates.DateFormatter("%Y")
    else:
        tick_count = min(max_ticks, max(2, int(np.ceil(span_days / 30))))
        tick_dates = list(pd.date_range(start=start, end=end, periods=tick_count))
        formatter = mdates.DateFormatter("%Y-%m" if span_days > 90 else "%Y-%m-%d")

    tick_dates = list(dict.fromkeys(tick_dates))
    ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(date) for date in tick_dates]))
    ax.xaxis.set_major_formatter(formatter)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")


def _sparse_positions(length: int, max_labels: int) -> list[int]:
    """Return sparse tick positions while preserving the first and last item."""

    if length <= 0:
        return []
    if max_labels <= 0 or length <= max_labels:
        return list(range(length))
    positions = np.rint(np.linspace(0, length - 1, max_labels)).astype(int)
    return sorted(set(int(position) for position in positions))


def _legend_if_requested(
    ax: plt.Axes,
    *,
    enabled: bool,
    title: str | None = None,
) -> None:
    """Add a legend only when plotted artists expose labels."""

    if not enabled:
        return
    handles, labels = ax.get_legend_handles_labels()
    labeled = [(handle, label) for handle, label in zip(handles, labels, strict=False) if label]
    if labeled:
        handles, labels = zip(*labeled, strict=True)
        ax.legend(handles, labels, title=_legend_title(title))


def _finish_axis(
    ax: plt.Axes,
    *,
    title: str | None,
    xlabel: str,
    ylabel: str,
    format_dates: bool = False,
) -> None:
    if title:
        ax.set_title(title, loc="left")
    ax.set_xlabel(_display_label(xlabel))
    ax.set_ylabel(_display_label(ylabel))
    if format_dates:
        ax.figure.autofmt_xdate(rotation=0, ha="center")


def _pad_marker_limits(
    ax: plt.Axes,
    x_values: object,
    y_values: object,
    sizes: object,
    *,
    label_padding_px: float = 0.0,
) -> None:
    """Expand numeric axis limits so scatter bubbles are not clipped."""

    x = pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(dtype=float)
    marker_sizes = pd.to_numeric(pd.Series(sizes), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return

    x = x[finite]
    y = y[finite]
    max_size = float(np.nanmax(marker_sizes)) if np.isfinite(marker_sizes).any() else 40.0
    marker_radius_px = (np.sqrt(max(max_size, 1.0)) * ax.figure.dpi / 72.0 * 0.55) + 4.0

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    x_range = x_max - x_min if x_max > x_min else max(abs(x_max), 1.0)
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
    ax.set_xlim(x_min - 0.06 * x_range, x_max + 0.08 * x_range)
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.12 * y_range)

    for _ in range(2):
        ax.figure.canvas.draw()
        bbox = ax.get_window_extent()
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        x_per_px = abs(x_right - x_left) / max(float(bbox.width), 1.0)
        y_per_px = abs(y_top - y_bottom) / max(float(bbox.height), 1.0)
        required_left = x_min - marker_radius_px * x_per_px
        required_right = x_max + marker_radius_px * x_per_px
        required_bottom = y_min - marker_radius_px * y_per_px
        required_top = y_max + (marker_radius_px + label_padding_px) * y_per_px
        ax.set_xlim(min(x_left, required_left), max(x_right, required_right))
        ax.set_ylim(min(y_bottom, required_bottom), max(y_top, required_top))


def _adjusted_y_positions(
    ax: plt.Axes,
    values: Sequence[float],
    *,
    min_gap_px: float = 16.0,
    margin_px: float = 10.0,
) -> list[float]:
    """Return y positions adjusted in display space to avoid label overlap."""

    if not values:
        return []

    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    lower = bbox.y0 + margin_px
    upper = bbox.y1 - margin_px
    transformed = [
        float(ax.transData.transform((0, value))[1])
        for value in values
    ]
    order = sorted(range(len(transformed)), key=lambda index: transformed[index])
    adjusted_by_order: list[float] = []
    for index in order:
        target = min(max(transformed[index], lower), upper)
        if adjusted_by_order:
            target = max(target, adjusted_by_order[-1] + min_gap_px)
        adjusted_by_order.append(target)

    if adjusted_by_order and adjusted_by_order[-1] > upper:
        adjusted_by_order[-1] = upper
        for pos in range(len(adjusted_by_order) - 2, -1, -1):
            adjusted_by_order[pos] = min(
                adjusted_by_order[pos],
                adjusted_by_order[pos + 1] - min_gap_px,
            )
    if adjusted_by_order and adjusted_by_order[0] < lower:
        adjusted_by_order[0] = lower
        for pos in range(1, len(adjusted_by_order)):
            adjusted_by_order[pos] = max(
                adjusted_by_order[pos],
                adjusted_by_order[pos - 1] + min_gap_px,
            )

    adjusted_pixels = [0.0] * len(values)
    for order_index, original_index in enumerate(order):
        adjusted_pixels[original_index] = adjusted_by_order[order_index]

    return [
        float(ax.transData.inverted().transform((0, pixel_value))[1])
        for pixel_value in adjusted_pixels
    ]


def _annotate_endpoint_label(
    ax: plt.Axes,
    *,
    label: str,
    xy: tuple[object, float],
    label_y: float,
    side: str,
    color: str,
    leader_color: str | None = None,
    fontsize: float,
    style: str,
) -> None:
    x_ref = ax.get_xlim()[1]
    original_px = ax.transData.transform((x_ref, xy[1]))[1]
    label_px = ax.transData.transform((x_ref, label_y))[1]
    offset_y_points = float((label_px - original_px) * 72.0 / ax.figure.dpi)
    offset_x = -8 if side == "left" else 8
    annotation = ax.annotate(
        label,
        xy=xy,
        xytext=(offset_x, offset_y_points),
        textcoords="offset points",
        ha="right" if side == "left" else "left",
        va="center",
        color=color,
        fontsize=fontsize,
        bbox={
            "boxstyle": "square,pad=0.08",
            "facecolor": ax.get_facecolor(),
            "edgecolor": "none",
            "alpha": 0.86 if style == "ft" else 0.74,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": leader_color or color,
            "alpha": 0.55,
            "linewidth": 0.65,
            "shrinkA": 0,
            "shrinkB": 0,
        }
        if abs(offset_y_points) > 2.0
        else None,
        zorder=5,
    )
    annotation.set_gid("figure_label")


def _figure_label_artists(ax: plt.Axes) -> list[Text]:
    """Return rendered direct-label text artists created by this module."""

    return [
        text
        for text in ax.findobj(match=Text)
        if text.get_gid() == "figure_label"
        and text.get_visible()
        and text.get_text().strip()
    ]


def _text_artist_extent(text: Text, renderer: object):
    bbox_patch = text.get_bbox_patch()
    return (
        bbox_patch.get_window_extent(renderer)
        if bbox_patch is not None
        else text.get_window_extent(renderer)
    )


def _direct_labels_are_unsafe(ax: plt.Axes) -> bool:
    """Return whether direct labels overlap or fall outside the figure canvas."""

    labels = _figure_label_artists(ax)
    if not labels:
        return False

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = [_text_artist_extent(label, renderer) for label in labels]
    figure_box = ax.figure.bbox

    for box in boxes:
        if (
            box.x0 < figure_box.x0
            or box.x1 > figure_box.x1
            or box.y0 < figure_box.y0
            or box.y1 > figure_box.y1
        ):
            return True

    for index, left_box in enumerate(boxes):
        for right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                return True
    return False


def _remove_direct_labels(ax: plt.Axes) -> None:
    """Remove all direct labels from an axis."""

    for label in _figure_label_artists(ax):
        label.remove()


def _fallback_direct_labels_if_unsafe(ax: plt.Axes) -> bool:
    """Drop direct labels when rendered validation says they are not readable."""

    if _direct_labels_are_unsafe(ax):
        _remove_direct_labels(ax)
        return True
    return False


def _add_line_end_labels(ax: plt.Axes, *, style: str = "fins") -> None:
    """Add collision-aware in-plot labels to the latest point of visible lines."""

    candidates: list[tuple[str, object, float, str]] = []

    for line in ax.get_lines():
        label = line.get_label()
        if not label or label.startswith("_"):
            continue
        x_data = line.get_xdata(orig=False)
        y_data = np.asarray(line.get_ydata(orig=False), dtype=float)
        valid = np.isfinite(y_data)
        if not valid.any():
            continue
        x_value = np.asarray(x_data)[valid][-1]
        y_value = y_data[valid][-1]
        candidates.append((label, x_value, float(y_value), line.get_color()))

    adjusted_y = _adjusted_y_positions(
        ax,
        [candidate[2] for candidate in candidates],
        min_gap_px=17.0 if style == "ft" else 15.0,
    )
    for (label, x_value, y_value, color), label_y in zip(candidates, adjusted_y, strict=True):
        _annotate_endpoint_label(
            ax,
            label=label,
            xy=(x_value, y_value),
            label_y=label_y,
            side="left",
            color=color,
            fontsize=8.5 if style == "ft" else 8.8,
            style=style,
        )


def _format_growth_dollars(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    if value >= 100:
        return f"${value:,.0f}"
    if value >= 10:
        return f"${value:.0f}"
    if value >= 1:
        return f"${value:g}"
    return f"${value:.2g}"


def _format_heatmap_annotation(value: object, fmt: str) -> str:
    """Format heatmap cell text while suppressing visual negative zero."""

    try:
        text = format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)
    return re.sub(r"^-0(?=(?:[.,]0+)?(?:%|$))", "0", text)


def add_source_note(
    fig: plt.Figure,
    text: str,
    *,
    x: float = 0.01,
    y: float = 0.01,
    fontsize: float = 8.5,
    style: str = "fins",
) -> Text:
    """Add a compact source note to a standalone PNG/PDF figure."""

    return fig.text(
        x,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=_line_color(style, "text"),
    )


def _clean_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _axis_date_range(ax: plt.Axes) -> tuple[pd.Timestamp, pd.Timestamp]:
    left, right = ax.get_xlim()
    start = _clean_timestamp(mdates.num2date(left))
    end = _clean_timestamp(mdates.num2date(right))
    return start, end


def recession_windows_for_range(
    start: object,
    end: object,
    *,
    windows: Sequence[RecessionWindow] = NBER_RECESSIONS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return NBER recession windows clipped to a plotted date range."""

    range_start = _clean_timestamp(start)
    range_end = _clean_timestamp(end)
    matches: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for window in windows:
        window_start = pd.Timestamp(window.start)
        window_end = pd.Timestamp(window.end)
        if window_end < range_start or window_start > range_end:
            continue
        matches.append(
            (
                max(window_start, range_start),
                min(window_end, range_end),
                window.label,
            )
        )
    return matches


def add_nber_recession_shading(
    ax: plt.Axes,
    *,
    data_start: object | None = None,
    data_end: object | None = None,
    label: bool = False,
    alpha: float = 0.60,
    style: str = "fins",
) -> list[plt.Artist]:
    """Add range-aware NBER recession shading without changing x-axis limits."""

    original_xlim = ax.get_xlim()
    if data_start is None or data_end is None:
        data_start, data_end = _axis_date_range(ax)

    artists: list[plt.Artist] = []
    for start, end, recession_label in recession_windows_for_range(data_start, data_end):
        artist = ax.axvspan(
            start,
            end,
            facecolor=_line_color(style, "recession"),
            edgecolor="none",
            alpha=alpha,
            lw=0,
            zorder=0,
        )
        artists.append(artist)
        if label:
            midpoint = start + (end - start) / 2
            y_top = ax.get_ylim()[1]
            artists.append(
                ax.text(
                    midpoint,
                    y_top,
                    recession_label,
                    ha="center",
                    va="top",
                    color=_line_color(style, "text"),
                    fontsize=8,
                )
            )
    ax.set_xlim(original_xlim)
    return artists


def time_series_plot(
    data: pd.DataFrame,
    y: str | Sequence[str],
    *,
    date: str | None = None,
    title: str | None = None,
    ylabel: str = "Value",
    xlabel: str = "Date",
    shade_recessions: bool = True,
    shade_crises: bool | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    direct_labels: bool = False,
    line_width: float | None = None,
    line_alpha: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more time-series columns."""

    with figure_style(profile, style=style, ft_background=ft_background):
        if shade_crises is not None:
            shade_recessions = shade_crises
        frame = _frame_with_date(data, date)
        columns = _column_list(y)
        observed_range = _observed_date_range(frame, columns)
        effective_line_width = (
            _dense_line_width(style, len(columns)) if line_width is None else line_width
        )
        effective_line_alpha = (
            _dense_line_alpha(style, len(columns)) if line_alpha is None else line_alpha
        )
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for column in columns:
            series = frame[column].dropna()
            if not series.empty:
                line_kwargs: dict[str, object] = {
                    "label": str(column),
                    "alpha": effective_line_alpha,
                    "zorder": 3,
                }
                if effective_line_width is not None:
                    line_kwargs["linewidth"] = effective_line_width
                ax.plot(
                    series.index,
                    series,
                    **line_kwargs,
                )
        if observed_range is not None:
            ax.set_xlim(observed_range)
        if shade_recessions:
            add_nber_recession_shading(
                ax,
                data_start=observed_range[0] if observed_range else frame.index.min(),
                data_end=observed_range[1] if observed_range else frame.index.max(),
                style=style,
            )
        if observed_range is not None:
            _format_date_axis(ax, date_start=observed_range[0], date_end=observed_range[1])
        else:
            _format_date_axis(ax)
        _apply_horizontal_grid(ax, style=style)
        if direct_labels:
            _add_line_end_labels(ax, style=style)
        _legend_if_requested(ax, enabled=legend, title=legend_title)
        _finish_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, format_dates=True)
        if direct_labels and _fallback_direct_labels_if_unsafe(ax) and ax.get_legend() is None:
            _legend_if_requested(ax, enabled=True, title=legend_title)
        return fig, ax


def cumulative_returns_plot(
    data: pd.DataFrame,
    returns: str | Sequence[str],
    *,
    date: str | None = None,
    returns_are_percent: bool = True,
    wealth_index: bool = False,
    base: float = 1.0,
    log_scale: bool = False,
    title: str | None = None,
    legend: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    direct_labels: bool = False,
    line_width: float | None = None,
    line_alpha: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot cumulative returns from a simple return series."""

    frame = _frame_with_date(data, date)
    columns = _column_list(returns)
    cumulative_series: dict[str, pd.Series] = {}
    for column in columns:
        values = frame[column].dropna().astype(float)
        if returns_are_percent:
            values = values / 100.0
        wealth = (1.0 + values).cumprod()
        cumulative_series[str(column)] = wealth * base if wealth_index else wealth - 1.0
    plot_frame = pd.DataFrame(cumulative_series)
    default_title = (
        f"Cumulative return from {columns[0]}"
        if len(columns) == 1
        else "Cumulative returns"
    )
    fig, ax = time_series_plot(
        plot_frame,
        list(cumulative_series),
        title=title or default_title,
        ylabel=f"Growth of ${base:g}" if wealth_index else "Cumulative return",
        shade_recessions=True,
        legend=legend,
        profile=profile,
        style=style,
        ft_background=ft_background,
        direct_labels=False,
        line_width=line_width,
        line_alpha=line_alpha,
    )
    if log_scale:
        ax.set_yscale("log")
    if wealth_index:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_growth_dollars))
    else:
        ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    if direct_labels:
        _add_line_end_labels(ax, style=style)
        if _fallback_direct_labels_if_unsafe(ax) and not legend:
            _legend_if_requested(ax, enabled=True)
    return fig, ax


def indexed_time_series_plot(
    data: pd.DataFrame,
    y: str | Sequence[str],
    *,
    date: str | None = None,
    base: float = 100.0,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str = "Date",
    shade_recessions: bool = True,
    legend: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    direct_labels: bool = False,
    line_width: float | None = None,
    line_alpha: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more series indexed to a common starting value."""

    frame = _frame_with_date(data, date)
    columns = _column_list(y)
    indexed: dict[str, pd.Series] = {}
    for column in columns:
        series = frame[column].dropna().astype(float)
        series = series.loc[series != 0]
        if series.empty:
            continue
        indexed[str(column)] = series / series.iloc[0] * base
    if not indexed:
        raise ValueError("no non-zero observations available to index")
    plot_frame = pd.DataFrame(indexed)
    fig, ax = time_series_plot(
        plot_frame,
        list(indexed),
        title=title,
        ylabel=ylabel or f"Index ({base:g} = first observation)",
        xlabel=xlabel,
        shade_recessions=shade_recessions,
        legend=legend,
        profile=profile,
        style=style,
        ft_background=ft_background,
        direct_labels=direct_labels,
        line_width=line_width,
        line_alpha=line_alpha,
    )
    return fig, ax


def drawdown_plot(
    data: pd.DataFrame,
    returns: str,
    *,
    date: str | None = None,
    returns_are_percent: bool = True,
    title: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    line_width: float | None = None,
    line_alpha: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot drawdowns from a simple return series."""

    frame = _frame_with_date(data, date)
    values = frame[returns].astype(float)
    if returns_are_percent:
        values = values / 100.0
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    fig, ax = time_series_plot(
        pd.DataFrame({"drawdown": drawdown}, index=frame.index),
        "drawdown",
        title=title or f"Drawdown from {returns}",
        ylabel="Drawdown",
        shade_recessions=True,
        profile=profile,
        style=style,
        ft_background=ft_background,
        line_width=line_width,
        line_alpha=line_alpha,
    )
    ax.fill_between(
        drawdown.index,
        drawdown.to_numpy(),
        0,
        color=_line_color(style, "negative"),
        alpha=0.25,
    )
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    return fig, ax


def bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a clean bar chart."""

    with figure_style(profile, style=style, ft_background=ft_background):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        sns.barplot(data=data, x=x, y=y, ax=ax, color=_line_color(style, "primary"))
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel=x, ylabel=ylabel or y)
        ax.tick_params(axis="x", rotation=30)
        return fig, ax


def mean_return_bar_plot(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    title: str | None = None,
    ylabel: str = "Mean monthly return (%)",
    error: str = "se",
    sort: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    show_error_note: bool = False,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot mean returns with standard-error or standard-deviation bars."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = (
            data[list(columns)]
            if columns is not None
            else data.select_dtypes(include=[np.number])
        )
        means = frame.mean()
        if error == "se":
            errors = frame.sem()
            error_label = "standard errors"
        elif error == "sd":
            errors = frame.std()
            error_label = "standard deviations"
        elif error == "none":
            errors = pd.Series(0.0, index=means.index)
            error_label = ""
        else:
            raise ValueError("error must be one of: se, sd, none")

        summary = pd.DataFrame({"mean": means, error: errors})
        if sort:
            summary = summary.sort_values("mean", ascending=False)

        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        colors = [
            _palette_for_style(style)[index % len(_palette_for_style(style))]
            for index in range(len(summary))
        ]
        ax.bar(
            summary.index.astype(str),
            summary["mean"],
            yerr=None if error == "none" else summary[error],
            capsize=3,
            color=colors,
            edgecolor=_line_color(style, "zero"),
            linewidth=0.5,
        )
        ax.axhline(0, color=_line_color(style, "zero"), linewidth=0.8)
        _apply_horizontal_grid(ax, style=style)
        if show_error_note and error_label:
            ax.text(
                0.99,
                0.98,
                f"Error bars: {error_label}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=_line_color(style, "text"),
                fontsize=8.5,
            )
        _finish_axis(ax, title=title, xlabel="Series", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=30)
        return fig, ax, summary


def stacked_bar_plot(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    date: str | None = None,
    title: str | None = None,
    ylabel: str = "Monthly return (%)",
    max_bars: int | None = 12,
    max_x_tick_labels: int = 8,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    bar_alpha: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot stacked bars with positive and negative values stacked separately."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = _frame_with_date(data, date)
        frame = frame[list(columns)].tail(max_bars) if max_bars else frame[list(columns)]
        labels = [item.strftime("%Y-%m") for item in frame.index]
        x_positions = np.arange(len(frame))
        positive_bottom = np.zeros(len(frame))
        negative_bottom = np.zeros(len(frame))
        palette = _palette_for_style(style)
        effective_alpha = 0.72 if style == "ft" and bar_alpha is None else bar_alpha

        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for index, column in enumerate(columns):
            values = frame[column].astype(float).to_numpy()
            bottoms = np.where(values >= 0, positive_bottom, negative_bottom)
            ax.bar(
                x_positions,
                values,
                bottom=bottoms,
                label=str(column),
                color=palette[index % len(palette)],
                edgecolor="white",
                alpha=effective_alpha,
                linewidth=0.4,
            )
            positive_bottom = np.where(values >= 0, positive_bottom + values, positive_bottom)
            negative_bottom = np.where(values < 0, negative_bottom + values, negative_bottom)

        ax.axhline(0, color=_line_color(style, "zero"), linewidth=0.8)
        tick_positions = _sparse_positions(len(labels), max_x_tick_labels)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [labels[position] for position in tick_positions],
            rotation=0,
            ha="center",
        )
        ax.legend()
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel="Month", ylabel=ylabel)
        return fig, ax


def proportional_stacked_bar_plot(
    data: pd.DataFrame,
    category: str,
    segment: str,
    value: str,
    *,
    title: str | None = None,
    ylabel: str = "Share",
    legend_title: str | None = None,
    normalize: bool = True,
    max_categories: int | None = 10,
    colors: Mapping[object, str] | Sequence[str] | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a part-to-whole stacked bar chart with optional normalization."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[category, segment, value]].dropna().copy()
        ordered_categories = pd.api.types.is_numeric_dtype(frame[category]) or (
            pd.api.types.is_datetime64_any_dtype(frame[category])
        )
        pivot = frame.pivot_table(
            index=category,
            columns=segment,
            values=value,
            aggfunc="sum",
            fill_value=0.0,
            sort=False,
        )
        if max_categories is not None and len(pivot) > max_categories:
            if ordered_categories:
                pivot = pivot.sort_index().tail(max_categories)
            else:
                totals = pivot.sum(axis=1).sort_values(ascending=False)
                pivot = pivot.loc[totals.head(max_categories).index]
        if ordered_categories:
            pivot = pivot.sort_index()
        else:
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        if normalize:
            denominator = pivot.sum(axis=1).replace(0, np.nan)
            pivot = pivot.div(denominator, axis=0).fillna(0.0)

        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x_positions = np.arange(len(pivot))
        bottoms = np.zeros(len(pivot))
        palette = _resolve_plot_colors(style, list(pivot.columns), colors)
        for index, column in enumerate(pivot.columns):
            values = pivot[column].astype(float).to_numpy()
            ax.bar(
                x_positions,
                values,
                bottom=bottoms,
                label=str(column),
                color=palette[index % len(palette)],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.78 if style == "ft" else 0.90,
            )
            bottoms += values

        ax.set_xticks(x_positions)
        ax.set_xticklabels(pivot.index.astype(str), rotation=25, ha="right")
        if normalize:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(lambda tick, _pos: f"{tick:.0%}")
        ax.legend(title=_legend_title(legend_title), ncols=2 if len(pivot.columns) > 4 else 1)
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel=_display_label(category), ylabel=ylabel)
        return fig, ax


def stacked_area_plot(
    data: pd.DataFrame,
    x: str,
    columns: Sequence[str],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Share",
    normalize: bool = True,
    legend_title: str | None = None,
    max_x_tick_labels: int = 8,
    colors: Mapping[object, str] | Sequence[str] | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot stacked areas for part-to-whole changes over time or ordered categories."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[x, *columns]].dropna(subset=[x]).copy()
        ordered = pd.api.types.is_numeric_dtype(frame[x]) or (
            pd.api.types.is_datetime64_any_dtype(frame[x])
        )
        if ordered:
            frame = frame.sort_values(x)
        values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if normalize:
            denominator = values.sum(axis=1).replace(0, np.nan)
            values = values.div(denominator, axis=0).fillna(0.0)

        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        datetime_x = pd.api.types.is_datetime64_any_dtype(frame[x])
        if datetime_x:
            x_values = mdates.date2num(pd.to_datetime(frame[x]).dt.to_pydatetime())
        else:
            x_values = frame[x].to_numpy()
        palette = _resolve_plot_colors(style, list(columns), colors)
        ax.stackplot(
            x_values,
            [values[column].to_numpy(dtype=float) for column in columns],
            labels=[str(column) for column in columns],
            colors=palette,
            alpha=0.78 if style == "ft" else 0.88,
            linewidth=0.4,
            edgecolor="white",
        )
        ax.margins(x=0)
        if normalize:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(lambda tick, _pos: f"{tick:.0%}")

        if datetime_x:
            dates = pd.to_datetime(frame[x])
            _format_date_axis(
                ax,
                date_start=dates.min(),
                date_end=dates.max(),
                max_ticks=max_x_tick_labels,
            )
        else:
            tick_positions = _sparse_positions(len(frame), max_x_tick_labels)
            if tick_positions:
                tick_values = frame[x].iloc[tick_positions]
                ax.set_xticks(tick_values.to_numpy())
                ax.set_xticklabels(tick_values.astype(str), rotation=0, ha="center")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncols=2 if len(columns) > 4 else min(len(columns), 4),
            title=_legend_title(legend_title),
        )
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel=xlabel or _display_label(x), ylabel=ylabel)
        return fig, ax


def diverging_bar_plot(
    data: pd.DataFrame,
    category: str,
    value: str,
    *,
    reference: float = 0.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    sort: bool = True,
    limit: int | None = None,
    color_mode: str = "auto",
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot values diverging above and below a fixed reference."""

    if color_mode not in {"auto", "signed", "categorical"}:
        raise ValueError("color_mode must be one of: auto, signed, categorical")

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[category, value]].dropna().copy()
        frame["deviation"] = frame[value].astype(float) - reference
        if sort:
            frame = frame.sort_values("deviation", ascending=True)
        if limit is not None and len(frame) > limit:
            frame = frame.iloc[-limit:]

        fig, ax = plt.subplots(figsize=(7.0, max(3.6, 0.34 * len(frame) + 1.2)))
        y_positions = np.arange(len(frame))
        deviations = frame["deviation"].to_numpy()
        has_positive = bool(np.any(deviations >= 0))
        has_negative = bool(np.any(deviations < 0))
        use_signed_colors = color_mode == "signed" or (
            color_mode == "auto" and has_positive and has_negative
        )
        if use_signed_colors:
            colors = np.where(
                deviations >= 0,
                _line_color(style, "positive"),
                _line_color(style, "negative"),
            )
        else:
            colors = _categorical_colors(style, len(frame))
        ax.barh(y_positions, frame["deviation"], color=colors, alpha=0.82, zorder=3)
        ax.axvline(0, color=_line_color(style, "zero"), linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(frame[category].astype(str))
        ax.grid(True, axis="x", color=_line_color(style, "grid"), linewidth=0.7, alpha=0.75)
        ax.grid(False, axis="y")
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel or f"{_display_label(value)} relative to {reference:g}",
            ylabel=ylabel or _display_label(category),
        )
        return fig, ax


def dumbbell_plot(
    data: pd.DataFrame,
    category: str,
    start: str,
    end: str,
    *,
    title: str | None = None,
    xlabel: str = "Value",
    ylabel: str | None = None,
    start_label: str | None = None,
    end_label: str | None = None,
    sort: bool = True,
    limit: int | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot two values per category with a connecting range line."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[category, start, end]].dropna().copy()
        if sort:
            frame = frame.assign(change=frame[end].astype(float) - frame[start].astype(float))
            frame = frame.sort_values("change", ascending=True)
        if limit is not None and len(frame) > limit:
            frame = frame.tail(limit)

        fig, ax = plt.subplots(figsize=(7.0, max(3.8, 0.36 * len(frame) + 1.2)))
        y_positions = np.arange(len(frame))
        start_values = frame[start].astype(float).to_numpy()
        end_values = frame[end].astype(float).to_numpy()
        ax.hlines(
            y_positions,
            xmin=np.minimum(start_values, end_values),
            xmax=np.maximum(start_values, end_values),
            color=_line_color(style, "grid"),
            linewidth=2.0,
            zorder=2,
        )
        ax.scatter(
            start_values,
            y_positions,
            label=start_label or start,
            color=_line_color(style, "secondary"),
            s=42,
            zorder=3,
        )
        ax.scatter(
            end_values,
            y_positions,
            label=end_label or end,
            color=_line_color(style, "primary"),
            s=46,
            zorder=4,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(frame[category].astype(str))
        ax.legend()
        ax.grid(True, axis="x", color=_line_color(style, "grid"), linewidth=0.7, alpha=0.75)
        ax.grid(False, axis="y")
        _finish_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel or _display_label(category))
        return fig, ax


def grouped_bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    legend_title: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot grouped bars for categorical comparisons."""

    with figure_style(profile, style=style, ft_background=ft_background):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        groups = list(pd.Series(data[group]).dropna().unique())
        palette_values = _categorical_colors(style, len(groups))
        palette = {
            value: palette_values[index % len(palette_values)]
            for index, value in enumerate(groups)
        }
        sns.barplot(data=data, x=x, y=y, hue=group, ax=ax, palette=palette)
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel=_display_label(x), ylabel=ylabel or _display_label(y))
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title=_legend_title(legend_title))
        return fig, ax


def scatter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    fit: bool = True,
    annotate: bool = True,
    label_outliers: int = 0,
    stats_location: str = "upper left",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_title: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a scatter chart with an optional fitted line."""

    with figure_style(profile, style=style, ft_background=ft_background):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        columns = [x, y] + ([hue] if hue else [])
        frame = data[columns].dropna().copy()
        if fit and hue is None:
            sns.regplot(
                data=frame,
                x=x,
                y=y,
                ax=ax,
                color=_line_color(style, "primary"),
                scatter_kws={"s": 34, "alpha": 0.78},
                line_kws={"color": _line_color(style, "secondary"), "linewidth": 1.8},
            )
            if annotate and len(frame) >= 3:
                x_values = frame[x].astype(float).to_numpy()
                y_values = frame[y].astype(float).to_numpy()
                slope, intercept = np.polyfit(x_values, y_values, deg=1)
                fitted = slope * x_values + intercept
                ss_res = float(np.square(y_values - fitted).sum())
                ss_tot = float(np.square(y_values - y_values.mean()).sum())
                r_squared = 1.0 - ss_res / ss_tot if ss_tot else np.nan
                if stats_location == "upper left":
                    x_anchor, y_anchor, ha, va = 0.03, 0.97, "left", "top"
                elif stats_location == "upper right":
                    x_anchor, y_anchor, ha, va = 0.97, 0.97, "right", "top"
                elif stats_location == "lower left":
                    x_anchor, y_anchor, ha, va = 0.03, 0.03, "left", "bottom"
                elif stats_location == "lower right":
                    x_anchor, y_anchor, ha, va = 0.97, 0.03, "right", "bottom"
                else:
                    raise ValueError(
                        "stats_location must be one of: upper left, upper right, "
                        "lower left, lower right"
                    )
                ax.text(
                    x_anchor,
                    y_anchor,
                    f"Slope: {slope:.2f}\nR-squared: {r_squared:.2f}\nN: {len(frame):,}",
                    transform=ax.transAxes,
                    ha=ha,
                    va=va,
                    fontsize=8.8,
                    color=_line_color(style, "text"),
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": ax.get_facecolor(),
                        "edgecolor": _line_color(style, "grid"),
                        "alpha": 0.9,
                    },
                )
        else:
            sns.scatterplot(
                data=frame,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                s=42,
                alpha=0.85,
                palette=_categorical_colors(style, frame[hue].nunique()) if hue else None,
            )
            if hue and ax.get_legend() is not None:
                ax.legend(title=_legend_title(legend_title))
        marker_size = 34.0 if fit and hue is None else 42.0
        _pad_marker_limits(ax, frame[x], frame[y], marker_size)
        if label_outliers > 0 and not frame.empty:
            z_x = ((frame[x] - frame[x].mean()) / frame[x].std(ddof=0)).fillna(0)
            z_y = ((frame[y] - frame[y].mean()) / frame[y].std(ddof=0)).fillna(0)
            outlier_scores = (z_x.abs() + z_y.abs()).nlargest(label_outliers * 8)
            selected: list[object] = []
            selected_points: list[np.ndarray] = []
            for index in outlier_scores.index:
                point = np.array([z_x.loc[index], z_y.loc[index]], dtype=float)
                if any(np.linalg.norm(point - existing) < 0.75 for existing in selected_points):
                    continue
                selected.append(index)
                selected_points.append(point)
                if len(selected) == label_outliers:
                    break
            median_x = frame[x].median()
            for index in selected:
                label = index.strftime("%Y-%m") if hasattr(index, "strftime") else str(index)
                xytext = (-8, 7) if frame.loc[index, x] > median_x else (7, 7)
                ha = "right" if frame.loc[index, x] > median_x else "left"
                annotation = ax.annotate(
                    label,
                    (frame.loc[index, x], frame.loc[index, y]),
                    textcoords="offset points",
                    xytext=xytext,
                    ha=ha,
                    fontsize=8,
                    color=_line_color(style, "text"),
                )
                annotation.set_gid("figure_label")
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel or _display_label(x),
            ylabel=ylabel or _display_label(y),
        )
        return fig, ax


def bubble_scatter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    *,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size_label: str | None = None,
    legend_title: str | None = None,
    label: str | None = None,
    label_top: int = 0,
    min_size: float = 40.0,
    max_size: float = 620.0,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a scatterplot with a third variable encoded as bubble size."""

    with figure_style(profile, style=style, ft_background=ft_background):
        columns = [x, y, size]
        if hue:
            columns.append(hue)
        if label:
            columns.append(label)
        frame = data[columns].dropna().copy()
        size_values = frame[size].astype(float)
        if size_values.max() == size_values.min():
            point_sizes = np.full(len(frame), (min_size + max_size) / 2)
        else:
            scaled = (size_values - size_values.min()) / (size_values.max() - size_values.min())
            point_sizes = min_size + scaled.to_numpy() * (max_size - min_size)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        if hue:
            groups = list(frame[hue].astype(str).drop_duplicates())
            palette = _categorical_colors(style, len(groups))
            for index, group in enumerate(groups):
                mask = frame[hue].astype(str) == group
                ax.scatter(
                    frame.loc[mask, x],
                    frame.loc[mask, y],
                    s=point_sizes[mask.to_numpy()],
                    label=group,
                    color=palette[index % len(palette)],
                    alpha=0.64 if style == "ft" else 0.70,
                    edgecolor="white",
                    linewidth=0.6,
                )
            ax.legend(title=_legend_title(legend_title))
        else:
            ax.scatter(
                frame[x],
                frame[y],
                s=point_sizes,
                color=_line_color(style, "primary"),
                alpha=0.64 if style == "ft" else 0.70,
                edgecolor="white",
                linewidth=0.6,
                label=size_label or size,
            )

        _pad_marker_limits(
            ax,
            frame[x],
            frame[y],
            point_sizes,
            label_padding_px=16.0 if label and label_top > 0 else 0.0,
        )

        if label and label_top > 0:
            selected = frame.nlargest(label_top, size)
            for _, row in selected.iterrows():
                annotation = ax.annotate(
                    str(row[label]),
                    (row[x], row[y]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=8.5,
                    color=_line_color(style, "text"),
                )
                annotation.set_gid("figure_label")
            _fallback_direct_labels_if_unsafe(ax)

        _apply_horizontal_grid(ax, style=style)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel or _display_label(x),
            ylabel=ylabel or _display_label(y),
        )
        return fig, ax


def bubble_matrix_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size_label: str | None = None,
    x_order: Sequence[object] | None = None,
    y_order: Sequence[object] | None = None,
    colors: Mapping[object, str] | Sequence[str] | None = None,
    min_size: float = 35.0,
    max_size: float = 1120.0,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a categorical bubble matrix with marker area encoding magnitude."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[x, y, size]].dropna().copy()
        frame[size] = pd.to_numeric(frame[size], errors="coerce")
        frame = frame.dropna(subset=[size]).copy()
        if frame.empty:
            raise ValueError("bubble matrix data is empty")

        x_categories = (
            list(x_order) if x_order is not None else list(frame[x].drop_duplicates())
        )
        y_categories = (
            list(y_order) if y_order is not None else list(frame[y].drop_duplicates())
        )
        x_positions = {category: index for index, category in enumerate(x_categories)}
        y_positions = {category: index for index, category in enumerate(y_categories)}
        frame = frame[frame[x].isin(x_positions) & frame[y].isin(y_positions)].copy()
        if frame.empty:
            raise ValueError("bubble matrix has no rows after applying category order")

        values = frame[size].astype(float)
        positive = values > 0
        point_sizes = np.zeros(len(frame), dtype=float)
        if positive.any():
            positive_values = values[positive]
            if positive_values.max() == positive_values.min():
                point_sizes[positive.to_numpy()] = (min_size + max_size) / 2.0
            else:
                scaled = (
                    (positive_values - positive_values.min())
                    / (positive_values.max() - positive_values.min())
                )
                point_sizes[positive.to_numpy()] = min_size + scaled.to_numpy() * (
                    max_size - min_size
                )

        resolved_colors = _resolve_plot_colors(style, x_categories, colors)
        color_lookup = dict(zip(x_categories, resolved_colors, strict=True))
        fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="none")
        ax.scatter(
            frame[x].map(x_positions),
            frame[y].map(y_positions),
            s=point_sizes,
            color=[color_lookup[value] for value in frame[x]],
            alpha=0.68 if style == "ft" else 0.72,
            edgecolor="white",
            linewidth=0.75,
        )

        ax.set_xlim(-0.35, len(x_categories) - 0.65)
        ax.set_ylim(len(y_categories) - 0.45, -0.55)
        x_labels = []
        for category in x_categories:
            label = _display_label(category)
            if len(label) > 12 and " " in label:
                label = "\n".join(label.split())
            x_labels.append(label)
        ax.set_xticks(
            range(len(x_categories)),
            x_labels,
        )
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(45 if len(x_categories) >= 4 else 0)
            tick_label.set_ha("right" if len(x_categories) >= 4 else "center")
        ax.set_yticks(
            range(len(y_categories)),
            [_display_label(category) for category in y_categories],
        )
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", color=_line_color(style, "grid"), linewidth=0.75, alpha=0.72)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel or _display_label(x),
            ylabel=ylabel or _display_label(y),
        )

        if positive.any():
            positive_values = values[positive]
            max_value = float(positive_values.max())
            legend_values = [max_value / 3.0, max_value * 2.0 / 3.0, max_value]

            def legend_size(value: float) -> float:
                if positive_values.max() == positive_values.min():
                    return (min_size + max_size) / 2.0
                scaled = (value - positive_values.min()) / (
                    positive_values.max() - positive_values.min()
                )
                return float(min_size + np.clip(scaled, 0.0, 1.0) * (max_size - min_size))

            handles = [
                ax.scatter(
                    [],
                    [],
                    s=legend_size(value),
                    color=_line_color(style, "text"),
                    alpha=0.40,
                    edgecolor="white",
                    linewidth=0.75,
                    label=f"{value:.0f}%",
                )
                for value in legend_values
            ]
            ax.legend(
                handles=handles,
                title=_legend_title(size_label or size),
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                labelspacing=1.5,
                borderaxespad=0.0,
            )
        fig.subplots_adjust(bottom=0.30, left=0.16, right=0.84, top=0.86)
        return fig, ax


def distribution_plot(
    data: pd.DataFrame,
    x: str,
    *,
    title: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram with a density overlay."""

    with figure_style(profile, style=style, ft_background=ft_background):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        sns.histplot(data=data, x=x, kde=True, ax=ax, color=_line_color(style, "primary"))
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel=_display_label(x), ylabel="Count")
        return fig, ax


def distribution_comparison_plot(
    data: pd.DataFrame,
    value: str,
    group: str,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    kind: str = "box",
    order: Sequence[str] | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare distributions across groups with a box or violin plot."""

    if kind not in {"box", "violin"}:
        raise ValueError("kind must be one of: box, violin")

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[value, group]].dropna().copy()
        group_values = list(order) if order is not None else list(frame[group].drop_duplicates())
        palette = _categorical_colors(style, len(group_values))
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        if kind == "box":
            sns.boxplot(
                data=frame,
                x=group,
                y=value,
                hue=group,
                order=order,
                ax=ax,
                palette=palette,
                width=0.58,
                fliersize=1.8,
                legend=False,
            )
        else:
            sns.violinplot(
                data=frame,
                x=group,
                y=value,
                hue=group,
                order=order,
                ax=ax,
                palette=palette,
                inner="quartile",
                cut=0,
                legend=False,
            )
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(
            ax,
            title=title,
            xlabel=_display_label(group),
            ylabel=ylabel or _display_label(value),
        )
        ax.tick_params(axis="x", rotation=25)
        return fig, ax


def ecdf_plot(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    title: str | None = None,
    xlabel: str = "Value",
    ylabel: str = "Cumulative share",
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical cumulative distribution curves."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data.copy()
        cols = _column_list(columns)
        palette = _palette_for_style(style)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for index, column in enumerate(cols):
            values = np.sort(frame[column].dropna().astype(float).to_numpy())
            if len(values) == 0:
                continue
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.plot(
                values,
                cumulative,
                label=str(column),
                color=palette[index % len(palette)],
                linewidth=1.6 if style == "ft" else 1.8,
                alpha=0.86,
            )
        ax.yaxis.set_major_formatter(lambda tick, _pos: f"{tick:.0%}")
        _apply_horizontal_grid(ax, style=style)
        _legend_if_requested(ax, enabled=len(cols) > 1)
        _finish_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return fig, ax


def correlation_heatmap(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a correlation heatmap."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[list(columns)] if columns else data.select_dtypes(include=[np.number])
        labels = _unique_display_labels(frame.columns)
        frame = frame.rename(columns=dict(zip(frame.columns, labels, strict=True)))
        corr = frame.corr()
        size = len(corr.columns)
        max_label_length = max((len(str(label)) for label in labels), default=8)
        width = min(8.4, max(7.2, 0.62 * size + 0.08 * max_label_length + 1.9))
        fig, ax = plt.subplots(figsize=(width, max(5.6, 0.48 * size + 1.4)))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Correlation"},
        )
        ax.set_title(title or "Correlation heatmap", loc="left")
        tick_fontsize = 7.6 if size >= 8 else 8.8
        x_rotation = 55 if max_label_length > 12 or size >= 8 else 45
        ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=x_rotation)
        ax.tick_params(axis="y", labelsize=tick_fontsize, rotation=0)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")
        ax.set_xlabel("" if xlabel is None else xlabel)
        ax.set_ylabel("" if ylabel is None else ylabel)
        return fig, ax


def value_heatmap(
    data: pd.DataFrame,
    row: str,
    column: str,
    value: str,
    *,
    title: str | None = None,
    cbar_label: str | None = None,
    annot: bool = True,
    fmt: str = ".1f",
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap for a row-column-value table."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[row, column, value]].dropna().copy()
        pivot = frame.pivot_table(
            index=row,
            columns=column,
            values=value,
            aggfunc="mean",
            sort=False,
        )
        fig, ax = plt.subplots(
            figsize=(max(6.6, 0.46 * len(pivot.columns) + 2.2), max(4.6, 0.35 * len(pivot) + 1.4))
        )
        annotations = (
            pivot.map(lambda item: _format_heatmap_annotation(item, fmt))
            if annot
            else False
        )
        annotation_fontsize = 7.2 if max(len(pivot.columns), len(pivot.index)) >= 12 else 8.8
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="vlag",
            center=0 if pivot.min().min() < 0 < pivot.max().max() else None,
            annot=annotations,
            fmt="" if annot else fmt,
            annot_kws={"fontsize": annotation_fontsize},
            linewidths=0.4,
            cbar_kws={"label": cbar_label or _display_label(value)},
        )
        ax.tick_params(axis="x", rotation=0, length=0)
        ax.tick_params(axis="y", rotation=0, length=0)
        _finish_axis(ax, title=title or "Value heatmap", xlabel=column, ylabel=row)
        return fig, ax


def small_multiples(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    date: str | None = None,
    title: str | None = None,
    ylabel: str | Sequence[str] = "Value",
    shade_recessions: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
    line_width: float | None = None,
    line_alpha: float | None = None,
    title_fontsize: float | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a panel of time-series small multiples."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = _frame_with_date(data, date)
        ncols = 2
        nrows = int(np.ceil(len(columns) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, max(3.0, 2.4 * nrows)),
            sharex=True,
        )
        palette = _palette_for_style(style)
        effective_line_width = (
            1.0 if style == "ft" and line_width is None else line_width
        )
        effective_line_alpha = (
            0.82 if style == "ft" and line_alpha is None else line_alpha
        )
        effective_title_fontsize = (
            9.2 if style == "ft" and title_fontsize is None else title_fontsize
        )
        axes_array = np.atleast_1d(axes).ravel()
        if isinstance(ylabel, str):
            ylabels = [ylabel] * len(columns)
        else:
            ylabels = list(ylabel)
            if len(ylabels) != len(columns):
                raise ValueError("ylabel must be a string or match the number of columns")
        for index, (ax, column) in enumerate(zip(axes_array, columns, strict=False)):
            series = frame[column].dropna()
            line_kwargs: dict[str, object] = {
                "color": palette[index % len(palette)],
                "label": str(column),
            }
            if effective_line_width is not None:
                line_kwargs["linewidth"] = effective_line_width
            if effective_line_alpha is not None:
                line_kwargs["alpha"] = effective_line_alpha
            ax.plot(series.index, series, **line_kwargs)
            if not series.empty:
                ax.set_xlim(series.index.min(), series.index.max())
                _format_date_axis(
                    ax,
                    date_start=series.index.min(),
                    date_end=series.index.max(),
                    max_ticks=4,
                )
                if shade_recessions:
                    add_nber_recession_shading(
                        ax,
                        data_start=series.index.min(),
                        data_end=series.index.max(),
                        style=style,
                    )
            title_kwargs: dict[str, object] = {"loc": "left"}
            if effective_title_fontsize is not None:
                title_kwargs["fontsize"] = effective_title_fontsize
            ax.set_title(str(column), **title_kwargs)
            ax.set_ylabel(ylabels[index])
            _apply_horizontal_grid(ax, style=style)
        for ax in axes_array[len(columns) :]:
            ax.set_visible(False)
        if title:
            fig.suptitle(title, x=0.01, ha="left", weight="bold")
        for ax in axes_array[: len(columns)]:
            for label in ax.get_xticklabels():
                label.set_rotation(0)
                label.set_ha("center")
                label.set_rotation_mode("default")
        return fig, axes_array


def lollipop_plot(
    data: pd.DataFrame,
    category: str,
    value: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    sort: bool = True,
    limit: int | None = None,
    highlight: str | Sequence[str] | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a ranked lollipop chart for magnitude or ranking comparisons."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[category, value]].dropna().copy()
        if sort:
            frame = frame.sort_values(value, ascending=True)
        if limit is not None and len(frame) > limit:
            frame = frame.tail(limit)
        if sort:
            frame = frame.sort_values(value, ascending=True)

        highlight_order = [
            str(item)
            for item in (_column_list(highlight) if highlight is not None else [])
        ]
        highlighted = set(highlight_order)
        y_positions = np.arange(len(frame))
        palette = _categorical_colors(style, max(len(highlight_order), 1))
        highlight_colors = {
            name: palette[index % len(palette)]
            for index, name in enumerate(highlight_order)
        }
        neutral_color = _neutral_mark_color(style)
        point_colors = [
            to_rgba(highlight_colors[str(name)], 0.95)
            if str(name) in highlighted
            else to_rgba(
                neutral_color if highlighted else palette[0],
                0.45 if highlighted else 0.90,
            )
            for name in frame[category].astype(str)
        ]

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        values = frame[value].astype(float).to_numpy()
        ax.hlines(
            y_positions,
            xmin=0,
            xmax=values,
            color=_line_color(style, "grid"),
            linewidth=2.0,
            zorder=2,
        )
        ax.scatter(
            values,
            y_positions,
            s=58,
            color=point_colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        if np.nanmin(values) < 0 < np.nanmax(values):
            ax.axvline(0, color=_line_color(style, "zero"), linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(frame[category].astype(str))
        ax.set_axisbelow(True)
        ax.grid(True, axis="x", color=_line_color(style, "grid"), linewidth=0.7, alpha=0.75)
        ax.grid(False, axis="y")
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel or _display_label(value),
            ylabel=ylabel or _display_label(category),
        )
        return fig, ax


def slope_chart(
    data: pd.DataFrame,
    category: str,
    start: str,
    end: str,
    *,
    title: str | None = None,
    ylabel: str = "Value",
    start_label: str | None = None,
    end_label: str | None = None,
    sort: bool = True,
    limit: int | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot changes between two periods with paired endpoints."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = data[[category, start, end]].dropna().copy()
        if sort:
            frame = frame.sort_values(end, ascending=False)
        if limit is not None and len(frame) > limit:
            frame = frame.head(limit)
        if sort:
            frame = frame.sort_values(end, ascending=True)

        fig, ax = plt.subplots(figsize=(6.4, max(4.8, 0.48 * len(frame) + 1.6)))
        palette = _palette_for_style(style)
        x_values = np.array([0, 1])
        start_labels: list[tuple[str, float, str]] = []
        end_labels: list[tuple[str, float, str]] = []
        for index, (_row_index, row) in enumerate(frame.iterrows()):
            name = str(row[category])
            start_value = float(row[start])
            end_value = float(row[end])
            color = palette[index % len(palette)]
            ax.plot(x_values, [start_value, end_value], color=color, linewidth=1.65, alpha=0.90)
            ax.scatter(x_values, [start_value, end_value], color=color, s=32, zorder=3)
            start_labels.append((name, start_value, color))
            end_labels.append((name, end_value, color))
        ax.set_xlim(-0.22, 1.22)
        ax.margins(y=0.16)
        ax.set_xticks([0, 1], [start_label or start, end_label or end])
        adjusted_start = _adjusted_y_positions(
            ax,
            [item[1] for item in start_labels],
            min_gap_px=30.0,
        )
        adjusted_end = _adjusted_y_positions(
            ax,
            [item[1] for item in end_labels],
            min_gap_px=30.0,
        )
        for (name, value, color), label_y in zip(start_labels, adjusted_start, strict=True):
            _annotate_endpoint_label(
                ax,
                label=name,
                xy=(0, value),
                label_y=label_y,
                side="left",
                color=_line_color(style, "text"),
                leader_color=color,
                fontsize=8.0 if style == "ft" else 8.5,
                style=style,
            )
        for (name, value, color), label_y in zip(end_labels, adjusted_end, strict=True):
            _annotate_endpoint_label(
                ax,
                label=name,
                xy=(1, value),
                label_y=label_y,
                side="right",
                color=_line_color(style, "text"),
                leader_color=color,
                fontsize=8.0 if style == "ft" else 8.5,
                style=style,
            )
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(ax, title=title, xlabel="Period", ylabel=ylabel)
        return fig, ax


def connected_scatter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    date: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label_start_end: bool = True,
    max_points: int | None = 36,
    max_year_span: int | None = 8,
    arrows: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a short episode showing how two variables move through time."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = _frame_with_date(data, date)
        frame = frame[[x, y]].dropna().copy()
        if isinstance(frame.index, pd.DatetimeIndex) and max_year_span is not None:
            span_years = (frame.index.max() - frame.index.min()).days / 365.25
            if span_years > max_year_span:
                raise ValueError(
                    "connected_scatter_plot is for short episodes; "
                    "filter the date range or raise max_year_span explicitly"
                )
        if max_points is not None and len(frame) > max_points:
            positions = _sparse_positions(len(frame), max_points)
            frame = frame.iloc[positions]

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(
            frame[x],
            frame[y],
            color=_line_color(style, "primary"),
            linewidth=1.4,
            marker="o",
            markersize=4,
            alpha=0.92,
        )
        if arrows and len(frame) >= 3:
            arrow_positions = _sparse_positions(len(frame) - 1, min(5, len(frame) - 1))
            for position in arrow_positions[1:]:
                start_point = frame.iloc[position - 1]
                end_point = frame.iloc[position]
                ax.annotate(
                    "",
                    xy=(end_point[x], end_point[y]),
                    xytext=(start_point[x], start_point[y]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": _line_color(style, "primary"),
                        "alpha": 0.45,
                        "linewidth": 0.8,
                    },
                    zorder=4,
                )
        if label_start_end and len(frame) >= 2:
            for label, index in [("Start", frame.index[0]), ("End", frame.index[-1])]:
                point = frame.loc[index]
                suffix = f" {index.strftime('%Y-%m')}" if hasattr(index, "strftime") else ""
                annotation = ax.annotate(
                    f"{label}{suffix}",
                    (point[x], point[y]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    ha="left",
                    fontsize=8.5,
                    color=_line_color(style, "text"),
                )
                annotation.set_gid("figure_label")
        _apply_horizontal_grid(ax, style=style)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel or _display_label(x),
            ylabel=ylabel or _display_label(y),
        )
        return fig, ax


def area_balance_plot(
    data: pd.DataFrame,
    y: str,
    *,
    date: str | None = None,
    reference: float = 0.0,
    series_label: str | None = None,
    positive_label: str = "Above reference",
    negative_label: str = "Below reference",
    title: str | None = None,
    ylabel: str = "Value",
    xlabel: str = "Date",
    shade_recessions: bool = True,
    legend: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot positive and negative deviations from a reference value."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = _frame_with_date(data, date)
        series = frame[y].dropna().astype(float)
        if series.empty:
            raise ValueError("series is empty")

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        x_values = series.index.to_pydatetime()
        values = series.to_numpy()
        ax.plot(
            x_values,
            values,
            color=_line_color(style, "primary"),
            label=series_label or str(y),
            zorder=3,
        )
        ax.fill_between(
            x_values,
            values,
            reference,
            where=values >= reference,
            color=_line_color(style, "positive"),
            alpha=0.22,
            interpolate=True,
            label=positive_label,
            zorder=2,
        )
        ax.fill_between(
            x_values,
            values,
            reference,
            where=values < reference,
            color=_line_color(style, "negative"),
            alpha=0.24,
            interpolate=True,
            label=negative_label,
            zorder=2,
        )
        ax.axhline(reference, color=_line_color(style, "zero"), linewidth=0.8)
        ax.set_xlim(series.index.min(), series.index.max())
        if shade_recessions:
            add_nber_recession_shading(
                ax,
                data_start=series.index.min(),
                data_end=series.index.max(),
                style=style,
            )
        _format_date_axis(ax, date_start=series.index.min(), date_end=series.index.max())
        _apply_horizontal_grid(ax, style=style)
        _legend_if_requested(ax, enabled=legend)
        _finish_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, format_dates=True)
        return fig, ax


def uncertainty_band_plot(
    data: pd.DataFrame,
    y: str,
    lower: str,
    upper: str,
    *,
    date: str | None = None,
    title: str | None = None,
    ylabel: str = "Value",
    xlabel: str = "Date",
    shade_recessions: bool = True,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a central series with an uncertainty or forecast band."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = _frame_with_date(data, date)
        plot_frame = frame[[y, lower, upper]].dropna().copy()
        if plot_frame.empty:
            raise ValueError("series is empty")

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        x_values = plot_frame.index.to_pydatetime()
        ax.fill_between(
            x_values,
            plot_frame[lower].astype(float).to_numpy(),
            plot_frame[upper].astype(float).to_numpy(),
            color=_line_color(style, "secondary"),
            alpha=0.20,
            label="Band",
            zorder=2,
        )
        ax.plot(
            x_values,
            plot_frame[y].astype(float).to_numpy(),
            color=_line_color(style, "primary"),
            label=str(y),
            linewidth=1.55 if style == "ft" else 1.8,
            zorder=3,
        )
        ax.set_xlim(plot_frame.index.min(), plot_frame.index.max())
        if shade_recessions:
            add_nber_recession_shading(
                ax,
                data_start=plot_frame.index.min(),
                data_end=plot_frame.index.max(),
                style=style,
            )
        _format_date_axis(ax, date_start=plot_frame.index.min(), date_end=plot_frame.index.max())
        _apply_horizontal_grid(ax, style=style)
        _legend_if_requested(ax, enabled=True)
        _finish_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, format_dates=True)
        return fig, ax


def rolling_stat_plot(
    data: pd.DataFrame,
    y: str | Sequence[str],
    *,
    date: str | None = None,
    window: int = 12,
    statistic: str = "mean",
    title: str | None = None,
    ylabel: str = "Value",
    xlabel: str = "Date",
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot rolling mean or volatility for one or more time-series."""

    if statistic not in {"mean", "volatility"}:
        raise ValueError("statistic must be one of: mean, volatility")

    frame = _frame_with_date(data, date)
    columns = _column_list(y)
    rolling = frame[columns].rolling(window=window, min_periods=max(2, window // 3))
    plot_frame = rolling.mean() if statistic == "mean" else rolling.std()

    fig, ax = time_series_plot(
        plot_frame,
        columns,
        title=title or f"Rolling {statistic}",
        ylabel=ylabel,
        xlabel=xlabel,
        shade_recessions=True,
        profile=profile,
        style=style,
        ft_background=ft_background,
    )
    return fig, ax


def calendar_heatmap(
    data: pd.DataFrame,
    value: str,
    *,
    date: str | None = None,
    year: int | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    profile: str = "paper",
    style: str = "fins",
    ft_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot daily values as a month-by-day calendar heatmap."""

    with figure_style(profile, style=style, ft_background=ft_background):
        frame = _frame_with_date(data, date)
        series = frame[value].dropna().astype(float)
        if year is None:
            year = int(series.index.max().year)
        series = series.loc[str(year)]
        if series.empty:
            raise ValueError(f"no observations for year {year}")

        heatmap_frame = pd.DataFrame(
            {
                "month": series.index.month,
                "day": series.index.day,
                "value": series.to_numpy(),
            }
        ).pivot_table(index="month", columns="day", values="value", aggfunc="mean")
        heatmap_frame = heatmap_frame.reindex(index=range(1, 13), columns=range(1, 32))

        fig, ax = plt.subplots(figsize=(7.4, 4.3))
        sns.heatmap(
            heatmap_frame,
            ax=ax,
            cmap="rocket_r" if style == "ft" else "mako_r",
            linewidths=0.2,
            linecolor=ax.get_facecolor(),
            cbar_kws={"label": cbar_label or _display_label(value)},
        )
        ax.set_yticks(np.arange(12) + 0.5)
        ax.set_yticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            rotation=0,
        )
        tick_positions = [0.5, 9.5, 19.5, 30.5]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(["1", "10", "20", "31"], rotation=0)
        _finish_axis(ax, title=title or f"{value} calendar heatmap", xlabel="Day", ylabel="Month")
        return fig, ax


COLOR_SAFE_SEABORN = [
    FINS_COLORS["navy"],
    FINS_COLORS["crimson"],
    FINS_COLORS["forest"],
    FINS_COLORS["gold"],
    FINS_COLORS["teal"],
    FINS_COLORS["violet"],
]
