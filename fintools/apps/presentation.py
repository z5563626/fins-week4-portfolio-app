"""Presentation helpers for Streamlit coursework apps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def app_label(value: object) -> str:
    """Return a readable app-facing label for a dataframe column or key."""

    text = str(value).strip().replace("_", " ")
    text = " ".join(text.split())
    if not text:
        return ""
    if text.isupper():
        return text
    return text[0].upper() + text[1:]


def prepare_display_frame(
    frame: pd.DataFrame,
    *,
    index_label: str = "Date",
    labels: Mapping[Any, str] | None = None,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with presentation-safe column labels."""

    result = frame.copy()
    if reset_index and not isinstance(result.index, pd.RangeIndex):
        result = result.rename_axis(index_label).reset_index()

    label_map = labels or {}
    rename_map = {column: label_map.get(column, app_label(column)) for column in result.columns}
    return result.rename(columns=rename_map)


def display_column_config(
    frame: pd.DataFrame,
    *,
    date_format: str = "YYYY-MM-DD",
    number_format: str = "%.2f",
    percent_format: str = "%.2f%%",
) -> dict[str, tuple[str, str]]:
    """Return a serializable column-type plan for app table rendering.

    Streamlit's real ``st.column_config`` objects are created in
    ``fintools.apps.streamlit_ui`` so this function stays easy to unit test.
    """

    config: dict[str, tuple[str, str]] = {}
    for column in frame.columns:
        series = frame[column]
        label = str(column)
        if pd.api.types.is_datetime64_any_dtype(series):
            config[label] = ("date", date_format)
        elif pd.api.types.is_numeric_dtype(series):
            lower = label.lower()
            fmt = percent_format if "%" in label or "percent" in lower else number_format
            config[label] = ("number", fmt)
    return config


def dataframe_csv(frame: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes suitable for Streamlit downloads."""

    return frame.to_csv(index=False).encode("utf-8")


def safe_query_choice(
    value: object,
    options: Sequence[str],
    *,
    default: str,
) -> str:
    """Return ``value`` when it is a valid option, otherwise ``default``."""

    text = str(value) if value is not None else ""
    return text if text in options else default


def safe_query_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
    step: int = 1,
) -> int:
    """Return an integer query value clipped to a valid widget range."""

    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    clipped = min(max(parsed, minimum), maximum)
    if step > 1:
        clipped = minimum + round((clipped - minimum) / step) * step
        clipped = min(max(clipped, minimum), maximum)
    return int(clipped)


def data_health_summary(
    frame: pd.DataFrame,
    *,
    source: str,
    date_column: str | None = None,
    value_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return standard data-health fields for an app-facing status panel."""

    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(frame.index, index=frame.index)
    else:
        dates = pd.Series(dtype="datetime64[ns]")

    selected = list(value_columns) if value_columns else list(frame.columns)
    selected = [column for column in selected if column in frame.columns]
    missing = int(frame[selected].isna().sum().sum()) if selected else 0
    date_count = int(dates.notna().sum())
    return {
        "Source": source,
        "Sample start": dates.min().strftime("%Y-%m-%d") if date_count else "n/a",
        "Sample end": dates.max().strftime("%Y-%m-%d") if date_count else "n/a",
        "Observations": f"{len(frame):,}",
        "Missing values": f"{missing:,}",
    }


def latest_delta(series: pd.Series, *, periods: int) -> float | None:
    """Return the latest minus the lagged value when enough data exist."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= periods:
        return None
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def latest_percentile(series: pd.Series) -> float | None:
    """Return the percentile rank of the latest numeric observation."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.rank(pct=True).iloc[-1] * 100.0)
