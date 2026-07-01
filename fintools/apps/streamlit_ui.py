"""Small Streamlit UI helpers imported lazily by app templates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .presentation import (
    data_health_summary,
    dataframe_csv,
    display_column_config,
    prepare_display_frame,
    safe_query_choice,
    safe_query_int,
)


@dataclass(frozen=True)
class MetricCard:
    """A standard app metric card."""

    label: str
    value: object
    delta: object | None = None
    delta_color: str = "normal"
    help: str | None = None
    delta_description: str | None = None


def require_streamlit():
    """Import Streamlit with an actionable dependency error."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Streamlit is required for app work. Install repo requirements with "
            "`python -m pip install -r requirements.txt -r requirements-dev.txt`."
        ) from exc
    return st


def configure_page(title: str, *, icon: str = ":chart_with_upwards_trend:") -> object:
    """Apply the standard page setup for course apps."""

    st = require_streamlit()
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    return st


def metric_row(metrics: Mapping[str, object], *, columns: int = 4) -> None:
    """Render a row of compact metric cards."""

    st = require_streamlit()
    cols = st.columns(columns)
    for index, (label, value) in enumerate(metrics.items()):
        cols[index % columns].metric(str(label), value)


def query_choice(key: str, options: list[str], *, default: str) -> str:
    """Read a URL query parameter as a validated single-choice value."""

    st = require_streamlit()
    return safe_query_choice(st.query_params.get(key), options, default=default)


def query_int(
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
    step: int = 1,
) -> int:
    """Read a URL query parameter as a bounded integer."""

    st = require_streamlit()
    return safe_query_int(
        st.query_params.get(key),
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
    )


def sync_query_params(**params: object) -> None:
    """Update URL query parameters only when they have changed."""

    st = require_streamlit()
    current = st.query_params.to_dict()
    wanted = {
        str(key): str(value)
        for key, value in params.items()
        if value is not None and str(value) != ""
    }
    if current != wanted:
        st.query_params.from_dict(wanted)


def stable_tab_default(labels: list[str], *, default: str, key: str) -> str:
    """Return a tab default that stays stable after the first render.

    Stateful Streamlit tabs include the ``default`` value in their element
    identity. When an app reads ``default`` from URL query params on every rerun,
    tab clicks can briefly navigate and then bounce back to the prior URL state.
    Store the first valid default separately, then let ``st.session_state[key]``
    own subsequent tab navigation.
    """

    if not labels:
        return default
    st = require_streamlit()
    initial_default_key = f"{key}__initial_default"
    requested_default = safe_query_choice(default, labels, default=labels[0])
    if initial_default_key not in st.session_state:
        st.session_state[initial_default_key] = requested_default
    else:
        st.session_state[initial_default_key] = safe_query_choice(
            st.session_state.get(initial_default_key),
            labels,
            default=requested_default,
        )
    if key not in st.session_state:
        st.session_state[key] = st.session_state[initial_default_key]
    else:
        st.session_state[key] = safe_query_choice(
            st.session_state.get(key),
            labels,
            default=st.session_state[initial_default_key],
        )
    return str(st.session_state[initial_default_key])


def lazy_tabs(labels: list[str], *, default: str, key: str):
    """Return tabs configured for lazy rendering on modern Streamlit."""

    st = require_streamlit()
    stable_default = stable_tab_default(labels, default=default, key=key)
    try:
        return st.tabs(labels, default=stable_default, key=key, on_change="rerun")
    except TypeError:  # pragma: no cover - old Streamlit fallback
        return st.tabs(labels)


def active_tab_label(labels: list[str], tabs: object, *, default: str) -> str:
    """Return the currently open tab label when Streamlit exposes it."""

    for label, tab in zip(labels, tabs, strict=False):
        if getattr(tab, "open", None) is True:
            return label
    return default if default in labels else labels[0]


def tab_is_open(tab: object, *, fallback: bool = False) -> bool:
    """Return whether a lazily tracked tab should be rendered."""

    state = getattr(tab, "open", None)
    if state is None:
        return fallback
    return bool(state)


def render_metric_strip(metrics: list[MetricCard], *, columns: int | None = None) -> None:
    """Render standard metric cards with optional deltas and help text."""

    st = require_streamlit()
    count = min(columns or min(max(len(metrics), 1), 3), 3)
    for start in range(0, len(metrics), count):
        cols = st.columns(count)
        for index, metric in enumerate(metrics[start : start + count]):
            kwargs: dict[str, Any] = {
                "label": metric.label,
                "value": metric.value,
                "delta": metric.delta,
                "delta_color": metric.delta_color,
                "help": metric.help,
            }
            if metric.delta_description is not None:
                kwargs["delta_description"] = metric.delta_description
            cols[index].metric(**kwargs)


def render_data_health(
    frame: pd.DataFrame,
    *,
    source: str,
    date_column: str | None = None,
    value_columns: list[str] | None = None,
) -> None:
    """Render a compact data-health strip for graders and app users."""

    st = require_streamlit()
    summary = data_health_summary(
        frame,
        source=source,
        date_column=date_column,
        value_columns=value_columns,
    )
    cols = st.columns(len(summary))
    for index, (label, value) in enumerate(summary.items()):
        cols[index].caption(label)
        cols[index].write(f"**{value}**")


def _column_alignment(
    column: str,
    alignments: Mapping[str, str] | str | None,
) -> str | None:
    if alignments is None:
        return None
    if isinstance(alignments, str):
        return alignments
    return alignments.get(column)


def streamlit_column_config(
    frame: pd.DataFrame,
    *,
    alignments: Mapping[str, str] | str | None = None,
) -> dict[str, object]:
    """Build Streamlit column config objects from a display dataframe."""

    st = require_streamlit()
    config: dict[str, object] = {}
    for column, (kind, fmt) in display_column_config(frame).items():
        alignment = _column_alignment(column, alignments)
        if kind == "date":
            config[column] = st.column_config.DateColumn(
                column,
                format=fmt,
                alignment=alignment,
            )
        elif kind == "number":
            config[column] = st.column_config.NumberColumn(
                column,
                format=fmt,
                alignment=alignment,
            )
        elif alignment is not None:
            config[column] = st.column_config.TextColumn(column, alignment=alignment)
    return config


def render_compact_metric_strip(
    metrics: list[MetricCard],
    *,
    columns: int | None = None,
) -> None:
    """Render compact KPI cards for short or medium-length values."""

    st = require_streamlit()
    count = min(columns or min(max(len(metrics), 1), 3), 3)
    for start in range(0, len(metrics), count):
        cols = st.columns(count)
        for index, metric in enumerate(metrics[start : start + count]):
            with cols[index].container(border=True):
                st.caption(metric.label)
                st.markdown(
                    (
                        "<div style='font-size:1.35rem;font-weight:650;"
                        "line-height:1.2;overflow-wrap:anywhere;'>"
                        f"{metric.value}</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if metric.delta is not None:
                    st.caption(str(metric.delta))
                if metric.help:
                    st.caption(metric.help)


def render_display_table(
    frame: pd.DataFrame,
    *,
    labels: Mapping[object, str] | None = None,
    index_label: str = "Date",
    reset_index: bool = True,
    height: int | None = None,
    column_alignments: Mapping[str, str] | str | None = None,
) -> pd.DataFrame:
    """Render a presentation-safe dataframe and return the displayed frame."""

    st = require_streamlit()
    display = prepare_display_frame(
        frame,
        labels=labels,
        index_label=index_label,
        reset_index=reset_index,
    )
    kwargs: dict[str, object] = {
        "width": "stretch",
        "hide_index": True,
        "column_config": streamlit_column_config(
            display,
            alignments=column_alignments,
        ),
    }
    if height is not None:
        kwargs["height"] = height
    st.dataframe(display, **kwargs)
    return display


def render_csv_download(
    frame: pd.DataFrame,
    *,
    label: str,
    file_name: str,
    key: str,
) -> None:
    """Render a standard CSV download button."""

    st = require_streamlit()
    st.download_button(
        label,
        data=dataframe_csv(frame),
        file_name=file_name,
        mime="text/csv",
        key=key,
        icon=":material/download:",
    )
