"""Streamlit app helpers — deploy-only subset for fins-week4-portfolio-app."""

from __future__ import annotations

from .plotly import (
    add_nber_recession_vrects,
    add_time_axis_controls,
    apply_app_plotly_theme,
    backtest_figure,
    forecast_figure,
    target_forecast_figure,
)
from .streamlit_ui import (
    MetricCard,
    active_tab_label,
    configure_page,
    lazy_tabs,
    query_choice,
    query_int,
    render_compact_metric_strip,
    render_csv_download,
    render_data_health,
    render_display_table,
    render_metric_strip,
    stable_tab_default,
    streamlit_column_config,
    sync_query_params,
    tab_is_open,
)

__all__ = [
    "MetricCard",
    "active_tab_label",
    "add_nber_recession_vrects",
    "add_time_axis_controls",
    "apply_app_plotly_theme",
    "backtest_figure",
    "configure_page",
    "forecast_figure",
    "lazy_tabs",
    "query_choice",
    "query_int",
    "render_compact_metric_strip",
    "render_csv_download",
    "render_data_health",
    "render_display_table",
    "render_metric_strip",
    "stable_tab_default",
    "streamlit_column_config",
    "sync_query_params",
    "tab_is_open",
    "target_forecast_figure",
]
