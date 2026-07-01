"""Publication-grade Plotly helpers for Streamlit coursework apps."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from fintools.figures import recession_windows_for_range

from .forecasting import ForecastResult
from .targets import TargetForecastResult

APP_COLORS = {
    "observed": "#355C7D",
    "forecast": "#A51C30",
    "band": "rgba(165, 28, 48, 0.16)",
    "grid": "#E2E6EA",
    "recession": "#D9DDE2",
    "muted": "#6B6F76",
    "axis": "#262A33",
}


def add_nber_recession_vrects(
    fig: go.Figure,
    *,
    start: object,
    end: object,
    opacity: float = 0.42,
) -> go.Figure:
    """Add light gray NBER recession shading to a Plotly figure."""

    for window_start, window_end, _label in recession_windows_for_range(start, end):
        fig.add_vrect(
            x0=window_start,
            x1=window_end,
            fillcolor=APP_COLORS["recession"],
            opacity=opacity,
            layer="below",
            line_width=0,
        )
    return fig


def add_time_axis_controls(
    fig: go.Figure,
    *,
    range_slider: bool = False,
    range_selector: bool = True,
) -> go.Figure:
    """Add standard time-axis controls to an interactive Plotly chart."""

    axis_update: dict[str, object] = {
        "showgrid": False,
        "tickformatstops": [
            {"dtickrange": [None, 86_400_000 * 90], "value": "%b %Y"},
            {"dtickrange": [86_400_000 * 90, "M12"], "value": "%b\n%Y"},
            {"dtickrange": ["M12", None], "value": "%Y"},
        ],
        "rangeslider": {"visible": range_slider},
    }
    if range_selector:
        axis_update["rangeselector"] = {
            "x": 0,
            "xanchor": "left",
            "y": 1.14,
            "yanchor": "top",
            "buttons": [
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
                {"count": 20, "label": "20Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
        }
    fig.update_xaxes(**axis_update)
    return fig


def _has_visible_legend_entries(fig: go.Figure) -> bool:
    """Return whether a figure has named traces that will render in a legend."""

    for trace in fig.data:
        if getattr(trace, "name", None) and getattr(trace, "showlegend", None) is not False:
            return True
    return False


def apply_app_plotly_theme(
    fig: go.Figure,
    *,
    yaxis_title: str | None = None,
    height: int = 500,
    legend_y: float | None = None,
    range_slider: bool = False,
    range_selector: bool = True,
) -> go.Figure:
    """Apply the standard interactive app chart style."""

    has_legend = _has_visible_legend_entries(fig)
    use_range_selector = range_selector and not has_legend
    use_range_slider = range_slider or (range_selector and has_legend)
    resolved_legend_y = (1.08 if has_legend else -0.22) if legend_y is None else legend_y
    bottom_margin = 76 if use_range_slider else 44
    top_margin = 64 if resolved_legend_y > 1 or use_range_selector else 34
    legend_x = 1 if has_legend else 0
    legend_xanchor = "right" if has_legend else "left"
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 32, "r": 28, "t": top_margin, "b": bottom_margin},
        legend={
            "orientation": "h",
            "y": resolved_legend_y,
            "x": legend_x,
            "xanchor": legend_xanchor,
            "yanchor": "top" if resolved_legend_y < 0 else "bottom",
            "title": None,
        },
        hovermode="x unified",
        font={"color": APP_COLORS["axis"]},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    add_time_axis_controls(
        fig,
        range_slider=use_range_slider,
        range_selector=use_range_selector,
    )
    fig.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor=APP_COLORS["grid"],
        zerolinecolor=APP_COLORS["grid"],
        automargin=True,
    )
    return fig


def _series_range(series: pd.Series, extra_dates: Sequence[object] = ()) -> tuple[object, object]:
    dates = [pd.Timestamp(series.index.min()), pd.Timestamp(series.index.max())]
    dates.extend(pd.Timestamp(item) for item in extra_dates)
    return min(dates), max(dates)


def forecast_figure(
    result: ForecastResult,
    *,
    units: str,
    indicator_name: str | None = None,
    shade_recessions: bool = True,
    range_slider: bool = False,
) -> go.Figure:
    """Build a standard observed/forecast/interval Plotly figure."""

    observed = result.observed
    forecast = result.forecast
    forecast_x = [observed.index[-1], *forecast.index]
    forecast_y = [float(observed.iloc[-1]), *forecast["forecast"].astype(float)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed,
            name="Observed",
            mode="lines",
            line={"color": APP_COLORS["observed"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Observed: %{y:.2f}<extra></extra>",
            legendrank=1,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
            fill="toself",
            fillcolor=APP_COLORS["band"],
            line={"color": "rgba(255, 255, 255, 0)"},
            name="Approx. 95% band",
            hoverinfo="skip",
            legendrank=3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            name="Forecast",
            mode="lines",
            line={"color": APP_COLORS["forecast"], "width": 3},
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
            legendrank=2,
        )
    )
    fig.add_shape(
        type="line",
        x0=observed.index[-1],
        x1=observed.index[-1],
        y0=0,
        y1=1,
        yref="paper",
        line={"color": APP_COLORS["muted"], "dash": "dot", "width": 1},
    )
    start, end = _series_range(observed, forecast.index)
    if shade_recessions:
        add_nber_recession_vrects(fig, start=start, end=end)
    fig.update_layout(
        title={"text": indicator_name or "", "x": 0, "xanchor": "left"} if indicator_name else None,
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title=units,
        height=500,
        range_slider=range_slider,
        range_selector=True,
    )
    fig.update_xaxes(range=[start, end])
    return fig


def target_forecast_figure(
    result: TargetForecastResult,
    *,
    indicator_name: str | None = None,
    shade_recessions: bool = True,
    range_slider: bool = False,
) -> go.Figure:
    """Build a forecast figure from a metadata-aware forecast result."""

    observed = result.observed_level if result.spec.display_level else result.observed_target
    forecast = result.display_forecast
    units = result.spec.units if result.spec.display_level else result.spec.target_units
    units = units or result.spec.units
    forecast_x = [observed.index[-1], *forecast.index]
    forecast_y = [float(observed.iloc[-1]), *forecast["forecast"].astype(float)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed,
            name="Observed",
            mode="lines",
            line={"color": APP_COLORS["observed"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Observed: %{y:.2f}<extra></extra>",
            legendrank=1,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
            fill="toself",
            fillcolor=APP_COLORS["band"],
            line={"color": "rgba(255, 255, 255, 0)"},
            name="Approx. 95% band",
            hoverinfo="skip",
            legendrank=3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            name="Implied level path" if result.spec.display_level else "Forecast",
            mode="lines",
            line={"color": APP_COLORS["forecast"], "width": 3},
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
            legendrank=2,
        )
    )
    fig.add_shape(
        type="line",
        x0=observed.index[-1],
        x1=observed.index[-1],
        y0=0,
        y1=1,
        yref="paper",
        line={"color": APP_COLORS["muted"], "dash": "dot", "width": 1},
    )
    start, end = _series_range(observed, forecast.index)
    if shade_recessions:
        add_nber_recession_vrects(fig, start=start, end=end)
    fig.update_layout(
        title={"text": indicator_name or result.spec.label, "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title=units,
        height=500,
        range_slider=range_slider,
        range_selector=True,
    )
    fig.update_xaxes(range=[start, end])
    return fig


def backtest_figure(
    backtest: pd.DataFrame,
    *,
    units: str,
    shade_recessions: bool = True,
    range_slider: bool = False,
) -> go.Figure:
    """Build a standard actual-versus-forecast backtest Plotly figure."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["actual"],
            name="Actual",
            mode="lines",
            line={"color": APP_COLORS["observed"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra></extra>",
            legendrank=1,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["forecast"],
            name="Forecast",
            mode="lines",
            line={"color": APP_COLORS["forecast"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
            legendrank=2,
        )
    )
    if not backtest.empty and shade_recessions:
        add_nber_recession_vrects(
            fig,
            start=backtest.index.min(),
            end=backtest.index.max(),
            opacity=0.36,
        )
    apply_app_plotly_theme(
        fig,
        yaxis_title=units,
        height=430,
        range_slider=range_slider,
        range_selector=True,
    )
    return fig
