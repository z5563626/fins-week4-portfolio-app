"""Forecast-target metadata for Streamlit insight apps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .forecasting import ForecastModel, forecast_series, rolling_backtest

ForecastTarget = Literal["level", "change", "log_change", "annualized_growth", "yoy_growth", "none"]
SeriesFrequency = Literal["daily", "monthly", "quarterly"]


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata that defines how an app should model and present one series."""

    series_id: str
    label: str
    units: str
    target: ForecastTarget = "level"
    target_label: str | None = None
    target_units: str | None = None
    role: str = "market"
    frequency: SeriesFrequency = "daily"
    allow_forecast: bool = True
    display_level: bool = True
    release_lag_days: int = 0
    caveat: str | None = None


@dataclass(frozen=True)
class TargetForecastResult:
    """Forecast output after applying a series-specific target transform."""

    spec: SeriesSpec
    model: str
    observed_level: pd.Series
    observed_target: pd.Series
    target_forecast: pd.DataFrame
    display_forecast: pd.DataFrame
    residual_std: float
    details: dict[str, object]


def _clean_level(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    if clean.empty:
        raise ValueError("series has no numeric observations")
    return clean


def target_name(spec: SeriesSpec) -> str:
    """Return the app-facing forecast target label."""

    if spec.target_label:
        return spec.target_label
    if spec.target == "change":
        return f"Change in {spec.label}"
    if spec.target == "log_change":
        return f"Log change in {spec.label}"
    if spec.target == "annualized_growth":
        return f"Annualized growth in {spec.label}"
    if spec.target == "yoy_growth":
        return f"Year-over-year growth in {spec.label}"
    return spec.label


def target_units(spec: SeriesSpec) -> str:
    """Return app-facing units for the modeled target."""

    if spec.target_units:
        return spec.target_units
    if spec.target in {"annualized_growth", "yoy_growth", "log_change"}:
        return "Percent"
    return spec.units


def availability_dates(index: pd.Index, spec: SeriesSpec) -> pd.DatetimeIndex:
    """Shift observation dates to approximate when users could have seen them."""

    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if spec.release_lag_days <= 0:
        return dates
    return dates + pd.DateOffset(days=spec.release_lag_days)


def build_forecast_target(series: pd.Series, spec: SeriesSpec) -> pd.Series:
    """Transform raw observations into the economically meaningful target."""

    if spec.target == "none" or not spec.allow_forecast:
        raise ValueError(f"{spec.label} is marked as not forecastable")

    level = _clean_level(series)
    if spec.target == "level":
        return level
    if spec.target == "change":
        return level.diff().dropna()
    if spec.target == "log_change":
        prior = level.shift(1)
        return (np.log(level / prior) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if spec.target == "annualized_growth":
        return ((level / level.shift(1)) ** 4 - 1.0).mul(100.0).dropna()
    if spec.target == "yoy_growth":
        return level.pct_change(periods=4).mul(100.0).dropna()
    raise ValueError(f"unknown forecast target: {spec.target}")


def _annualized_to_period_growth(values: pd.Series) -> pd.Series:
    base = 1.0 + pd.to_numeric(values, errors="coerce").astype(float).div(100.0)
    base = base.clip(lower=1e-9)
    return base.pow(0.25) - 1.0


def reconstruct_implied_level(
    series: pd.Series,
    target_forecast: pd.DataFrame,
    spec: SeriesSpec,
    *,
    residual_std: float = 0.0,
) -> pd.DataFrame:
    """Convert target forecasts into the level path users expect to inspect."""

    level = _clean_level(series)
    frame = target_forecast[["forecast", "lower", "upper"]].astype(float).copy()
    if frame.empty:
        return frame
    latest = float(level.iloc[-1])

    if spec.target == "level":
        return frame

    if spec.target == "change":
        center = latest + frame["forecast"].cumsum()
        steps = pd.Series(np.arange(1, len(frame) + 1, dtype=float), index=frame.index)
        scale = max(float(residual_std), 0.0) * np.sqrt(steps)
        return pd.DataFrame(
            {
                "forecast": center,
                "lower": center - 1.96 * scale,
                "upper": center + 1.96 * scale,
            },
            index=frame.index,
        )

    if spec.target == "log_change":
        center = latest * np.exp(frame["forecast"].cumsum() / 100.0)
        lower = latest * np.exp(frame["lower"].cumsum() / 100.0)
        upper = latest * np.exp(frame["upper"].cumsum() / 100.0)
        return pd.DataFrame(
            {"forecast": center, "lower": lower, "upper": upper},
            index=frame.index,
        )

    if spec.target == "annualized_growth":
        growth_center = _annualized_to_period_growth(frame["forecast"])
        growth_lower = _annualized_to_period_growth(frame["lower"])
        growth_upper = _annualized_to_period_growth(frame["upper"])
        return pd.DataFrame(
            {
                "forecast": latest * (1.0 + growth_center).cumprod(),
                "lower": latest * (1.0 + growth_lower).cumprod(),
                "upper": latest * (1.0 + growth_upper).cumprod(),
            },
            index=frame.index,
        )

    return frame


def forecast_series_spec(
    series: pd.Series,
    spec: SeriesSpec,
    *,
    horizon: int = 12,
    model: ForecastModel = "drift",
    exog: pd.DataFrame | pd.Series | None = None,
    future_exog: pd.DataFrame | pd.Series | None = None,
    candidate_lags: list[int] | None = None,
    candidate_orders: list[tuple[int, int]] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = 0.5,
) -> TargetForecastResult:
    """Forecast a series using its app metadata and return display-ready paths."""

    observed_level = _clean_level(series)
    observed_target = build_forecast_target(observed_level, spec)
    result = forecast_series(
        observed_target,
        horizon=horizon,
        model=model,
        exog=exog,
        future_exog=future_exog,
        candidate_lags=candidate_lags,
        candidate_orders=candidate_orders,
        alpha_grid=alpha_grid,
        l1_wt=l1_wt,
    )
    display_forecast = reconstruct_implied_level(
        observed_level,
        result.forecast,
        spec,
        residual_std=result.residual_std,
    )
    return TargetForecastResult(
        spec=spec,
        model=result.model,
        observed_level=observed_level,
        observed_target=result.observed,
        target_forecast=result.forecast,
        display_forecast=display_forecast,
        residual_std=result.residual_std,
        details=result.details,
    )


def rolling_backtest_spec(
    series: pd.Series,
    spec: SeriesSpec,
    *,
    model: ForecastModel = "drift",
    horizon: int = 1,
    min_train: int = 36,
    step: int = 3,
    exog: pd.DataFrame | pd.Series | None = None,
    candidate_lags: list[int] | None = None,
    candidate_orders: list[tuple[int, int]] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = 0.5,
) -> pd.DataFrame:
    """Backtest the economically meaningful target and add implied levels."""

    level = _clean_level(series)
    target = build_forecast_target(level, spec)
    backtest = rolling_backtest(
        target,
        model=model,
        horizon=horizon,
        min_train=min_train,
        step=step,
        exog=exog,
        candidate_lags=candidate_lags,
        candidate_orders=candidate_orders,
        alpha_grid=alpha_grid,
        l1_wt=l1_wt,
    )
    if backtest.empty:
        return backtest

    actual_level = level.reindex(backtest.index)
    if spec.target == "level":
        forecast_level = backtest["forecast"].astype(float)
    else:
        previous_level = level.shift(horizon).reindex(backtest.index)
        if spec.target == "change":
            forecast_level = previous_level + backtest["forecast"].astype(float)
        elif spec.target == "log_change":
            forecast_level = previous_level * np.exp(backtest["forecast"].astype(float) / 100.0)
        elif spec.target == "annualized_growth":
            growth = _annualized_to_period_growth(backtest["forecast"].astype(float))
            forecast_level = previous_level * (1.0 + growth)
        else:
            forecast_level = pd.Series(np.nan, index=backtest.index)

    out = backtest.copy()
    out["actual_level"] = actual_level
    out["forecast_level"] = forecast_level
    out["level_error"] = out["actual_level"] - out["forecast_level"]
    out["absolute_level_error"] = out["level_error"].abs()
    return out


def forecastable_specs(specs: dict[str, SeriesSpec]) -> dict[str, SeriesSpec]:
    """Return specs that are permitted to appear in forecast controls."""

    return {
        key: spec for key, spec in specs.items() if spec.allow_forecast and spec.target != "none"
    }


def week2_market_specs() -> dict[str, SeriesSpec]:
    """Return Week 2 market series metadata."""

    return {
        "DGS10": SeriesSpec(
            "DGS10",
            "10-Year Treasury",
            "Percent",
            target="change",
            target_label="Daily change in the 10-year Treasury yield",
            target_units="Percentage points",
            role="rate",
            frequency="daily",
        ),
        "DGS2": SeriesSpec(
            "DGS2",
            "2-Year Treasury",
            "Percent",
            target="change",
            target_label="Daily change in the 2-year Treasury yield",
            target_units="Percentage points",
            role="rate",
            frequency="daily",
        ),
        "DTB3": SeriesSpec(
            "DTB3",
            "3-Month Treasury Bill",
            "Percent",
            target="change",
            target_label="Daily change in the 3-month Treasury bill yield",
            target_units="Percentage points",
            role="rate",
            frequency="daily",
        ),
        "T10Y2Y": SeriesSpec(
            "T10Y2Y",
            "10Y-2Y Treasury Spread",
            "Percentage points",
            target="change",
            target_label="Daily change in the 10Y-2Y Treasury spread",
            target_units="Percentage points",
            role="spread",
            frequency="daily",
        ),
        "BAMLH0A0HYM2": SeriesSpec(
            "BAMLH0A0HYM2",
            "High-Yield OAS",
            "Percent",
            target="change",
            target_label="Daily change in high-yield OAS",
            target_units="Percentage points",
            role="spread",
            frequency="daily",
        ),
        "VIXCLS": SeriesSpec(
            "VIXCLS",
            "VIX Index",
            "Percent",
            target="none",
            role="risk",
            frequency="daily",
            allow_forecast=False,
            caveat=(
                "VIX is shown as volatility context; it is not forecast with "
                "the simple baseline models."
            ),
        ),
    }


def week2_gdp_specs() -> dict[str, SeriesSpec]:
    """Return Week 2 quarterly macro series metadata."""

    return {
        "GDPC1": SeriesSpec(
            "GDPC1",
            "Real GDP",
            "Billions of chained 2017 dollars",
            target="annualized_growth",
            target_label="Annualized quarterly real GDP growth",
            target_units="Percent",
            role="macro",
            frequency="quarterly",
            release_lag_days=30,
            caveat=(
                "GDP is released with a lag and is revised. Use latest-available "
                "vintages for serious nowcasting work."
            ),
        )
    }
