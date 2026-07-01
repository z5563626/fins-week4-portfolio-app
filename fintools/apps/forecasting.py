"""Transparent forecasting helpers for student Streamlit apps."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

ForecastModel = Literal["naive", "drift", "ar", "ar1", "arma", "armax", "enet"]


@dataclass(frozen=True)
class ForecastResult:
    """Container returned by transparent forecasting helpers."""

    model: str
    observed: pd.Series
    forecast: pd.DataFrame
    residual_std: float
    details: dict[str, object] = field(default_factory=dict)


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    if clean.empty:
        raise ValueError("series has no numeric observations")
    return clean


def _clean_exog(exog: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if exog is None:
        return pd.DataFrame()
    frame = exog.to_frame() if isinstance(exog, pd.Series) else exog.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(how="all")
    return frame


def _align_series_and_exog(
    series: pd.Series,
    exog: pd.DataFrame | pd.Series | None,
) -> tuple[pd.Series, pd.DataFrame]:
    clean = _clean_series(series)
    frame = _clean_exog(exog)
    if frame.empty:
        return clean, frame
    merged = pd.concat([clean.rename("__target__"), frame], axis=1, join="inner").dropna()
    if merged.empty:
        raise ValueError("series and exogenous features have no overlapping observations")
    target = merged.pop("__target__")
    return target.astype(float), merged.astype(float)


def _infer_frequency(series: pd.Series) -> str:
    inferred = pd.infer_freq(series.index)
    if inferred:
        upper = inferred.upper()
        if upper.startswith("Q"):
            return "quarterly"
        if upper.startswith("M"):
            return "monthly"
        if upper.startswith("B"):
            return "business_day"
        if upper.startswith("D"):
            return "daily"
    if len(series.index) < 2:
        return "monthly"
    delta = series.index.to_series().diff().dropna().median()
    if pd.isna(delta):
        return "monthly"
    if delta >= pd.Timedelta(days=70):
        return "quarterly"
    if delta >= pd.Timedelta(days=25):
        return "monthly"
    if delta <= pd.Timedelta(days=3):
        weekdays = pd.Series(series.index.weekday, index=series.index)
        return "business_day" if float((weekdays < 5).mean()) >= 0.95 else "daily"
    return "daily"


def _default_candidate_lags(series: pd.Series) -> list[int]:
    frequency = _infer_frequency(series)
    if frequency == "quarterly":
        return [1, 2, 4]
    if frequency == "monthly":
        return [1, 3, 6, 12]
    if frequency == "business_day":
        return [1, 5, 21, 63]
    return [1, 2, 5]


def _default_candidate_orders(series: pd.Series) -> list[tuple[int, int]]:
    frequency = _infer_frequency(series)
    if frequency == "quarterly":
        return [(1, 0), (2, 0), (1, 1), (2, 1)]
    if frequency == "monthly":
        return [(1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
    return [(1, 0), (1, 1), (2, 0)]


def _forecast_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    inferred = pd.infer_freq(series.index)
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        return pd.date_range(series.index[-1] + offset, periods=horizon, freq=offset)
    if len(series.index) >= 2:
        delta = series.index.to_series().diff().dropna().median()
        weekdays = pd.Series(series.index.weekday, index=series.index)
        mostly_weekdays = float((weekdays < 5).mean()) >= 0.95
        if mostly_weekdays and pd.notna(delta) and delta <= pd.Timedelta(days=3):
            return pd.bdate_range(series.index[-1] + pd.offsets.BDay(), periods=horizon)
        if pd.notna(delta) and delta > pd.Timedelta(0):
            return pd.DatetimeIndex(
                [series.index[-1] + delta * step for step in range(1, horizon + 1)]
            )
    return pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")


def _interval_frame(
    index: pd.DatetimeIndex,
    center: np.ndarray,
    residual_std: float,
) -> pd.DataFrame:
    scale = max(float(residual_std), 1e-9)
    return pd.DataFrame(
        {
            "forecast": center,
            "lower": center - 1.96 * scale,
            "upper": center + 1.96 * scale,
        },
        index=index,
    )


def _attach_details(result: ForecastResult, **details: object) -> ForecastResult:
    merged = {**result.details, **details}
    return ForecastResult(
        model=result.model,
        observed=result.observed,
        forecast=result.forecast,
        residual_std=result.residual_std,
        details=merged,
    )


def naive_forecast(series: pd.Series, horizon: int = 12) -> ForecastResult:
    """Forecast the last observed value forward with empirical uncertainty."""

    clean = _clean_series(series)
    fitted = clean.shift(1).dropna()
    residuals = clean.loc[fitted.index] - fitted
    residual_std = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    index = _forecast_index(clean, horizon)
    center = np.repeat(float(clean.iloc[-1]), horizon)
    forecast = _interval_frame(index, center, residual_std)
    return ForecastResult("naive", clean, forecast, residual_std)


def drift_forecast(series: pd.Series, horizon: int = 12, *, window: int = 60) -> ForecastResult:
    """Forecast with a simple linear trend over the recent sample window."""

    clean = _clean_series(series)
    train = clean.tail(min(max(window, 6), len(clean)))
    if len(train) < 3 or train.nunique() <= 1:
        return naive_forecast(clean, horizon)
    x = np.arange(len(train), dtype=float)
    slope, intercept = np.polyfit(x, train.to_numpy(dtype=float), deg=1)
    fitted_values = intercept + slope * x
    residual_std = float(np.std(train.to_numpy(dtype=float) - fitted_values, ddof=1))
    future_x = np.arange(len(train), len(train) + horizon, dtype=float)
    center = intercept + slope * future_x
    forecast = _interval_frame(_forecast_index(clean, horizon), center, residual_std)
    return ForecastResult("drift", clean, forecast, residual_std)


def _fit_best_autoreg(
    train: pd.Series,
    *,
    candidate_lags: list[int],
) -> tuple[object | None, int | None]:
    best_fit = None
    best_lag = None
    best_bic = np.inf
    for lag in candidate_lags:
        if lag < 1 or len(train) <= lag + 4:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = AutoReg(train.to_numpy(dtype=float), lags=lag, old_names=False).fit()
        except Exception:
            continue
        bic = float(getattr(fit, "bic", np.inf))
        if bic < best_bic:
            best_bic = bic
            best_fit = fit
            best_lag = lag
    return best_fit, best_lag


def ar_forecast(
    series: pd.Series,
    horizon: int = 12,
    *,
    candidate_lags: list[int] | None = None,
    max_train: int = 240,
) -> ForecastResult:
    """Forecast with an autoregressive model chosen by BIC over candidate lags."""

    clean = _clean_series(series)
    train = clean.tail(min(max_train, len(clean)))
    lags = candidate_lags or _default_candidate_lags(train)
    fit, selected_lag = _fit_best_autoreg(train, candidate_lags=lags)
    if fit is None or selected_lag is None:
        return _attach_details(
            drift_forecast(clean, horizon),
            fallback_model="drift",
            candidate_lags=lags,
        )
    center = np.asarray(
        fit.predict(start=len(train), end=len(train) + horizon - 1),
        dtype=float,
    )
    residual_std = float(np.std(fit.resid, ddof=1)) if len(fit.resid) > 1 else 0.0
    forecast = _interval_frame(_forecast_index(clean, horizon), center, residual_std)
    return ForecastResult(
        "ar",
        clean,
        forecast,
        residual_std,
        details={"selected_lag": selected_lag, "candidate_lags": lags},
    )


def ar1_forecast(series: pd.Series, horizon: int = 12, *, max_train: int = 240) -> ForecastResult:
    """Forecast with an AR(1) baseline."""

    result = ar_forecast(series, horizon, candidate_lags=[1], max_train=max_train)
    return ForecastResult(
        "ar1",
        result.observed,
        result.forecast,
        result.residual_std,
        details=result.details,
    )


def _fit_best_arima(
    train: pd.Series,
    *,
    candidate_orders: list[tuple[int, int]],
    exog: pd.DataFrame | None = None,
) -> tuple[object | None, tuple[int, int] | None]:
    best_fit = None
    best_order = None
    best_bic = np.inf
    train_exog = None if exog is None or exog.empty else exog.astype(float)
    for p, q in candidate_orders:
        if p < 0 or q < 0 or len(train) <= max(p, q) + 6:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(
                    train,
                    exog=train_exog,
                    order=(p, 0, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
        except Exception:
            continue
        bic = float(getattr(fit, "bic", np.inf))
        if bic < best_bic:
            best_bic = bic
            best_fit = fit
            best_order = (p, q)
    return best_fit, best_order


def arma_forecast(
    series: pd.Series,
    horizon: int = 12,
    *,
    candidate_orders: list[tuple[int, int]] | None = None,
    max_train: int = 240,
) -> ForecastResult:
    """Forecast with an ARMA model selected by BIC."""

    clean = _clean_series(series)
    train = clean.tail(min(max_train, len(clean)))
    orders = candidate_orders or _default_candidate_orders(train)
    fit, selected_order = _fit_best_arima(train, candidate_orders=orders)
    if fit is None or selected_order is None:
        return _attach_details(
            ar_forecast(clean, horizon),
            fallback_model="ar",
            candidate_orders=orders,
        )
    forecast_values = np.asarray(fit.forecast(steps=horizon), dtype=float)
    residual_std = float(np.std(fit.resid, ddof=1)) if len(fit.resid) > 1 else 0.0
    forecast = _interval_frame(_forecast_index(clean, horizon), forecast_values, residual_std)
    return ForecastResult(
        "arma",
        clean,
        forecast,
        residual_std,
        details={"selected_order": selected_order, "candidate_orders": orders},
    )


def _prepare_future_exog(
    future_exog: pd.DataFrame | pd.Series | None,
    *,
    columns: list[str],
    horizon: int,
) -> pd.DataFrame:
    frame = _clean_exog(future_exog)
    if frame.empty:
        raise ValueError("future exogenous features are required for this model")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"future exogenous features are missing columns: {missing}")
    frame = frame.reindex(columns=columns).dropna()
    if len(frame) < horizon:
        raise ValueError("future exogenous features do not cover the requested horizon")
    return frame.iloc[:horizon].copy()


def armax_forecast(
    series: pd.Series,
    *,
    exog: pd.DataFrame | pd.Series | None,
    future_exog: pd.DataFrame | pd.Series | None,
    horizon: int = 1,
    candidate_orders: list[tuple[int, int]] | None = None,
    max_train: int = 240,
) -> ForecastResult:
    """Forecast with an ARMA model plus exogenous regressors."""

    if horizon != 1:
        raise ValueError("ARMA + exogenous forecasts are limited to one step in v1")
    clean, aligned_exog = _align_series_and_exog(series, exog)
    if aligned_exog.empty:
        raise ValueError("ARMA + exogenous requires non-empty exogenous features")
    train = clean.tail(min(max_train, len(clean)))
    train_exog = aligned_exog.reindex(train.index)
    orders = candidate_orders or _default_candidate_orders(train)
    fit, selected_order = _fit_best_arima(train, candidate_orders=orders, exog=train_exog)
    if fit is None or selected_order is None:
        raise ValueError("could not fit any ARMA + exogenous candidate")
    future = _prepare_future_exog(
        future_exog,
        columns=list(train_exog.columns),
        horizon=horizon,
    )
    forecast_values = np.asarray(fit.forecast(steps=horizon, exog=future), dtype=float)
    residual_std = float(np.std(fit.resid, ddof=1)) if len(fit.resid) > 1 else 0.0
    forecast = _interval_frame(future.index, forecast_values, residual_std)
    return ForecastResult(
        "armax",
        clean,
        forecast,
        residual_std,
        details={
            "selected_order": selected_order,
            "candidate_orders": orders,
            "exog_columns": list(train_exog.columns),
        },
    )


def _lagged_design(
    series: pd.Series,
    exog: pd.DataFrame,
    *,
    lags: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(index=series.index)
    for lag in sorted(set(lags)):
        frame[f"lag_{lag}"] = series.shift(lag)
    for column in exog.columns:
        frame[column] = exog[column]
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("lagged design matrix is empty")
    target = series.reindex(frame.index).astype(float)
    return frame.astype(float), target


def _future_design_row(
    series: pd.Series,
    future_exog: pd.DataFrame,
    *,
    lags: list[int],
) -> pd.DataFrame:
    if len(future_exog) != 1:
        raise ValueError("future design rows are limited to one step in v1")
    row = future_exog.copy().astype(float)
    for lag in sorted(set(lags)):
        if len(series) < lag:
            raise ValueError("series is too short for the requested lagged design")
        row[f"lag_{lag}"] = float(series.iloc[-lag])
    ordered = [f"lag_{lag}" for lag in sorted(set(lags))] + list(future_exog.columns)
    return row.reindex(columns=ordered).astype(float)


def _validation_window(length: int) -> int:
    return max(2, min(8, max(length // 5, 1)))


def _fit_enet(
    design: pd.DataFrame,
    target: pd.Series,
    *,
    alpha: float,
    l1_wt: float,
) -> tuple[object, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    means = design.mean()
    stds = design.std(ddof=0).replace(0.0, 1.0)
    scaled = (design - means) / stds
    model = sm.OLS(target.astype(float), sm.add_constant(scaled, has_constant="add"))
    penalty = np.array([0.0, *([alpha] * scaled.shape[1])], dtype=float)
    fit = model.fit_regularized(alpha=penalty, L1_wt=l1_wt, refit=True)
    return fit, means, stds, penalty, np.asarray(fit.predict(), dtype=float)


def enet_forecast(
    series: pd.Series,
    *,
    exog: pd.DataFrame | pd.Series | None,
    future_exog: pd.DataFrame | pd.Series | None,
    horizon: int = 1,
    candidate_lags: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = 0.5,
) -> ForecastResult:
    """Forecast with a lagged dynamic regression and elastic-net regularization."""

    if horizon != 1:
        raise ValueError("Elastic-net forecasts are limited to one step in v1")
    clean, aligned_exog = _align_series_and_exog(series, exog)
    if aligned_exog.empty:
        raise ValueError("Elastic-net requires non-empty exogenous features")
    lags = candidate_lags or _default_candidate_lags(clean)
    design, target = _lagged_design(clean, aligned_exog, lags=lags)
    validation_size = _validation_window(len(design))
    if len(design) <= validation_size + 2:
        raise ValueError("not enough observations for elastic-net validation")
    alphas = alpha_grid or [0.001, 0.01, 0.1, 1.0]
    train_design = design.iloc[:-validation_size]
    train_target = target.iloc[:-validation_size]
    valid_design = design.iloc[-validation_size:]
    valid_target = target.iloc[-validation_size:]

    best_alpha = None
    best_mae = np.inf
    for alpha in alphas:
        try:
            fit, means, stds, _, _ = _fit_enet(
                train_design,
                train_target,
                alpha=float(alpha),
                l1_wt=l1_wt,
            )
            scaled_valid = (valid_design - means) / stds
            preds = np.asarray(
                fit.predict(sm.add_constant(scaled_valid, has_constant="add")),
                dtype=float,
            )
            mae = float(np.mean(np.abs(valid_target.to_numpy(dtype=float) - preds)))
        except Exception:
            continue
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    if best_alpha is None:
        raise ValueError("could not fit any elastic-net candidate")

    fit, means, stds, _, fitted_values = _fit_enet(
        design,
        target,
        alpha=best_alpha,
        l1_wt=l1_wt,
    )
    future = _prepare_future_exog(
        future_exog,
        columns=list(aligned_exog.columns),
        horizon=horizon,
    )
    future_design = _future_design_row(clean, future, lags=lags)
    scaled_future = (future_design - means) / stds
    forecast_values = np.asarray(
        fit.predict(sm.add_constant(scaled_future, has_constant="add")),
        dtype=float,
    )
    residuals = target.to_numpy(dtype=float) - fitted_values
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    forecast = _interval_frame(future.index, forecast_values, residual_std)
    return ForecastResult(
        "enet",
        clean,
        forecast,
        residual_std,
        details={
            "selected_alpha": best_alpha,
            "alpha_grid": alphas,
            "l1_wt": l1_wt,
            "candidate_lags": lags,
            "exog_columns": list(aligned_exog.columns),
        },
    )


def forecast_series(
    series: pd.Series,
    *,
    horizon: int = 12,
    model: ForecastModel = "drift",
    exog: pd.DataFrame | pd.Series | None = None,
    future_exog: pd.DataFrame | pd.Series | None = None,
    candidate_lags: list[int] | None = None,
    candidate_orders: list[tuple[int, int]] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = 0.5,
) -> ForecastResult:
    """Dispatch to one of the transparent baseline forecasting models."""

    if model == "naive":
        return naive_forecast(series, horizon)
    if model == "drift":
        return drift_forecast(series, horizon)
    if model == "ar":
        return ar_forecast(series, horizon, candidate_lags=candidate_lags)
    if model == "ar1":
        return ar1_forecast(series, horizon)
    if model == "arma":
        return arma_forecast(series, horizon, candidate_orders=candidate_orders)
    if model == "armax":
        return armax_forecast(
            series,
            exog=exog,
            future_exog=future_exog,
            horizon=horizon,
            candidate_orders=candidate_orders,
        )
    if model == "enet":
        return enet_forecast(
            series,
            exog=exog,
            future_exog=future_exog,
            horizon=horizon,
            candidate_lags=candidate_lags,
            alpha_grid=alpha_grid,
            l1_wt=l1_wt,
        )
    raise ValueError(f"unknown forecast model: {model}")


def rolling_backtest(
    series: pd.Series,
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
    """Run a rolling-origin backtest for a transparent forecast."""

    clean = _clean_series(series)
    aligned_exog = pd.DataFrame()
    if exog is not None:
        clean, aligned_exog = _align_series_and_exog(clean, exog)
    if len(clean) <= min_train + horizon:
        return pd.DataFrame(columns=["actual", "forecast", "error", "absolute_error"])

    rows: list[dict[str, float | pd.Timestamp]] = []
    for end in range(min_train, len(clean) - horizon + 1, step):
        train = clean.iloc[:end]
        actual_date = clean.index[end + horizon - 1]
        actual = float(clean.iloc[end + horizon - 1])
        train_exog = None
        future_exog = None
        if not aligned_exog.empty:
            train_exog = aligned_exog.iloc[:end]
            future_exog = aligned_exog.iloc[end : end + horizon]
            if len(future_exog) < horizon:
                break
        forecast = forecast_series(
            train,
            horizon=horizon,
            model=model,
            exog=train_exog,
            future_exog=future_exog,
            candidate_lags=candidate_lags,
            candidate_orders=candidate_orders,
            alpha_grid=alpha_grid,
            l1_wt=l1_wt,
        )
        prediction = float(forecast.forecast["forecast"].iloc[-1])
        rows.append(
            {
                "date": actual_date,
                "actual": actual,
                "forecast": prediction,
                "error": actual - prediction,
                "absolute_error": abs(actual - prediction),
            }
        )
    return pd.DataFrame(rows).set_index("date")
