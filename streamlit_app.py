"""50-stock in-sample portfolio optimizer — Week 4 DFF.

Run from the repo root:
    streamlit run fins2026/week4/scratch/week4_dff_lecture_scripts/app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_APP_DIR, *_APP_DIR.parents] if (p / "fintools").is_dir()),
    _APP_DIR,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import functools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco
import streamlit as st

import data_access
from fintools.apps import (
    MetricCard,
    active_tab_label,
    add_nber_recession_vrects,
    apply_app_plotly_theme,
    configure_page,
    lazy_tabs,
    query_choice,
    render_compact_metric_strip,
    render_csv_download,
    render_display_table,
    sync_query_params,
    tab_is_open,
)

# ── constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS = 252
SQRT_252 = np.sqrt(TRADING_DAYS)
RISK_FREE = 0.0  # daily risk-free (set to 0 for simplicity)

PORTFOLIO_LABELS = ["Equal-weight", "Minimum variance", "Mean-variance"]
COLORS = {
    "Equal-weight": "#7A746B",
    "Minimum variance": "#4E8B84",
    "Mean-variance": "#8E3B46",
}

TABS = ["Overview", "Growth of $1", "Efficient Frontier", "Methodology"]
DEFAULT_TAB = "Overview"

CONSTRAINT_LABELS = {"long_only": "Long-only (no short selling)", "unconstrained": "Unconstrained"}

_USE_SYNTHETIC = os.environ.get("PORTFOLIO_APP_USE_SYNTHETIC") == "1"


# ── synthetic data for smoke tests ────────────────────────────────────────────

def _synthetic_returns() -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(12)]
    dates = pd.date_range("2020-01-02", "2023-12-29", freq="B")
    rets = pd.DataFrame(
        rng.normal(0.0004, 0.015, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    sectors = ["Technology", "Financials", "Healthcare"]
    sector_map = {t: sectors[i % 3] for i, t in enumerate(tickers)}
    return rets, sector_map


# ── data loading ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_returns() -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (daily_returns_wide, sector_map) for the 50-stock universe."""
    if _USE_SYNTHETIC:
        return _synthetic_returns()
    eq = data_access.load_equity_prices()
    prices = (
        eq.sort_values(["ticker", "date"])
        .pivot(index="date", columns="ticker", values="adjClose")
    )
    rets = prices.pct_change().dropna(how="all").dropna(axis=1, how="any")
    sector_map = (
        eq[["ticker", "sector"]].drop_duplicates()
        .set_index("ticker")["sector"]
        .to_dict()
    )
    return rets, sector_map


# ── portfolio math ─────────────────────────────────────────────────────────────

def _ew(n: int) -> np.ndarray:
    return np.ones(n) / n


def _mv_closed(cov: np.ndarray) -> np.ndarray:
    ones = np.ones(cov.shape[0])
    v = np.linalg.solve(cov, ones)
    return v / (ones @ v)


def _tan_closed(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    ones = np.ones(len(mean))
    v = np.linalg.solve(cov, mean - RISK_FREE * ones)
    return v / (ones @ v)


def _mv_long(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    res = sco.minimize(
        lambda w: float(w @ cov @ w),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def _tan_long(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mean)

    def neg_sharpe(w: np.ndarray) -> float:
        exc = float(w @ mean - RISK_FREE)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        return -(exc / vol) if vol > 1e-12 else 1e9

    res = sco.minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def solve_weights(
    tickers: list[str],
    returns: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    """Return a tickers × 3 weight DataFrame for the three portfolios."""
    arr = returns[tickers].to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)
    n = len(tickers)
    ew = _ew(n)
    if mode == "long_only":
        mv = _mv_long(cov)
        tan = _tan_long(mean, cov)
    else:
        mv = _mv_closed(cov)
        tan = _tan_closed(mean, cov)
    return pd.DataFrame(
        {"Equal-weight": ew, "Minimum variance": mv, "Mean-variance": tan},
        index=tickers,
    )


def scorecard(
    tickers: list[str],
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Return annualized performance metrics for each portfolio."""
    port_rets = returns[tickers] @ weights_df
    growth = (1.0 + port_rets).cumprod()
    rows = []
    for col in PORTFOLIO_LABELS:
        r = port_rets[col]
        ann_ret = float(r.mean() * TRADING_DAYS * 100)
        ann_vol = float(r.std(ddof=1) * SQRT_252 * 100)
        sharpe = float(r.mean() / r.std(ddof=1) * SQRT_252) if r.std() > 0 else np.nan
        mdd = float((growth[col] / growth[col].cummax() - 1.0).min() * 100)
        rows.append({
            "Portfolio": col,
            "Ann. return (%)": round(ann_ret, 1),
            "Ann. volatility (%)": round(ann_vol, 1),
            "Sharpe ratio": round(sharpe, 2),
            "Max drawdown (%)": round(mdd, 1),
        })
    return pd.DataFrame(rows)


def _frontier_curve(
    mean: np.ndarray, cov: np.ndarray, *, n_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical unconstrained efficient frontier, annualized."""
    ones = np.ones(len(mean))
    a = float(ones @ np.linalg.solve(cov, ones))
    b = float(ones @ np.linalg.solve(cov, mean))
    c = float(mean @ np.linalg.solve(cov, mean))
    det = a * c - b * b
    targets = np.linspace(b / a, float(mean.max()), n_points)
    vols = np.sqrt(np.clip((a * targets**2 - 2 * b * targets + c) / det, 0.0, None))
    return targets * TRADING_DAYS * 100, vols * SQRT_252 * 100


# ── figures ────────────────────────────────────────────────────────────────────

def _weights_figure(weights_df: pd.DataFrame, portfolio: str) -> go.Figure:
    w = weights_df[portfolio].sort_values()
    bar_colors = [
        "rgba(200,50,50,0.75)" if v < 0 else COLORS[portfolio]
        for v in w.values
    ]
    fig = go.Figure(go.Bar(
        x=w.values * 100,
        y=w.index.tolist(),
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=0, line_color="#999", line_dash="dot", line_width=1)
    chart_height = max(380, len(w) * 16 + 120)
    fig.update_layout(
        title=f"{portfolio} portfolio weights",
        height=chart_height,
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 80, "r": 30, "t": 56, "b": 44},
        font={"color": "#262A33"},
        hovermode="y",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E6EA", title="Weight (%)")
    fig.update_yaxes(showgrid=False, automargin=True, tickfont={"size": 10})
    return fig


def _growth_figure(
    tickers: list[str],
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
) -> go.Figure:
    port_rets = returns[tickers] @ weights_df
    growth = (1.0 + port_rets).cumprod()
    fig = go.Figure()
    for col in PORTFOLIO_LABELS:
        fig.add_trace(go.Scatter(
            x=port_rets.index,
            y=growth[col],
            mode="lines",
            name=col,
            line={"width": 2.2, "color": COLORS[col]},
            hovertemplate="%{x|%Y-%m-%d}  $%{y:.2f}<extra></extra>",
        ))
    add_nber_recession_vrects(
        fig,
        start=port_rets.index.min(),
        end=port_rets.index.max(),
    )
    apply_app_plotly_theme(fig, yaxis_title="Growth of $1", height=480)
    fig.update_yaxes(type="log")
    return fig


def _frontier_figure(
    tickers: list[str],
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
    sc: pd.DataFrame,
) -> go.Figure:
    arr = returns[tickers].to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)
    asset_vol = arr.std(axis=0) * SQRT_252 * 100
    asset_ret = mean * TRADING_DAYS * 100

    fig = go.Figure()

    # Individual stock scatter
    text_labels = tickers if len(tickers) <= 25 else None
    fig.add_trace(go.Scatter(
        x=asset_vol,
        y=asset_ret,
        mode="markers+text" if text_labels else "markers",
        text=text_labels,
        textposition="top center",
        textfont={"size": 9},
        marker={"size": 7, "color": "rgba(100,100,100,0.35)", "line": {"width": 1, "color": "rgba(100,100,100,0.6)"}},
        name="Individual stocks",
        hovertemplate="%{text}<br>Vol: %{x:.1f}%  Return: %{y:.1f}%<extra></extra>",
    ))

    # Analytical frontier curve (unconstrained)
    try:
        fret, fvol = _frontier_curve(mean, cov)
        fig.add_trace(go.Scatter(
            x=fvol,
            y=fret,
            mode="lines",
            name="Efficient frontier (unconstrained)",
            line={"color": "#2F455C", "width": 2.2},
            hovertemplate="Vol: %{x:.1f}%  Return: %{y:.1f}%<extra></extra>",
        ))
    except np.linalg.LinAlgError:
        pass

    # Portfolio points
    sc_idx = sc.set_index("Portfolio")
    offsets = {"Equal-weight": (22, -24), "Minimum variance": (22, 20), "Mean-variance": (22, -24)}
    for label in PORTFOLIO_LABELS:
        row = sc_idx.loc[label]
        pvol = float(row["Ann. volatility (%)"])
        pret = float(row["Ann. return (%)"])
        dx, dy = offsets.get(label, (22, -20))
        fig.add_trace(go.Scatter(
            x=[pvol], y=[pret],
            mode="markers",
            marker={"size": 14, "color": COLORS[label], "line": {"width": 2, "color": "white"}},
            name=label,
            hovertemplate=f"{label}<br>Vol: %{{x:.1f}}%  Return: %{{y:.1f}}%<extra></extra>",
        ))
        fig.add_annotation(
            x=pvol, y=pret, ax=dx, ay=dy,
            xref="x", yref="y", axref="pixel", ayref="pixel",
            text=label,
            showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=COLORS[label],
            bgcolor="rgba(255,255,255,0.92)", bordercolor=COLORS[label], borderwidth=1,
            font={"size": 11, "color": COLORS[label]},
        )

    fig.update_layout(
        title="Mean-variance efficient frontier (in-sample, annualized)",
        height=540,
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        font={"color": "#262A33"},
        hovermode="closest",
        legend={
            "orientation": "h", "y": 1.08, "x": 1,
            "xanchor": "right", "yanchor": "bottom",
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E6EA", title="Annualized volatility (%)")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E6EA", title="Annualized return (%)")
    return fig


# ── app ────────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_page("50-Stock Portfolio Optimizer")

    with st.spinner("Loading equity data …"):
        returns, sector_map = load_returns()
    all_tickers = sorted(returns.columns.tolist())
    sectors = sorted({sector_map.get(t, "Unknown") for t in all_tickers})

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
        sector_sel = st.multiselect(
            "Sector filter",
            sectors,
            default=sectors,
            help="Limit the opportunity set to selected sectors.",
        )
        universe = sorted(t for t in all_tickers if sector_map.get(t, "") in sector_sel)
        selected = st.multiselect(
            "Stocks",
            universe,
            default=universe,
            help="Choose stocks to include in the portfolio.",
        )
        mode = st.radio(
            "Optimization mode",
            list(CONSTRAINT_LABELS),
            format_func=lambda k: CONSTRAINT_LABELS[k],
            key="mode",
            help="Long-only forces all weights ≥ 0. Unconstrained allows short positions.",
        )

    if len(selected) < 2:
        st.warning("Select at least 2 stocks to build portfolios.")
        return

    # ── compute ────────────────────────────────────────────────────────────────
    with st.spinner("Solving portfolios …"):
        w_df = solve_weights(selected, returns, mode=mode)
        sc = scorecard(selected, w_df, returns)

    # ── header ─────────────────────────────────────────────────────────────────
    st.title("50-Stock In-Sample Portfolio Optimizer")
    ret_sel = returns[selected]
    start_date = ret_sel.index.min().strftime("%Y-%m-%d")
    end_date = ret_sel.index.max().strftime("%Y-%m-%d")
    st.caption(
        f"Comparing equal-weight, minimum-variance, and mean-variance portfolios "
        f"across {len(selected)} stocks — {start_date} to {end_date}. "
        f"In-sample only ({CONSTRAINT_LABELS[mode]})."
    )

    # ── metric strip ──────────────────────────────────────────────────────────
    mv_row = sc.loc[sc["Portfolio"] == "Mean-variance"].iloc[0]
    render_compact_metric_strip(
        [
            MetricCard("Stocks", str(len(selected))),
            MetricCard("Mode", "Long-only" if mode == "long_only" else "Unconstrained"),
            MetricCard("MV Sharpe", f"{mv_row['Sharpe ratio']:.2f}", help="Mean-variance Sharpe ratio (rf = 0)"),
        ],
        columns=3,
    )

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab_default = query_choice("view", TABS, default=DEFAULT_TAB)
    tabs = lazy_tabs(TABS, default=tab_default, key="portfolio_tab")
    tab_ov, tab_gr, tab_ef, tab_me = tabs
    active = active_tab_label(TABS, tabs, default=tab_default)

    # Overview ----------------------------------------------------------------
    if tab_is_open(tab_ov, fallback=active == "Overview"):
        with tab_ov:
            st.subheader("Portfolio weights")
            focus = st.radio(
                "Show weights for",
                PORTFOLIO_LABELS,
                horizontal=True,
                key="weight_focus",
            )
            st.plotly_chart(_weights_figure(w_df, focus), width="stretch")

            st.subheader("Weights — all portfolios (%)")
            w_pct = w_df.mul(100).round(2)
            w_pct.index.name = "Ticker"
            render_display_table(w_pct.reset_index(), reset_index=False)

            st.subheader("Scorecard")
            render_display_table(sc, reset_index=False)
            render_csv_download(
                sc,
                label="Download scorecard",
                file_name="portfolio_scorecard.csv",
                key="dl_scorecard",
            )

    # Growth of $1 ------------------------------------------------------------
    if tab_is_open(tab_gr, fallback=active == "Growth of $1"):
        with tab_gr:
            st.subheader("In-sample growth of $1")
            st.markdown(
                "Each portfolio starts at **\\$1** on the first trading day and is "
                "rebalanced daily. Grey bands show NBER recessions. Y-axis is log scale."
            )
            st.plotly_chart(_growth_figure(selected, w_df, returns), width="stretch")
            st.warning(
                "**In-sample only.** The weights are estimated on the same data used to "
                "measure performance, so optimized portfolios look better than they will "
                "out-of-sample."
            )

    # Efficient Frontier ------------------------------------------------------
    if tab_is_open(tab_ef, fallback=active == "Efficient Frontier"):
        with tab_ef:
            st.subheader("Efficient frontier")
            st.markdown(
                "The curve shows the unconstrained analytical frontier. "
                "Portfolio markers reflect the selected optimization mode. "
                "Grey dots are individual stocks."
            )
            if mode == "long_only":
                st.info(
                    "Long-only portfolios may sit inside (not on) the unconstrained frontier "
                    "because the constraint prevents some short positions."
                )
            st.plotly_chart(
                _frontier_figure(selected, w_df, returns, sc),
                width="stretch",
            )

    # Methodology -------------------------------------------------------------
    if tab_is_open(tab_me, fallback=active == "Methodology"):
        with tab_me:
            st.subheader("Methodology")
            st.markdown(
                f"**Data:** 50 U.S. equities, daily adjusted close prices, 2020–2023. "
                f"Source: course data bundle (cached on first load).  \n"
                f"**Optimization mode:** {CONSTRAINT_LABELS[mode]}.  \n"
                f"**Risk-free rate:** 0 (daily).  \n"
                f"**Rebalancing:** Daily (frictionless)."
            )
            st.markdown("**Equal-weight**")
            st.latex(r"w_i = \frac{1}{N}")
            st.markdown("**Minimum-variance portfolio**")
            if mode == "unconstrained":
                st.latex(r"w_{mv} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}")
            else:
                st.latex(
                    r"w_{mv} = \arg\min_{w}\; w^\top\Sigma w \quad"
                    r"\text{s.t.}\quad \mathbf{1}^\top w = 1,\; w_i \ge 0"
                )
            st.markdown("**Mean-variance (maximum-Sharpe) portfolio**")
            if mode == "unconstrained":
                st.latex(
                    r"w_{tan} = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu - r_f\mathbf{1})}"
                )
            else:
                st.latex(
                    r"w_{tan} = \arg\max_{w}\;\frac{w^\top(\mu-r_f\mathbf{1})}{\sqrt{w^\top\Sigma w}}"
                    r"\quad\text{s.t.}\quad \mathbf{1}^\top w = 1,\; w_i \ge 0"
                )
            st.info(
                "All results are **in-sample only**. The same 2020–2023 data picks the "
                "weights and measures performance. In-sample returns overstate what you "
                "would earn on new data."
            )

    sync_query_params(view=active)


if __name__ == "__main__":
    main()
