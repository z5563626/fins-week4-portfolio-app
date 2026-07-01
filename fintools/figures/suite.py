"""Dataframe-to-figure-suite orchestration for report-ready plots."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .export import (
    FigureContext,
    WordFigureEntry,
    export_figure_bundle,
    export_word_figure,
    insert_figures_docx,
)
from .plots import (
    _display_label,
    _sparse_positions,
    bubble_matrix_plot,
    correlation_heatmap,
    cumulative_returns_plot,
    distribution_comparison_plot,
    distribution_plot,
    indexed_time_series_plot,
    lollipop_plot,
    scatter_plot,
    slope_chart,
    stacked_area_plot,
    time_series_plot,
)
from .theme import FT_COLORS, figure_style
from .validation import (
    FigureIssue,
    infer_return_scale,
    validate_category_label_count,
    validate_display_labels,
    validate_docx_images_fit_page,
    validate_image_not_blank,
    validate_markers_within_axes,
    validate_no_text_overlap,
    validate_no_tick_label_overlap,
    validate_series_identification,
    validate_titles_within_canvas,
)

_SUITE_DATE_COLUMN = "__figure_suite_date"
_ELECTRONIC_SEGMENTS = ("All", "IG", "HY")
_ELECTRONIC_DISPLAY_SEGMENTS = {
    "All": "All IG+HY",
    "IG": "Investment grade",
    "HY": "High yield",
}
_ELECTRONIC_VENUES = (
    "MarketAxess",
    "Tradeweb",
    "Trumid",
    "Other electronic",
    "Voice/high-touch",
)
_ELECTRONIC_VENUE_COLORS = {
    "Total electronic": "#262626",
    "Electronic": "#4EA3C8",
    "MarketAxess": "#1F77B4",
    "Tradeweb": "#3A9A44",
    "Trumid": "#7B5EA7",
    "Other electronic": "#D39B2D",
    "Voice/high-touch": "#9A3A4D",
    "Voice": "#9A3A4D",
}


@dataclass(frozen=True)
class DataFrameProfile:
    """Lightweight profile used to plan a figure suite from a dataframe."""

    row_count: int
    column_count: int
    date_column: str | None
    date_kind: str | None
    sample: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    percent_columns: tuple[str, ...]
    return_columns: tuple[str, ...]


@dataclass(frozen=True)
class FigurePlanItem:
    """One planned chart in a generated figure suite."""

    kind: str
    title: str
    reason: str
    params: Mapping[str, object]
    spec: str = "full_width"


@dataclass(frozen=True)
class GeneratedFigure:
    """One successfully generated figure and its output paths."""

    plan_item: FigurePlanItem
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class FigureSuiteResult:
    """Result returned by :func:`create_figure_suite`."""

    output_dir: Path
    profile: DataFrameProfile
    plan: tuple[FigurePlanItem, ...]
    generated_figures: tuple[GeneratedFigure, ...]
    docx_path: Path | None
    skipped: tuple[str, ...]
    issues: tuple[FigureIssue, ...]

    @property
    def generated_paths(self) -> tuple[Path, ...]:
        """Return every generated path as a flat tuple."""

        paths: list[Path] = []
        for figure in self.generated_figures:
            paths.extend(figure.paths.values())
        if self.docx_path is not None:
            paths.append(self.docx_path)
        return tuple(paths)


def profile_dataframe(data: pd.DataFrame, *, date: str | None = None) -> DataFrameProfile:
    """Profile a dataframe so an agent or script can plan a useful figure suite."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("dataframe is empty")

    frame = data.copy()
    date_column, date_kind, date_values = _detect_date(frame, date=date)
    datetime_columns = tuple(
        str(column)
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
    )

    numeric_columns = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        column_name = str(column)
        if column_name == date_column and date_kind == "year_column":
            continue
        if frame[column].dropna().nunique() <= 1:
            continue
        numeric_columns.append(column_name)

    categorical_columns = []
    for column in frame.columns:
        column_name = str(column)
        if column_name == date_column:
            continue
        if column_name in numeric_columns:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            unique_count = int(series.astype(str).nunique())
            if 1 < unique_count <= max(24, min(40, len(series) // 2 + 1)):
                categorical_columns.append(column_name)

    percent_columns = tuple(
        column for column in numeric_columns if _looks_like_percent_column(column, frame[column])
    )
    return_columns = tuple(
        column for column in numeric_columns if _looks_like_return_column(column)
    )

    return DataFrameProfile(
        row_count=len(frame),
        column_count=len(frame.columns),
        date_column=date_column,
        date_kind=date_kind,
        sample=_sample_label(date_values, len(frame)),
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        datetime_columns=datetime_columns,
        percent_columns=percent_columns,
        return_columns=return_columns,
    )


def plan_figure_suite(
    data: pd.DataFrame,
    *,
    date: str | None = None,
    title_prefix: str = "",
    max_figures: int = 8,
    narrative: bool = False,
) -> tuple[FigurePlanItem, ...]:
    """Plan a compact suite of useful figures for a dataframe."""

    profile = profile_dataframe(data, date=date)
    frame, plot_date = _prepare_frame(data, profile)
    title_prefix = title_prefix.strip()
    prefix = f"{title_prefix}: " if title_prefix else ""
    numeric = list(profile.numeric_columns)
    categorical = list(profile.categorical_columns)
    items: list[FigurePlanItem] = []

    if narrative:
        narrative_items = _narrative_plan(frame, profile, plot_date, prefix, max_figures)
        if narrative_items:
            return tuple(narrative_items[:max_figures])

    def add(item: FigurePlanItem) -> None:
        if len(items) >= max_figures:
            return
        if item.kind in {existing.kind for existing in items}:
            return
        items.append(item)

    long_plan = _long_panel_plan(frame, profile, plot_date)
    if long_plan and plot_date is not None:
        line_category = long_plan["line_category"]
        value = long_plan["value"]
        filter_column = long_plan.get("filter_column")
        filter_value = long_plan.get("filter_value")
        filter_phrase = (
            f" for {filter_value}" if filter_column is not None and filter_value is not None else ""
        )
        add(
            FigurePlanItem(
                kind="long_time_series",
                title=(
                    f"{prefix}{_display_label(value)} Over Time By "
                    f"{_display_label(line_category)}"
                ),
                reason=(
                    "Multi-series time-series chart from the long panel structure"
                    f"{filter_phrase}."
                ),
                params=long_plan,
            )
        )
        part_plan = _long_part_to_whole_plan(frame, profile, plot_date, long_plan)
        if part_plan is not None:
            add(
                FigurePlanItem(
                    kind="long_part_to_whole",
                    title=f"{prefix}{_display_label(part_plan['segment'])} Composition Over Time",
                    reason=(
                        "Part-to-whole stacked area chart using components that add up"
                        " within each time period."
                    ),
                    params=part_plan,
                    spec="two_panel",
                )
            )
        if _has_two_time_periods(frame, plot_date, line_category, value):
            add(
                FigurePlanItem(
                    kind="long_slope",
                    title=(
                        f"{prefix}{_display_label(value)} Change By "
                        f"{_display_label(line_category)}"
                    ),
                    reason="Slope chart comparing the first and last available time periods.",
                    params=long_plan,
                    spec="portrait_tall",
                )
            )
        add(
            FigurePlanItem(
                kind="long_latest_lollipop",
                title=f"{prefix}Latest {_display_label(value)} By {_display_label(line_category)}",
                reason="Ranked latest-period comparison across categories.",
                params=long_plan,
            )
        )

    if plot_date is not None and len(numeric) >= 2:
        columns = _best_numeric_columns(frame, numeric, count=4)
        add(
            FigurePlanItem(
                kind="wide_time_series",
                title=f"{prefix}Selected Series Over Time",
                reason="Multi-line time-series view of the most complete numeric series.",
                params={"columns": columns},
            )
        )

    if plot_date is not None and len(numeric) >= 2:
        positive = [
            column
            for column in _best_numeric_columns(frame, numeric, count=6)
            if _mostly_positive(frame[column])
        ]
        if len(positive) >= 2:
            add(
                FigurePlanItem(
                    kind="indexed_time_series",
                    title=f"{prefix}Indexed Series Comparison",
                    reason=(
                        "Indexed time-series chart so series with different units can"
                        " be compared on a common base."
                    ),
                    params={"columns": positive[:4]},
                )
            )

    if plot_date is not None and profile.return_columns:
        returns = list(profile.return_columns[:4])
        add(
            FigurePlanItem(
                kind="cumulative_returns",
                title=f"{prefix}Growth Of One Dollar",
                reason=(
                    "Cumulative return chart on a log scale, using dollar values on"
                    " the y-axis."
                ),
                params={"columns": returns},
            )
        )

    if categorical and numeric:
        category = _best_category(frame, categorical)
        value = _best_numeric_columns(frame, numeric, count=1)[0]
        add(
            FigurePlanItem(
                kind="mean_bar",
                title=f"{prefix}Average {_display_label(value)} By {_display_label(category)}",
                reason="Bar chart of category-level means for the clearest grouping variable.",
                params={"category": category, "value": value},
            )
        )
        add(
            FigurePlanItem(
                kind="distribution_by_category",
                title=f"{prefix}{_display_label(value)} Distribution By {_display_label(category)}",
                reason="Distribution comparison across categories.",
                params={"category": category, "value": value},
            )
        )

    if len(numeric) >= 2:
        x, y = _strongest_numeric_pair(frame, numeric)
        add(
            FigurePlanItem(
                kind="scatter_fit",
                title=f"{prefix}{_display_label(y)} Versus {_display_label(x)}",
                reason="Scatter plot with a fitted line and compact fit statistics.",
                params={"x": x, "y": y},
            )
        )

    if len(numeric) >= 3:
        add(
            FigurePlanItem(
                kind="correlation_heatmap",
                title=f"{prefix}Correlation Heatmap",
                reason="Correlation heatmap for the most complete numeric variables.",
                params={"columns": _best_numeric_columns(frame, numeric, count=8)},
                spec="two_panel",
            )
        )

    if numeric:
        value = _best_numeric_columns(frame, numeric, count=1)[0]
        add(
            FigurePlanItem(
                kind="distribution",
                title=f"{prefix}{_display_label(value)} Distribution",
                reason="Histogram and density plot for the most complete numeric series.",
                params={"value": value},
            )
        )

    return tuple(items)


def create_figure_suite(
    data: pd.DataFrame,
    output: str | Path = "results/figures",
    *,
    date: str | None = None,
    style: str = "ft",
    ft_background: bool = False,
    docx: bool = True,
    source: str = "",
    title_prefix: str = "",
    max_figures: int = 8,
    narrative: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
    strict: bool = False,
) -> FigureSuiteResult:
    """Create a validated FT/FINS-style figure suite from a dataframe.

    The generator intentionally plans more conservatively than an interactive
    analyst. If a planned chart renders with overlapping labels, clipped markers,
    blank pixels, or unreadable Word layout, it is skipped and recorded in the
    returned result.
    """

    if style not in {"fins", "ft"}:
        raise ValueError("style must be one of: fins, ft")
    if max_figures < 1:
        raise ValueError("max_figures must be at least 1")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_suite_outputs(output_dir)
    profile = profile_dataframe(data, date=date)
    frame, plot_date = _prepare_frame(data, profile)
    plan = plan_figure_suite(
        data,
        date=date,
        title_prefix=title_prefix,
        max_figures=max_figures,
        narrative=narrative,
    )
    generated: list[GeneratedFigure] = []
    docx_entries: list[WordFigureEntry] = []
    skipped: list[str] = []
    issues: list[FigureIssue] = []

    for number, item in enumerate(plan, start=1):
        stem = f"figure_suite_{number:02d}_{_slugify(item.kind)}"
        try:
            fig, context = _build_planned_figure(
                frame,
                profile,
                plot_date,
                item,
                style=style,
                ft_background=ft_background,
                source=source,
            )
            rendered_issues = _rendered_issues(fig)
            if rendered_issues:
                issues.extend(rendered_issues)
                skipped.append(_skip_message(item, rendered_issues))
                plt.close(fig)
                if strict:
                    break
                continue

            paths: dict[str, Path] = {}
            if "png" in formats:
                paths.update(
                    export_word_figure(
                        fig,
                        output_dir,
                        stem,
                        context=context,
                        spec=item.spec,
                    )
                )
            remaining_formats = tuple(fmt for fmt in formats if fmt != "png")
            if remaining_formats:
                paths.update(
                    export_figure_bundle(
                        fig,
                        output_dir,
                        stem,
                        context=None if "caption" in paths else context,
                        formats=remaining_formats,
                    )
                )

            image_issues = []
            if "png" in paths:
                image_issues.extend(validate_image_not_blank(paths["png"]))
            if image_issues:
                _delete_paths(paths.values())
                issues.extend(image_issues)
                skipped.append(_skip_message(item, image_issues))
                plt.close(fig)
                if strict:
                    break
                continue

            generated_figure = GeneratedFigure(item, paths)
            generated.append(generated_figure)
            if docx and "png" in paths:
                docx_entries.append(WordFigureEntry(paths["png"], context=context, spec=item.spec))
            plt.close(fig)
        except Exception as exc:
            skipped.append(f"{item.title}: {exc}")
            if strict:
                raise

    docx_path: Path | None = None
    if docx and docx_entries:
        docx_path = insert_figures_docx(
            docx_entries,
            output_dir / "figure_suite.docx",
            title="FT-Style Figure Suite" if style == "ft" else "Figure Suite",
        )
        docx_issues = validate_docx_images_fit_page(docx_path)
        if docx_issues:
            issues.extend(docx_issues)
            if strict:
                raise ValueError(
                    "; ".join(issue.message for issue in docx_issues)
                )

    if not generated:
        issues.append(
            FigureIssue(
                "no_figures_generated",
                "No planned figures passed validation for this dataframe.",
            )
        )

    return FigureSuiteResult(
        output_dir=output_dir,
        profile=profile,
        plan=plan,
        generated_figures=tuple(generated),
        docx_path=docx_path,
        skipped=tuple(skipped),
        issues=tuple(issues),
    )


def _narrative_plan(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
    prefix: str,
    max_figures: int,
) -> list[FigurePlanItem]:
    if _is_electronic_trading_share_data(frame, profile):
        value = _electronic_value_column(profile)
        segment = _electronic_segment_column(frame, profile)
        venue = _electronic_venue_column(frame, profile)
        params = {"value": value, "segment": segment, "venue": venue, "plot_date": plot_date}
        return [
            FigurePlanItem(
                kind="electronic_voice_small_multiples",
                title=f"{prefix}Electronic Versus Voice Trading By Credit Segment",
                reason=(
                    "Small multiples compare electronic and voice/high-touch trading"
                    " shares for the all-market, investment-grade, and high-yield"
                    " segments."
                ),
                params=params,
                spec="two_panel",
            ),
            FigurePlanItem(
                kind="electronic_whole_market_stacked_share",
                title=f"{prefix}Whole-Market Trading Volume By Venue",
                reason=(
                    "Stacked bars decompose all-market trading into named electronic"
                    " venues, other electronic trading, and voice/high-touch trading."
                ),
                params=params,
                spec="two_panel",
            ),
            FigurePlanItem(
                kind="electronic_venue_mix",
                title=f"{prefix}Electronic Venue Mix Within Electronic Trading",
                reason=(
                    "Line chart shows how MarketAxess, Tradeweb, Trumid, and other"
                    " electronic venues split electronic trading volume."
                ),
                params=params,
            ),
            FigurePlanItem(
                kind="electronic_share_slope",
                title=f"{prefix}Electronic Share Change By Credit Segment",
                reason=(
                    "Slope chart compares the first and latest electronic share for"
                    " each credit segment."
                ),
                params=params,
                spec="portrait_tall",
            ),
            FigurePlanItem(
                kind="electronic_latest_venue_lollipop",
                title=f"{prefix}Latest Whole-Market Venue Share",
                reason=(
                    "Ranked latest-period view highlights the balance between voice,"
                    " MarketAxess, Tradeweb, Trumid, and other electronic trading."
                ),
                params=params,
            ),
            FigurePlanItem(
                kind="electronic_unconditional_bubble_matrix",
                title=f"{prefix}Average Venue Shares",
                reason=(
                    "Bubble matrix summarizes full-year unconditional average venue"
                    " shares by segment without plotting a time path."
                ),
                params=params,
            ),
        ][:max_figures]
    portfolio_params = _portfolio_returns_params(frame, profile, plot_date)
    if portfolio_params is not None:
        return [
            FigurePlanItem(
                kind="portfolio_growth_of_one_dollar",
                title=f"{prefix}Growth Of One Dollar Across Return Series",
                reason=(
                    "Cumulative growth of one dollar from each return series, shown"
                    " on a log scale so long-run compounding differences are visible."
                ),
                params=portfolio_params,
            ),
            FigurePlanItem(
                kind="portfolio_drawdowns",
                title=f"{prefix}Drawdowns Across Return Series",
                reason=(
                    "Drawdowns show peak-to-trough losses implied by each return"
                    " series."
                ),
                params=portfolio_params,
            ),
            FigurePlanItem(
                kind="portfolio_risk_return",
                title=f"{prefix}Risk And Return By Series",
                reason=(
                    "Scatter plot compares annualized average return and annualized"
                    " volatility for each return series."
                ),
                params=portfolio_params,
            ),
            FigurePlanItem(
                kind="portfolio_return_correlations",
                title=f"{prefix}Return Correlations",
                reason="Correlation heatmap summarizes co-movement across return series.",
                params=portfolio_params,
                spec="two_panel",
            ),
            FigurePlanItem(
                kind="portfolio_return_distributions",
                title=f"{prefix}Return Distributions",
                reason="Distribution comparison shows dispersion and tail behavior by series.",
                params=portfolio_params,
            ),
        ][:max_figures]
    return []


def _is_electronic_trading_share_data(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
) -> bool:
    if profile.date_kind != "year_column":
        return False
    try:
        segment = _electronic_segment_column(frame, profile)
        venue = _electronic_venue_column(frame, profile)
        value = _electronic_value_column(profile)
    except ValueError:
        return False
    venues = set(frame[venue].dropna().astype(str).map(_normalise_venue))
    segments = set(frame[segment].dropna().astype(str))
    return bool(
        {"Total electronic", "Voice/high-touch"}.issubset(venues)
        and {"All", "IG", "HY"}.issubset(segments)
        and value in frame.columns
    )


def _portfolio_returns_params(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
) -> dict[str, object] | None:
    if plot_date is None:
        return None

    wide_columns = [
        column
        for column in profile.return_columns
        if column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 12
    ]
    if len(wide_columns) >= 2:
        selected = _select_return_columns(frame, wide_columns, count=6)
        return {"layout": "wide", "columns": selected}

    value_candidates = [
        column
        for column in profile.return_columns
        if column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 12
    ]
    if not value_candidates:
        value_candidates = [
            column
            for column in profile.numeric_columns
            if _looks_like_return_column(column)
            and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 12
        ]
    if len(value_candidates) != 1:
        return None

    value = value_candidates[0]
    categories = [
        column
        for column in profile.categorical_columns
        if _is_time_panel_category(frame, plot_date, column)
        and 2 <= frame[column].dropna().astype(str).nunique() <= 12
    ]
    if not categories:
        return None
    category = min(categories, key=lambda column: frame[column].dropna().astype(str).nunique())
    return {"layout": "long", "value": value, "category": category}


def _select_return_columns(frame: pd.DataFrame, columns: Sequence[str], *, count: int) -> list[str]:
    ranked = sorted(
        columns,
        key=lambda column: (
            int(pd.to_numeric(frame[column], errors="coerce").notna().sum()),
            float(pd.to_numeric(frame[column], errors="coerce").std(skipna=True) or 0.0),
        ),
        reverse=True,
    )
    return ranked[:count]


def _portfolio_returns_wide(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
    params: Mapping[str, object],
) -> tuple[pd.DataFrame, bool]:
    if plot_date is None:
        raise ValueError("portfolio return narrative requires a date column")
    if params.get("layout") == "wide":
        columns = list(params["columns"])
        returns = frame[[plot_date, *columns]].copy()
        returns = returns.dropna(subset=[plot_date]).set_index(plot_date)
        returns = returns[columns].apply(pd.to_numeric, errors="coerce")
    elif params.get("layout") == "long":
        value = str(params["value"])
        category = str(params["category"])
        returns = frame[[plot_date, category, value]].dropna(subset=[plot_date, category]).copy()
        returns[value] = pd.to_numeric(returns[value], errors="coerce")
        returns = returns.pivot_table(
            index=plot_date,
            columns=category,
            values=value,
            aggfunc="mean",
        )
    else:
        raise ValueError("unknown portfolio return layout")

    returns = returns.sort_index().dropna(how="all")
    if returns.shape[1] < 2:
        raise ValueError("portfolio return narrative requires at least two return series")
    returns.columns = [_display_label(column) for column in returns.columns]
    scale = _infer_return_scale(returns)
    returns_are_percent = scale != "decimal"
    return returns, returns_are_percent


def _portfolio_drawdown_figure(
    returns: pd.DataFrame,
    *,
    returns_are_percent: bool,
    title: str,
    style: str,
    ft_background: bool,
) -> plt.Figure:
    values = returns.astype(float) / 100.0 if returns_are_percent else returns.astype(float)
    wealth = (1.0 + values).cumprod()
    drawdowns = wealth / wealth.cummax() - 1.0
    fig, ax = time_series_plot(
        drawdowns,
        list(drawdowns.columns),
        title=title,
        ylabel="Drawdown",
        shade_recessions=True,
        profile="word_a4",
        style=style,
        ft_background=ft_background,
        line_width=1.2 if style == "ft" else None,
        line_alpha=0.82 if style == "ft" else None,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    return fig


def _portfolio_risk_return_figure(
    returns: pd.DataFrame,
    *,
    returns_are_percent: bool,
    title: str,
    style: str,
) -> plt.Figure:
    values = returns.astype(float) / 100.0 if returns_are_percent else returns.astype(float)
    periods_per_year = _infer_periods_per_year(values.index)
    summary = pd.DataFrame(
        {
            "Series": values.columns.astype(str),
            "Annualized average return": values.mean() * periods_per_year * 100.0,
            "Annualized volatility": values.std(ddof=1) * np.sqrt(periods_per_year) * 100.0,
        }
    ).dropna()
    with figure_style("word_a4", style=style):
        fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="none")
        palette = _portfolio_palette(style, len(summary))
        for index, row in summary.reset_index(drop=True).iterrows():
            color = palette[index % len(palette)]
            ax.scatter(
                row["Annualized volatility"],
                row["Annualized average return"],
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            text = ax.annotate(
                str(row["Series"]),
                (row["Annualized volatility"], row["Annualized average return"]),
                textcoords="offset points",
                xytext=(6, 5),
                ha="left",
                fontsize=8.5,
                color=color,
            )
            text.set_gid("figure_label")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.grid(True, axis="y", color=FT_COLORS["grid"], linewidth=0.8, alpha=0.78)
        ax.grid(False, axis="x")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Annualized volatility (%)")
        ax.set_ylabel("Annualized average return (%)")
        _pad_numeric_limits_for_text(ax)
        fig.subplots_adjust(bottom=0.15, left=0.14, right=0.94, top=0.88)
        return fig


def _portfolio_palette(style: str, count: int) -> list[str]:
    if style == "ft":
        base = [
            "#A51C50",
            "#2E75B6",
            "#0F7C6E",
            "#D39B2D",
            "#7B5EA7",
            "#6B7280",
        ]
    else:
        base = ["#1B365D", "#9F1D35", "#2E7D32", "#6B7280", "#7C3AED", "#C2410C"]
    return [base[index % len(base)] for index in range(max(count, 1))]


def _infer_periods_per_year(index: pd.Index) -> int:
    dates = pd.Series(pd.to_datetime(index, errors="coerce")).dropna().sort_values()
    if len(dates) < 3:
        return 12
    median_days = float(dates.diff().dropna().dt.days.median())
    if median_days <= 3:
        return 252
    if median_days <= 10:
        return 52
    if median_days <= 45:
        return 12
    if median_days <= 110:
        return 4
    return 1


def _pad_numeric_limits_for_text(ax: plt.Axes) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    ax.set_xlim(x_min - 0.04 * x_span, x_max + 0.14 * x_span)
    ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.12 * y_span)


def _electronic_segment_column(frame: pd.DataFrame, profile: DataFrameProfile) -> str:
    for column in profile.categorical_columns:
        values = set(frame[column].dropna().astype(str))
        if {"All", "IG", "HY"}.issubset(values):
            return column
    raise ValueError("electronic trading suite requires All, IG, and HY segment values")


def _electronic_venue_column(frame: pd.DataFrame, profile: DataFrameProfile) -> str:
    for column in profile.categorical_columns:
        values = set(frame[column].dropna().astype(str).map(_normalise_venue))
        if {"Total electronic", "Voice/high-touch"}.issubset(values):
            return column
    raise ValueError("electronic trading suite requires venue values")


def _electronic_value_column(profile: DataFrameProfile) -> str:
    for column in profile.percent_columns:
        if re.search(r"(share|percent|percentage|pct)", column, re.I):
            return column
    if profile.percent_columns:
        return profile.percent_columns[0]
    raise ValueError("electronic trading suite requires a percent/share value column")


def _normalise_venue(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"voice", "voice/high touch", "voice / high-touch"}:
        return "Voice/high-touch"
    return text


def _detect_date(
    frame: pd.DataFrame,
    *,
    date: str | None,
) -> tuple[str | None, str | None, pd.Series | pd.DatetimeIndex | None]:
    if date is not None:
        if date not in frame.columns:
            raise ValueError(f"date column not found: {date}")
        if _looks_like_year(frame[date]):
            return date, "year_column", _year_to_datetime(frame[date])
        parsed = pd.to_datetime(frame[date], errors="coerce")
        if parsed.notna().sum() == 0:
            raise ValueError(f"date column could not be parsed: {date}")
        return date, "date_column", parsed

    if isinstance(frame.index, pd.DatetimeIndex):
        return None, "index", frame.index

    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column), "date_column", pd.to_datetime(frame[column], errors="coerce")

    for column in frame.columns:
        column_name = str(column)
        if _looks_like_year(frame[column]) and column_name.lower() in {"year", "yr"}:
            return column_name, "year_column", _year_to_datetime(frame[column])

    date_name_re = re.compile(r"(date|month|period|quarter|time)", re.IGNORECASE)
    for column in frame.columns:
        column_name = str(column)
        if not date_name_re.search(column_name):
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return column_name, "date_column", parsed

    return None, None, None


def _prepare_frame(
    data: pd.DataFrame,
    profile: DataFrameProfile,
) -> tuple[pd.DataFrame, str | None]:
    frame = data.copy()
    if profile.date_kind == "index":
        frame.index = pd.to_datetime(frame.index)
        frame[_SUITE_DATE_COLUMN] = frame.index
        return frame, _SUITE_DATE_COLUMN
    if profile.date_kind == "date_column" and profile.date_column is not None:
        frame[profile.date_column] = pd.to_datetime(frame[profile.date_column], errors="coerce")
        return frame, profile.date_column
    if profile.date_kind == "year_column" and profile.date_column is not None:
        frame[_SUITE_DATE_COLUMN] = _year_to_datetime(frame[profile.date_column])
        return frame, _SUITE_DATE_COLUMN
    return frame, None


def _plot_date_argument(profile: DataFrameProfile, plot_date: str | None) -> str | None:
    if plot_date is None:
        return None
    if profile.date_kind == "index":
        return None
    return plot_date


_SHORT_PLOT_TITLES = {
    "electronic_voice_small_multiples": "Electronic Vs Voice",
    "electronic_whole_market_stacked_share": "Venue Composition",
    "electronic_venue_mix": "Electronic Venue Mix",
    "electronic_share_slope": "Electronic Share Change",
    "electronic_latest_venue_lollipop": "Latest Venue Share",
    "electronic_unconditional_bubble_matrix": "Average Venue Shares",
}


def _plot_title(item: FigurePlanItem) -> str:
    """Return the short title rendered inside the figure canvas."""

    if item.kind in _SHORT_PLOT_TITLES:
        return _SHORT_PLOT_TITLES[item.kind]
    if ":" in item.title:
        return item.title.split(":", 1)[1].strip()
    return item.title


def _build_planned_figure(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
    item: FigurePlanItem,
    *,
    style: str,
    ft_background: bool,
    source: str,
) -> tuple[plt.Figure, FigureContext]:
    plot_kwargs = {
        "profile": "word_a4",
        "style": style,
        "ft_background": ft_background,
    }
    date_arg = _plot_date_argument(profile, plot_date)
    sample = _item_sample(frame, profile, plot_date, item)
    units = _item_units(profile, item)
    context = FigureContext(
        title=item.title,
        note=item.reason,
        source=source,
        sample=sample,
        units=units,
    )
    params = dict(item.params)
    plot_title = _plot_title(item)

    if item.kind == "electronic_voice_small_multiples":
        fig = _electronic_voice_small_multiples(frame, params, plot_title, style=style)
        return fig, context

    if item.kind == "electronic_whole_market_stacked_share":
        fig = _electronic_whole_market_stacked_share(frame, params, plot_title, style=style)
        return fig, context

    if item.kind == "electronic_venue_mix":
        fig = _electronic_venue_mix(frame, params, plot_title, style=style)
        return fig, context

    if item.kind == "electronic_share_slope":
        fig = _electronic_share_slope(frame, params, plot_title, style=style)
        return fig, context

    if item.kind == "electronic_latest_venue_lollipop":
        fig = _electronic_latest_venue_lollipop(frame, params, plot_title, style=style)
        return fig, context

    if item.kind == "electronic_unconditional_bubble_matrix":
        fig = _electronic_unconditional_bubble_matrix(frame, params, plot_title, style=style)
        return fig, context

    if item.kind == "portfolio_growth_of_one_dollar":
        returns, returns_are_percent = _portfolio_returns_wide(frame, profile, plot_date, params)
        fig, _ = cumulative_returns_plot(
            returns,
            list(returns.columns),
            returns_are_percent=returns_are_percent,
            wealth_index=True,
            log_scale=True,
            title=plot_title,
            profile="word_a4",
            style=style,
            ft_background=ft_background,
            line_width=1.25 if style == "ft" else None,
            line_alpha=0.82 if style == "ft" else None,
        )
        return fig, context

    if item.kind == "portfolio_drawdowns":
        returns, returns_are_percent = _portfolio_returns_wide(frame, profile, plot_date, params)
        fig = _portfolio_drawdown_figure(
            returns,
            returns_are_percent=returns_are_percent,
            title=plot_title,
            style=style,
            ft_background=ft_background,
        )
        return fig, context

    if item.kind == "portfolio_risk_return":
        returns, returns_are_percent = _portfolio_returns_wide(frame, profile, plot_date, params)
        fig = _portfolio_risk_return_figure(
            returns,
            returns_are_percent=returns_are_percent,
            title=plot_title,
            style=style,
        )
        return fig, context

    if item.kind == "portfolio_return_correlations":
        returns, _returns_are_percent = _portfolio_returns_wide(frame, profile, plot_date, params)
        fig, _ = correlation_heatmap(
            returns,
            list(returns.columns),
            title=plot_title,
            profile="word_a4",
            style=style,
            ft_background=ft_background,
        )
        return fig, context

    if item.kind == "portfolio_return_distributions":
        returns, _returns_are_percent = _portfolio_returns_wide(frame, profile, plot_date, params)
        long = (
            returns.reset_index(names="Date")
            .melt(id_vars="Date", var_name="Series", value_name="Return")
            .dropna()
        )
        order = list(returns.columns)
        fig, _ = distribution_comparison_plot(
            long,
            "Return",
            "Series",
            title=plot_title,
            ylabel="Return",
            order=order,
            profile="word_a4",
            style=style,
            ft_background=ft_background,
        )
        return fig, context

    if item.kind == "wide_time_series":
        columns = list(params["columns"])
        plot_frame, plot_columns = _renamed_numeric_frame(frame, columns)
        if date_arg is not None and plot_date is not None:
            plot_frame[plot_date] = frame[plot_date]
        fig, _ = time_series_plot(
            plot_frame,
            plot_columns,
            date=date_arg,
            title=plot_title,
            ylabel=units,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "indexed_time_series":
        columns = list(params["columns"])
        plot_frame, plot_columns = _renamed_numeric_frame(frame, columns)
        if date_arg is not None and plot_date is not None:
            plot_frame[plot_date] = frame[plot_date]
        fig, _ = indexed_time_series_plot(
            plot_frame,
            plot_columns,
            date=date_arg,
            title=plot_title,
            ylabel="Index (first observation = 100)",
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "cumulative_returns":
        columns = list(params["columns"])
        plot_frame, plot_columns = _renamed_numeric_frame(frame, columns)
        if date_arg is not None and plot_date is not None:
            plot_frame[plot_date] = frame[plot_date]
        scale = _infer_return_scale(frame[columns])
        fig, _ = cumulative_returns_plot(
            plot_frame,
            plot_columns,
            date=date_arg,
            returns_are_percent=scale != "decimal",
            wealth_index=True,
            log_scale=True,
            title=plot_title,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "long_time_series":
        long_frame = _filtered_long_frame(frame, params)
        value = str(params["value"])
        line_category = str(params["line_category"])
        pivot = _long_time_pivot(long_frame, plot_date, line_category, value)
        fig, _ = time_series_plot(
            pivot,
            list(pivot.columns),
            title=plot_title,
            ylabel=_display_label(value),
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "long_part_to_whole":
        long_frame = _filtered_long_frame(frame, params)
        value = str(params["value"])
        segment = str(params["segment"])
        if bool(params.get("exclude_total_segments", False)):
            long_frame = long_frame[
                ~long_frame[segment].astype(str).map(_is_total_like_value)
            ]
        wide = _long_time_pivot(long_frame, plot_date, segment, value).reset_index()
        x_column = str(wide.columns[0])
        fig, _ = stacked_area_plot(
            wide,
            x_column,
            list(wide.columns[1:]),
            title=plot_title,
            xlabel="Date",
            ylabel="Share",
            normalize=True,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "long_slope":
        slope_frame, start_column, end_column = _long_first_last_frame(
            _filtered_long_frame(frame, params),
            plot_date,
            str(params["line_category"]),
            str(params["value"]),
        )
        fig, _ = slope_chart(
            slope_frame,
            str(params["line_category"]),
            start_column,
            end_column,
            title=plot_title,
            ylabel=_display_label(str(params["value"])),
            start_label=start_column,
            end_label=end_column,
            limit=8,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "long_latest_lollipop":
        latest = _long_latest_frame(
            _filtered_long_frame(frame, params),
            plot_date,
            str(params["line_category"]),
            str(params["value"]),
        )
        fig, _ = lollipop_plot(
            latest,
            str(params["line_category"]),
            str(params["value"]),
            title=plot_title,
            xlabel=_display_label(str(params["value"])),
            ylabel=_display_label(str(params["line_category"])),
            limit=12,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "mean_bar":
        category = str(params["category"])
        value = str(params["value"])
        summary = (
            frame[[category, value]]
            .dropna()
            .groupby(category, as_index=False)[value]
            .mean()
            .sort_values(value, ascending=False)
            .head(12)
        )
        fig, _ = lollipop_plot(
            summary,
            category,
            value,
            title=plot_title,
            xlabel=_display_label(value),
            ylabel=_display_label(category),
            sort=True,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "distribution_by_category":
        category = str(params["category"])
        value = str(params["value"])
        filtered = _limit_category_levels(frame, category, value, limit=8)
        fig, _ = distribution_comparison_plot(
            filtered,
            value,
            category,
            title=plot_title,
            ylabel=_display_label(value),
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "scatter_fit":
        x = str(params["x"])
        y = str(params["y"])
        fig, _ = scatter_plot(
            frame,
            x,
            y,
            fit=True,
            annotate=True,
            label_outliers=0,
            title=plot_title,
            xlabel=_display_label(x),
            ylabel=_display_label(y),
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "correlation_heatmap":
        columns = list(params["columns"])
        plot_frame, plot_columns = _renamed_numeric_frame(frame, columns)
        fig, _ = correlation_heatmap(
            plot_frame,
            plot_columns,
            title=plot_title,
            **plot_kwargs,
        )
        return fig, context

    if item.kind == "distribution":
        value = str(params["value"])
        plot_frame, plot_columns = _renamed_numeric_frame(frame, [value])
        fig, _ = distribution_plot(
            plot_frame,
            plot_columns[0],
            title=plot_title,
            **plot_kwargs,
        )
        return fig, context

    raise ValueError(f"unknown figure plan kind: {item.kind}")


def _electronic_clean_frame(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.DataFrame:
    segment = str(params["segment"])
    venue = str(params["venue"])
    value = str(params["value"])
    plot_date = params.get("plot_date")
    columns = [segment, venue, value]
    if plot_date is not None:
        columns.append(str(plot_date))
    if "period" in frame.columns:
        columns.append("period")
    clean = frame[columns].dropna(subset=[segment, venue, value]).copy()
    clean["segment"] = clean[segment].astype(str)
    clean["venue"] = clean[venue].map(_normalise_venue)
    clean["share"] = pd.to_numeric(clean[value], errors="coerce")
    if plot_date is not None:
        clean["year"] = pd.to_datetime(clean[str(plot_date)], errors="coerce").dt.year
    else:
        year_column = next(
            column for column in frame.columns if str(column).lower() in {"year", "yr"}
        )
        clean["year"] = pd.to_numeric(clean[year_column], errors="coerce")
    clean = clean.dropna(subset=["year", "share"]).copy()
    clean["year"] = clean["year"].astype(int)
    if "period" not in clean:
        clean["period"] = clean["year"].astype(str)
    return clean[["year", "period", "segment", "venue", "share"]]


def _electronic_wide(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.DataFrame:
    clean = _electronic_clean_frame(frame, params)
    wide = clean.pivot_table(
        index=["year", "segment"],
        columns="venue",
        values="share",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    for venue in ("Total electronic", *_ELECTRONIC_VENUES):
        if venue not in wide:
            wide[venue] = 0.0
    wide["segment"] = pd.Categorical(
        wide["segment"],
        categories=list(_ELECTRONIC_SEGMENTS),
        ordered=True,
    )
    return wide.sort_values(["year", "segment"]).reset_index(drop=True)


def _electronic_full_year_wide(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.DataFrame:
    """Return electronic trading rows, dropping latest partial-year estimates."""

    clean = _electronic_clean_frame(frame, params)
    latest_year = int(clean["year"].max())
    latest_periods = " ".join(clean.loc[clean["year"] == latest_year, "period"].astype(str))
    if "q1" in latest_periods.lower() or "ytd" in latest_periods.lower():
        clean = clean[clean["year"] < latest_year].copy()
    wide = clean.pivot_table(
        index=["year", "segment"],
        columns="venue",
        values="share",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    for venue in ("Total electronic", *_ELECTRONIC_VENUES):
        if venue not in wide:
            wide[venue] = 0.0
    wide["segment"] = pd.Categorical(
        wide["segment"],
        categories=list(_ELECTRONIC_SEGMENTS),
        ordered=True,
    )
    return wide.sort_values(["year", "segment"]).reset_index(drop=True)


def _electronic_latest_label(frame: pd.DataFrame, params: Mapping[str, object]) -> str:
    clean = _electronic_clean_frame(frame, params)
    latest_year = int(clean["year"].max())
    latest_periods = " ".join(clean.loc[clean["year"] == latest_year, "period"].astype(str))
    if "q1" in latest_periods.lower() or "ytd" in latest_periods.lower():
        return f"{latest_year} YTD"
    return str(latest_year)


def _electronic_compact_latest_label(frame: pd.DataFrame, params: Mapping[str, object]) -> str:
    return _electronic_latest_label(frame, params).replace(" YTD", "\nYTD")


def _electronic_year_labels(years: Sequence[int], latest_label: str) -> list[str]:
    if not years:
        return []
    max_year = max(years)
    return [latest_label if int(year) == max_year else str(int(year)) for year in years]


def _electronic_sparse_years(years: Sequence[int], *, max_ticks: int = 6) -> list[int]:
    ordered = sorted({int(year) for year in years})
    positions = _sparse_positions(len(ordered), max_ticks)
    return [ordered[position] for position in positions]


def _format_electronic_share_axis(
    ax: plt.Axes,
    *,
    ylabel: str = "Share of trading volume",
    style: str,
    ylim: tuple[float, float] | None = (0.0, 100.0),
) -> None:
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=FT_COLORS["grid"], linewidth=0.8, alpha=0.82)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if style != "ft":
        ax.grid(True, axis="y", color="#D8DDE6", linewidth=0.6, alpha=0.65)


def _electronic_unconditional_bubble_matrix(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    title: str,
    *,
    style: str,
) -> plt.Figure:
    wide = _electronic_full_year_wide(frame, params)
    venues = list(_ELECTRONIC_VENUES)
    display_venues = {
        "MarketAxess": "MarketAxess",
        "Tradeweb": "Tradeweb",
        "Trumid": "Trumid",
        "Other electronic": "Other electronic",
        "Voice/high-touch": "Voice",
    }
    records = []
    for segment in _ELECTRONIC_SEGMENTS:
        segment_frame = wide[wide["segment"].astype(str) == segment]
        if segment_frame.empty:
            continue
        means = segment_frame[venues].mean()
        for venue in venues:
            records.append(
                {
                    "Segment": _ELECTRONIC_DISPLAY_SEGMENTS[segment],
                    "Venue": display_venues[venue],
                    "Average share": float(means[venue]),
                }
            )
    matrix = pd.DataFrame.from_records(records)
    colors = {
        display_venues[venue]: _ELECTRONIC_VENUE_COLORS[venue]
        for venue in venues
    }
    fig, _ = bubble_matrix_plot(
        matrix,
        "Venue",
        "Segment",
        "Average share",
        title=title,
        xlabel="Venue",
        ylabel="Segment",
        size_label="Average share",
        x_order=[display_venues[venue] for venue in venues],
        y_order=[_ELECTRONIC_DISPLAY_SEGMENTS[segment] for segment in _ELECTRONIC_SEGMENTS],
        colors=colors,
        min_size=55,
        max_size=1150,
        profile="word_a4",
        style=style,
    )
    return fig


def _electronic_voice_small_multiples(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    title: str,
    *,
    style: str,
) -> plt.Figure:
    wide = _electronic_wide(frame, params)
    compact_latest_label = _electronic_compact_latest_label(frame, params)
    years = sorted(wide["year"].unique())
    tick_years = _electronic_sparse_years(years, max_ticks=3)
    with figure_style("word_a4", style=style):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(9.0, 3.8),
            sharex=True,
            sharey=True,
            layout="none",
        )
        for ax, segment in zip(axes, _ELECTRONIC_SEGMENTS, strict=True):
            segment_frame = wide[wide["segment"].astype(str) == segment].sort_values("year")
            for column, label in [
                ("Total electronic", "Electronic"),
                ("Voice/high-touch", "Voice"),
            ]:
                ax.plot(
                    segment_frame["year"],
                    segment_frame[column],
                    label=label if segment == "All" else "_nolegend_",
                    color=_ELECTRONIC_VENUE_COLORS[column],
                    marker="o" if column == "Total electronic" else "D",
                    markersize=3.8,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    linewidth=1.95,
                    alpha=0.92,
                )
            ax.set_title(
                _ELECTRONIC_DISPLAY_SEGMENTS[segment],
                loc="left",
                fontsize=10.8,
                weight="bold",
            )
            _format_electronic_share_axis(ax, style=style)
            ax.set_xlim(min(years) - 0.35, max(years) + 0.35)
            ax.set_xticks(tick_years)
            ax.set_xticklabels(
                _electronic_year_labels(tick_years, compact_latest_label),
                fontsize=8.2,
            )
            if segment != "All":
                ax.set_ylabel("")
            if segment == "IG":
                ax.set_xlabel("Year")
            if segment == "All":
                ax.legend(loc="center right", frameon=False, fontsize=8.4)
        fig.suptitle(title, x=0.01, y=0.98, ha="left", weight="bold", fontsize=13.5)
        fig.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98, wspace=0.24)
        return fig


def _electronic_whole_market_stacked_share(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    title: str,
    *,
    style: str,
) -> plt.Figure:
    wide = _electronic_wide(frame, params)
    latest_label = _electronic_latest_label(frame, params)
    all_market = wide[wide["segment"].astype(str) == "All"].sort_values("year")
    venues = [venue for venue in _ELECTRONIC_VENUES if venue in all_market]
    years = all_market["year"].astype(int).tolist()
    positions = np.arange(len(all_market))
    tick_positions = _sparse_positions(len(years), 6)
    with figure_style("word_a4", style=style):
        fig, ax = plt.subplots(figsize=(7.8, 4.2), layout="none")
        bottom = np.zeros(len(all_market))
        for venue in venues:
            values = all_market[venue].astype(float).to_numpy()
            ax.bar(
                positions,
                values,
                bottom=bottom,
                label="Voice" if venue == "Voice/high-touch" else venue,
                color=_ELECTRONIC_VENUE_COLORS[venue],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
            )
            bottom += values
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            _electronic_year_labels([years[position] for position in tick_positions], latest_label)
        )
        ax.set_xlabel("Year")
        _format_electronic_share_axis(ax, style=style)
        ax.set_title(title, loc="left")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncols=min(len(venues), 5),
            frameon=False,
            fontsize=8.2,
            handlelength=1.2,
        )
        fig.subplots_adjust(bottom=0.25, left=0.10, right=0.98, top=0.88)
        return fig


def _electronic_venue_mix(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    title: str,
    *,
    style: str,
) -> plt.Figure:
    wide = _electronic_wide(frame, params)
    latest_label = _electronic_latest_label(frame, params)
    all_market = wide[wide["segment"].astype(str) == "All"].sort_values("year")
    venues = ["MarketAxess", "Tradeweb", "Trumid", "Other electronic"]
    years = all_market["year"].astype(int).tolist()
    tick_years = _electronic_sparse_years(years, max_ticks=6)
    total = all_market["Total electronic"].replace(0, np.nan)
    with figure_style("word_a4", style=style):
        fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="none")
        for venue in venues:
            if venue not in all_market:
                continue
            mix = all_market[venue].astype(float) / total * 100.0
            if mix.fillna(0.0).abs().max() < 0.05:
                continue
            ax.plot(
                all_market["year"],
                mix,
                label=venue,
                color=_ELECTRONIC_VENUE_COLORS[venue],
                marker="o",
                markersize=3.7,
                markeredgecolor="white",
                markeredgewidth=0.4,
                linewidth=1.8,
                alpha=0.90,
            )
        _format_electronic_share_axis(
            ax,
            ylabel="Share of electronic trading",
            style=style,
        )
        ax.set_xlim(min(years) - 0.35, max(years) + 0.35)
        ax.set_xticks(tick_years)
        ax.set_xticklabels(_electronic_year_labels(tick_years, latest_label))
        ax.set_xlabel("Year")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", frameon=False, fontsize=8.4)
        fig.subplots_adjust(bottom=0.15, left=0.12, right=0.97, top=0.88)
        return fig


def _electronic_share_slope(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    title: str,
    *,
    style: str,
) -> plt.Figure:
    wide = _electronic_wide(frame, params)
    latest_label = _electronic_latest_label(frame, params)
    endpoints = (
        wide.pivot(index="segment", columns="year", values="Total electronic")
        .reindex(list(_ELECTRONIC_SEGMENTS))
        .dropna(axis=1, how="all")
    )
    start_year = int(endpoints.columns.min())
    end_year = int(endpoints.columns.max())
    colors = {"All": "#262626", "IG": "#1F77B4", "HY": "#9A3A4D"}
    with figure_style("word_a4", style=style):
        fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="none")
        values = []
        for segment in _ELECTRONIC_SEGMENTS:
            start_value = float(endpoints.loc[segment, start_year])
            end_value = float(endpoints.loc[segment, end_year])
            values.extend([start_value, end_value])
            color = colors[segment]
            ax.plot([0, 1], [start_value, end_value], color=color, linewidth=2.3)
            ax.scatter([0, 1], [start_value, end_value], color=color, s=44, zorder=3)
            for x_pos, value, label, ha, offset in [
                (0, start_value, f"{segment} {start_value:.1f}%", "right", -0.03),
                (1, end_value, f"{end_value:.1f}%", "left", 0.03),
            ]:
                text = ax.text(
                    x_pos + offset,
                    value,
                    label,
                    ha=ha,
                    va="center",
                    fontsize=8.8,
                    color=color,
                )
                text.set_gid("figure_label")
        y_min, y_max = min(values), max(values)
        y_pad = max((y_max - y_min) * 0.18, 4.0)
        y_limits = (y_min - y_pad, min(100, y_max + y_pad))
        ax.set_xlim(-0.33, 1.33)
        ax.set_xticks([0, 1], [str(start_year), latest_label])
        _format_electronic_share_axis(
            ax,
            ylabel="Electronic share",
            style=style,
            ylim=y_limits,
        )
        ax.grid(False, axis="x")
        ax.set_xlabel("Period")
        ax.set_title(title, loc="left")
        fig.subplots_adjust(bottom=0.16, left=0.17, right=0.88, top=0.88)
        return fig


def _electronic_latest_venue_lollipop(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    title: str,
    *,
    style: str,
) -> plt.Figure:
    wide = _electronic_wide(frame, params)
    latest_label = _electronic_latest_label(frame, params)
    all_market = wide[wide["segment"].astype(str) == "All"].sort_values("year")
    latest = all_market.loc[all_market["year"].idxmax(), list(_ELECTRONIC_VENUES)]
    latest = latest.astype(float).sort_values(ascending=True)
    y_labels = ["Voice" if venue == "Voice/high-touch" else str(venue) for venue in latest.index]
    with figure_style("word_a4", style=style):
        fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="none")
        y_positions = np.arange(len(latest))
        for position, (venue, value) in enumerate(latest.items()):
            color = _ELECTRONIC_VENUE_COLORS[venue]
            ax.hlines(position, 0, value, color=color, linewidth=2.1, alpha=0.72)
            ax.scatter(value, position, s=66, color=color, edgecolor="white", zorder=3)
            ax.text(value + 1.0, position, f"{value:.1f}%", va="center", fontsize=8.8, color=color)
        ax.set_yticks(y_positions, y_labels)
        ax.set_xlim(0, max(65, float(latest.max()) + 8))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
        ax.grid(True, axis="x", color=FT_COLORS["grid"], linewidth=0.8, alpha=0.82)
        ax.grid(False, axis="y")
        ax.set_xlabel(f"Share of all IG+HY trading, {latest_label}")
        ax.set_ylabel("")
        ax.set_title(title, loc="left")
        fig.subplots_adjust(bottom=0.16, left=0.22, right=0.95, top=0.88)
        return fig


def _long_panel_plan(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
) -> dict[str, object] | None:
    if plot_date is None or not profile.numeric_columns or not profile.categorical_columns:
        return None
    value = _long_value_column(profile, frame)
    categories = [
        column
        for column in profile.categorical_columns
        if 1 < frame[column].dropna().astype(str).nunique() <= 16
        and _is_time_panel_category(frame, plot_date, column)
    ]
    if not categories:
        return None

    total_column, total_value = _find_total_filter(frame, categories)
    if total_column is not None:
        remaining = [column for column in categories if column != total_column]
        if remaining:
            line_category = min(remaining, key=lambda col: frame[col].astype(str).nunique())
            return {
                "value": value,
                "line_category": line_category,
                "filter_column": total_column,
                "filter_value": total_value,
            }

    line_category = min(categories, key=lambda col: frame[col].astype(str).nunique())
    return {"value": value, "line_category": line_category}


def _long_part_to_whole_plan(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
    long_plan: Mapping[str, object],
) -> dict[str, object] | None:
    if plot_date is None or len(profile.categorical_columns) < 2:
        return None
    value = str(long_plan["value"])
    line_category = str(long_plan["line_category"])
    segment_candidates = [
        column
        for column in profile.categorical_columns
        if column != line_category and frame[column].dropna().astype(str).nunique() <= 16
        and (plot_date is None or _is_time_panel_category(frame, plot_date, column))
    ]
    if not segment_candidates:
        return None
    segment = max(segment_candidates, key=lambda col: frame[col].dropna().astype(str).nunique())
    filter_value = _preferred_category_value(frame[line_category])
    return {
        "value": value,
        "segment": segment,
        "filter_column": line_category,
        "filter_value": filter_value,
        "exclude_total_segments": True,
    }


def _filtered_long_frame(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.DataFrame:
    filtered = frame.copy()
    filter_column = params.get("filter_column")
    filter_value = params.get("filter_value")
    if filter_column is not None and filter_value is not None:
        filtered = filtered[filtered[str(filter_column)].astype(str) == str(filter_value)]
    return filtered


def _long_time_pivot(
    frame: pd.DataFrame,
    plot_date: str | None,
    category: str,
    value: str,
) -> pd.DataFrame:
    if plot_date is None:
        raise ValueError("long time-series figures require a date column")
    pivot = frame.pivot_table(
        index=plot_date,
        columns=category,
        values=value,
        aggfunc="mean",
    )
    pivot = pivot.sort_index().dropna(how="all")
    if pivot.empty or len(pivot.columns) < 1:
        raise ValueError("no long panel observations available to plot")
    pivot.columns = [_display_label(column) for column in pivot.columns]
    return pivot


def _long_first_last_frame(
    frame: pd.DataFrame,
    plot_date: str | None,
    category: str,
    value: str,
) -> tuple[pd.DataFrame, str, str]:
    if plot_date is None:
        raise ValueError("slope charts require a date column")
    pivot = _long_time_pivot(frame, plot_date, category, value)
    if len(pivot.index) < 2:
        raise ValueError("slope chart requires at least two time periods")
    start = pivot.index.min()
    end = pivot.index.max()
    start_label = _date_label(start)
    end_label = _date_label(end)
    result = (
        pivot.loc[[start, end]]
        .T.reset_index()
        .rename(columns={"index": category, start: start_label, end: end_label})
    )
    return result, start_label, end_label


def _long_latest_frame(
    frame: pd.DataFrame,
    plot_date: str | None,
    category: str,
    value: str,
) -> pd.DataFrame:
    if plot_date is None:
        raise ValueError("latest-period rankings require a date column")
    latest_date = frame[plot_date].dropna().max()
    latest = frame.loc[frame[plot_date] == latest_date, [category, value]].copy()
    latest = latest.groupby(category, as_index=False)[value].mean()
    return latest.dropna()


def _long_value_column(profile: DataFrameProfile, frame: pd.DataFrame) -> str:
    preferred = [
        column
        for column in profile.numeric_columns
        if re.search(r"(percent|percentage|pct|share|value|amount|return)", column, re.I)
    ]
    if preferred:
        return max(preferred, key=lambda column: frame[column].notna().sum())
    return _best_numeric_columns(frame, list(profile.numeric_columns), count=1)[0]


def _find_total_filter(
    frame: pd.DataFrame,
    categories: Sequence[str],
) -> tuple[str | None, str | None]:
    for column in categories:
        values = frame[column].dropna().astype(str).drop_duplicates()
        for value in values:
            if _is_total_like_value(value):
                return column, value
    return None, None


def _preferred_category_value(series: pd.Series) -> str:
    values = list(series.dropna().astype(str).drop_duplicates())
    for preferred in ["All", "Total", "Combined", "Market"]:
        for value in values:
            if value.lower() == preferred.lower():
                return value
    return str(series.dropna().astype(str).mode().iloc[0])


def _is_total_like_value(value: object) -> bool:
    text = str(value).strip().lower()
    return text.startswith("total") or text in {"all electronic", "aggregate"}


def _has_two_time_periods(
    frame: pd.DataFrame,
    plot_date: str | None,
    category: str,
    value: str,
) -> bool:
    if plot_date is None:
        return False
    candidate = frame[[plot_date, category, value]].dropna()
    return candidate[plot_date].nunique() >= 2 and candidate[category].nunique() >= 2


def _is_time_panel_category(frame: pd.DataFrame, plot_date: str, category: str) -> bool:
    candidate = frame[[plot_date, category]].dropna()
    if candidate[plot_date].nunique() < 2:
        return False
    per_period = candidate.groupby(plot_date)[category].nunique()
    return bool(per_period.median() >= 2)


def _looks_like_year(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    years = numeric.astype(float)
    return bool(
        (years % 1 == 0).all()
        and years.between(1800, 2200).mean() >= 0.95
        and years.nunique() > 1
    )


def _year_to_datetime(series: pd.Series) -> pd.Series:
    years = pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)
    years = years.where(years != "<NA>", np.nan)
    return pd.to_datetime(years + "-12-31", errors="coerce")


def _sample_label(
    values: pd.Series | pd.DatetimeIndex | None,
    row_count: int,
) -> str:
    if values is None:
        return f"{row_count:,} observations"
    series = pd.Series(values).dropna()
    if series.empty:
        return f"{row_count:,} observations"
    return f"{_date_label(series.min())} to {_date_label(series.max())}"


def _date_label(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.month == 12 and timestamp.day == 31:
        return timestamp.strftime("%Y")
    return timestamp.strftime("%Y-%m-%d")


def _looks_like_percent_column(column: str, series: pd.Series) -> bool:
    if re.search(r"(percent|percentage|pct|share|rate|yield|spread)", column, re.I):
        return True
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    return bool(values.between(0, 100).mean() >= 0.95 and values.max() > 1.0)


def _looks_like_return_column(column: str) -> bool:
    return bool(
        re.search(r"(^|[_\s-])(ret|return|returns)([_\s-]|$)", column, re.I)
        or column.lower() in {"mkt-rf", "smb", "hml", "rf"}
    )


def _best_numeric_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    count: int,
) -> list[str]:
    ranked = sorted(
        columns,
        key=lambda column: (
            int(frame[column].notna().sum()),
            int(pd.to_numeric(frame[column], errors="coerce").nunique()),
        ),
        reverse=True,
    )
    return ranked[:count]


def _best_category(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return min(
        columns,
        key=lambda column: (
            abs(frame[column].dropna().astype(str).nunique() - 6),
            frame[column].dropna().astype(str).nunique(),
        ),
    )


def _mostly_positive(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return bool(not values.empty and (values > 0).mean() >= 0.95)


def _strongest_numeric_pair(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[str, str]:
    candidates = _best_numeric_columns(frame, columns, count=min(8, len(columns)))
    corr = frame[candidates].corr(numeric_only=True).abs()
    if corr.shape[0] < 2:
        return candidates[0], candidates[1]
    corr = corr.mask(np.eye(len(corr), dtype=bool))
    stacked = corr.stack().dropna()
    if stacked.empty:
        return candidates[0], candidates[1]
    first, second = stacked.idxmax()
    return str(first), str(second)


def _renamed_numeric_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    mapping = _unique_display_mapping(columns)
    plot_frame = frame[list(columns)].rename(columns=mapping).copy()
    return plot_frame, [mapping[column] for column in columns]


def _unique_display_mapping(columns: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for column in columns:
        label = _display_label(column)
        candidate = label
        suffix = 2
        while candidate in used:
            candidate = f"{label} {suffix}"
            suffix += 1
        mapping[column] = candidate
        used.add(candidate)
    return mapping


def _infer_return_scale(frame: pd.DataFrame) -> str:
    values = frame.stack().dropna()
    if values.empty:
        return "percent"
    try:
        return infer_return_scale(values)
    except ValueError:
        return "percent"


def _limit_category_levels(
    frame: pd.DataFrame,
    category: str,
    value: str,
    *,
    limit: int,
) -> pd.DataFrame:
    levels = (
        frame[[category, value]]
        .dropna()
        .groupby(category)[value]
        .size()
        .sort_values(ascending=False)
        .head(limit)
        .index
    )
    return frame[frame[category].isin(levels)].copy()


def _item_sample(
    frame: pd.DataFrame,
    profile: DataFrameProfile,
    plot_date: str | None,
    item: FigurePlanItem,
) -> str:
    if item.kind == "electronic_unconditional_bubble_matrix":
        try:
            wide = _electronic_full_year_wide(frame, item.params)
            years = wide["year"].dropna().astype(int)
            if not years.empty:
                return f"{int(years.min())} full year to {int(years.max())} full year"
        except Exception:
            return profile.sample
    if plot_date is None:
        return profile.sample
    try:
        if item.kind.startswith("long_"):
            filtered = _filtered_long_frame(frame, item.params)
            dates = filtered[plot_date].dropna()
        else:
            dates = frame[plot_date].dropna()
        if dates.empty:
            return profile.sample
        return f"{_date_label(dates.min())} to {_date_label(dates.max())}"
    except Exception:
        return profile.sample


def _item_units(profile: DataFrameProfile, item: FigurePlanItem) -> str:
    params = dict(item.params)
    if item.kind == "electronic_venue_mix":
        return "Share of electronic trading (%)"
    if item.kind.startswith("electronic_"):
        return "Share of trading volume (%)"
    if item.kind == "portfolio_growth_of_one_dollar":
        return "Growth of one dollar, log scale"
    if item.kind == "portfolio_drawdowns":
        return "Drawdown"
    if item.kind == "portfolio_risk_return":
        return "Annualized return and volatility (%)"
    if item.kind.startswith("portfolio_return_"):
        return "Return"
    values = []
    for key in ["value", "x", "y"]:
        if key in params:
            values.append(str(params[key]))
    if "columns" in params:
        values.extend(str(column) for column in params["columns"])
    if item.kind == "cumulative_returns":
        return "Growth of one dollar, log scale"
    if values and all(value in profile.percent_columns for value in values):
        return "Percent"
    if values and all(value in profile.return_columns for value in values):
        return "Return"
    if len(values) == 1:
        return _display_label(values[0])
    return "Data values"


def _rendered_issues(fig: plt.Figure) -> list[FigureIssue]:
    issues: list[FigureIssue] = []
    issues.extend(validate_titles_within_canvas(fig))
    for axis in fig.axes:
        issues.extend(validate_display_labels(axis))
        issues.extend(validate_markers_within_axes(axis))
        issues.extend(validate_no_text_overlap(axis))
        issues.extend(validate_no_tick_label_overlap(axis))
        issues.extend(validate_no_tick_label_overlap(axis, axis="y"))
        issues.extend(validate_category_label_count(axis))
        issues.extend(validate_category_label_count(axis, axis="y"))
        issues.extend(validate_series_identification(axis))
    return issues


def _skip_message(item: FigurePlanItem, issues: Sequence[FigureIssue]) -> str:
    details = "; ".join(issue.message for issue in issues)
    return f"{item.title}: {details}"


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "figure"


def _delete_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        with suppress(OSError):
            path.unlink(missing_ok=True)


def _clean_suite_outputs(output_dir: Path) -> None:
    """Remove stale files created by previous dataframe-suite runs."""

    for path in output_dir.glob("figure_suite_*"):
        if path.is_file():
            with suppress(OSError):
                path.unlink()
    docx_path = output_dir / "figure_suite.docx"
    with suppress(OSError):
        docx_path.unlink()
