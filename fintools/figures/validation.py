"""Validation helpers for report-ready figures."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.text import Text


@dataclass(frozen=True)
class FigureIssue:
    """One issue found by a figure validation helper."""

    code: str
    message: str


def infer_return_scale(values: pd.Series | np.ndarray | list[float]) -> str:
    """Infer whether returns look like percent units or decimals."""

    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        raise ValueError("return series is empty")
    max_abs = float(series.abs().max())
    if max_abs > 2:
        return "percent"
    if max_abs < 0.5:
        return "decimal"
    return "ambiguous"


def validate_figure_context(context: object) -> list[FigureIssue]:
    """Check that a figure context has the fields students need for a caption."""

    issues: list[FigureIssue] = []
    for field in ["title", "source", "units"]:
        value = getattr(context, field, "")
        if not str(value).strip():
            issues.append(FigureIssue(f"missing_{field}", f"Figure context is missing {field}."))
    return issues


def validate_axes_labels(ax: plt.Axes) -> list[FigureIssue]:
    """Check that an axis has clear labels."""

    issues: list[FigureIssue] = []
    if not ax.get_xlabel().strip():
        issues.append(FigureIssue("missing_x_label", "Figure is missing an x-axis label."))
    if not ax.get_ylabel().strip():
        issues.append(FigureIssue("missing_y_label", "Figure is missing a y-axis label."))
    return issues


RAW_DISPLAY_LABELS = {
    "category",
    "country",
    "date",
    "group",
    "industry",
    "month",
    "return",
    "series",
    "size",
    "value",
    "variable",
    "year",
}


def _display_label_issue(kind: str, text: str) -> FigureIssue | None:
    label = text.strip()
    if not label:
        return None
    if "_" in label:
        return FigureIssue(
            "raw_display_label",
            f"{kind} label '{label}' contains underscores; use a display-ready label.",
        )
    if label.lower() in RAW_DISPLAY_LABELS and label == label.lower():
        return FigureIssue(
            "raw_display_label",
            f"{kind} label '{label}' looks like a raw dataframe field name.",
        )
    first = label[0]
    if first.isalpha() and first.islower():
        return FigureIssue(
            "lowercase_display_label",
            f"{kind} label '{label}' should start with an uppercase letter.",
        )
    return None


def validate_display_labels(ax: plt.Axes) -> list[FigureIssue]:
    """Check that visible labels are presentation-ready, not raw field names."""

    issues: list[FigureIssue] = []
    candidates = [
        ("x-axis", ax.get_xlabel()),
        ("y-axis", ax.get_ylabel()),
    ]
    legend = ax.get_legend()
    if legend is not None:
        legend_title = legend.get_title().get_text()
        if legend_title.strip():
            candidates.append(("legend title", legend_title))
    for kind, label in candidates:
        issue = _display_label_issue(kind, label)
        if issue is not None:
            issues.append(issue)
    return issues


def validate_legend_present(ax: plt.Axes) -> list[FigureIssue]:
    """Check that a figure has a legend."""

    if ax.get_legend() is None:
        return [FigureIssue("missing_legend", "Figure is missing a legend.")]
    return []


def validate_horizontal_grid(ax: plt.Axes) -> list[FigureIssue]:
    """Check that gridlines are horizontal-only."""

    issues: list[FigureIssue] = []
    if any(line.get_visible() for line in ax.get_xgridlines()):
        issues.append(FigureIssue("x_grid_visible", "Figure has vertical gridlines."))
    if not any(line.get_visible() for line in ax.get_ygridlines()):
        issues.append(FigureIssue("missing_y_grid", "Figure is missing horizontal gridlines."))
    return issues


def validate_no_tick_label_overlap(
    ax: plt.Axes,
    *,
    axis: str = "x",
) -> list[FigureIssue]:
    """Check rendered tick labels for overlap on one axis."""

    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    boxes = [
        label.get_window_extent(renderer)
        for label in labels
        if label.get_visible() and label.get_text().strip()
    ]
    boxes = sorted(boxes, key=lambda box: box.x0 if axis == "x" else box.y0)
    for previous, current in pairwise(boxes):
        if previous.overlaps(current):
            return [
                FigureIssue(
                    f"{axis}_tick_label_overlap",
                    f"Adjacent {axis}-axis tick labels overlap.",
                )
            ]
    return []


def validate_no_text_overlap(
    ax: plt.Axes,
    *,
    gid: str | None = "figure_label",
) -> list[FigureIssue]:
    """Check rendered annotation text for overlap."""

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    texts = []
    for text in ax.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if gid is not None and text.get_gid() != gid:
            continue
        texts.append(text)

    boxes = []
    for text in texts:
        bbox_patch = text.get_bbox_patch()
        box = (
            bbox_patch.get_window_extent(renderer)
            if bbox_patch is not None
            else text.get_window_extent(renderer)
        )
        boxes.append((text.get_text(), box))
    for index, (left_text, left_box) in enumerate(boxes):
        for right_text, right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                return [
                    FigureIssue(
                        "text_overlap",
                        f"Text labels overlap: '{left_text}' and '{right_text}'.",
                    )
                ]
    return []


def validate_unique_series_colors(
    ax: plt.Axes,
    *,
    minimum: int | None = None,
) -> list[FigureIssue]:
    """Check that visible multi-point line series use distinct colors."""

    lines = [
        line
        for line in ax.get_lines()
        if line.get_visible()
        and line.get_linewidth() > 0
        and len(line.get_xdata()) > 1
        and len(line.get_ydata()) > 1
    ]
    if not lines:
        return []

    required = min(len(lines), minimum or len(lines))
    colors = [line.get_color() for line in lines[:required]]
    if len(set(colors)) < required:
        return [
            FigureIssue(
                "repeated_series_color",
                f"Expected {required} distinct series colors, found {len(set(colors))}.",
            )
        ]
    return []


def validate_markers_within_axes(
    ax: plt.Axes,
    *,
    tolerance_px: float = 1.0,
) -> list[FigureIssue]:
    """Check that scatter markers are not clipped by the axes boundary."""

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    for collection in ax.collections:
        if not isinstance(collection, PathCollection) or not collection.get_visible():
            continue
        offsets = np.ma.asarray(collection.get_offsets())
        if offsets.size == 0:
            continue
        offsets = np.asarray(offsets.filled(np.nan), dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            continue
        finite = np.isfinite(offsets[:, 0]) & np.isfinite(offsets[:, 1])
        if not finite.any():
            continue
        offsets = offsets[finite]
        sizes = np.asarray(collection.get_sizes(), dtype=float)
        if sizes.size == 0:
            continue
        sizes = np.repeat(sizes, len(offsets)) if sizes.size == 1 else sizes[finite]
        linewidths = np.asarray(collection.get_linewidths(), dtype=float)
        linewidth = float(np.nanmax(linewidths)) if linewidths.size else 0.0
        points_to_px = ax.figure.dpi / 72.0
        radii_px = (np.sqrt(np.maximum(sizes, 1.0)) / 2.0 + linewidth / 2.0) * points_to_px
        points = collection.get_offset_transform().transform(offsets)
        left = points[:, 0] - radii_px
        right = points[:, 0] + radii_px
        bottom = points[:, 1] - radii_px
        top = points[:, 1] + radii_px
        if (
            np.nanmin(left) < axes_box.x0 - tolerance_px
            or np.nanmax(right) > axes_box.x1 + tolerance_px
            or np.nanmin(bottom) < axes_box.y0 - tolerance_px
            or np.nanmax(top) > axes_box.y1 + tolerance_px
        ):
            return [
                FigureIssue(
                    "marker_clipped",
                    "A scatter or bubble marker falls outside the axes boundary.",
                )
            ]
    return []


def validate_titles_within_canvas(
    fig: plt.Figure,
    *,
    tolerance_px: float = 1.0,
) -> list[FigureIssue]:
    """Check rendered axis and figure titles stay inside the image canvas."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_box = fig.bbox
    title_artists = []
    if getattr(fig, "_suptitle", None) is not None:
        title_artists.append(fig._suptitle)
    for ax in fig.axes:
        title_artists.extend([ax.title, ax._left_title, ax._right_title])

    for text in title_artists:
        if text is None or not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if (
            box.x0 < canvas_box.x0 - tolerance_px
            or box.x1 > canvas_box.x1 + tolerance_px
            or box.y0 < canvas_box.y0 - tolerance_px
            or box.y1 > canvas_box.y1 + tolerance_px
        ):
            return [
                FigureIssue(
                    "title_outside_canvas",
                    f"Title '{text.get_text()}' extends outside the figure canvas.",
                )
            ]
    return []


def validate_equal_subplot_widths(
    fig: plt.Figure,
    *,
    tolerance_fraction: float = 0.02,
) -> list[FigureIssue]:
    """Check visible data axes use equal widths in an intentional comparison layout."""

    axes = [
        ax
        for ax in fig.axes
        if ax.get_visible()
        and ax.has_data()
        and not ax.get_label().startswith("<colorbar")
    ]
    if len(axes) <= 1:
        return []

    widths = [float(ax.get_position().width) for ax in axes]
    if max(widths) - min(widths) > tolerance_fraction:
        return [
            FigureIssue(
                "unequal_subplot_widths",
                "Comparison subplots have unequal widths.",
            )
        ]
    return []


def validate_series_identification(ax: plt.Axes) -> list[FigureIssue]:
    """Check that multi-series line plots identify their series."""

    labeled_lines = [
        line
        for line in ax.get_lines()
        if line.get_visible()
        and len(line.get_xdata()) > 1
        and line.get_label()
        and not line.get_label().startswith("_")
    ]
    if len(labeled_lines) <= 1:
        return []
    has_legend = ax.get_legend() is not None
    has_direct_labels = any(
        text.get_gid() == "figure_label" and text.get_visible() and text.get_text().strip()
        for text in ax.findobj(match=Text)
    )
    if not has_legend and not has_direct_labels:
        return [
            FigureIssue(
                "unidentified_series",
                "Multi-series figure has no legend or direct labels.",
            )
        ]
    return []


def validate_category_label_count(
    ax: plt.Axes,
    *,
    axis: str = "x",
    max_labels: int = 12,
) -> list[FigureIssue]:
    """Check that categorical axes do not render too many labels."""

    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    visible_labels = [
        label for label in labels if label.get_visible() and label.get_text().strip()
    ]
    if len(visible_labels) > max_labels:
        return [
            FigureIssue(
                f"too_many_{axis}_tick_labels",
                f"{len(visible_labels)} visible {axis}-axis labels exceeds {max_labels}.",
            )
        ]
    return []


def validate_docx_images_fit_page(
    path: str | Path,
    *,
    tolerance_inches: float = 0.01,
) -> list[FigureIssue]:
    """Check that Word images fit inside the usable page width."""

    from docx import Document

    emu_per_inch = 914400
    docx_path = Path(path)
    if not docx_path.exists():
        return [FigureIssue("missing_docx", f"Word document does not exist: {docx_path}")]

    document = Document(docx_path)
    issues: list[FigureIssue] = []
    usable_widths = []
    for section in document.sections:
        if section.page_width > section.page_height:
            issues.append(
                FigureIssue("landscape_page", "Word document contains a landscape section.")
            )
        usable_widths.append(
            float(section.page_width - section.left_margin - section.right_margin)
            / emu_per_inch
        )
    usable_width = min(usable_widths) if usable_widths else 0.0
    for index, shape in enumerate(document.inline_shapes, start=1):
        width = float(shape.width) / emu_per_inch
        if width > usable_width + tolerance_inches:
            issues.append(
                FigureIssue(
                    "image_too_wide",
                    (
                        f"Inline image {index} is {width:.2f}in wide, exceeding "
                        f"usable page width {usable_width:.2f}in."
                    ),
                )
            )
    return issues


def validate_image_not_blank(
    path: str | Path,
    *,
    min_channel_range: float = 10.0,
) -> list[FigureIssue]:
    """Check that an exported image contains non-blank pixel variation."""

    from PIL import Image

    image_path = Path(path)
    if not image_path.exists():
        return [FigureIssue("missing_image", f"Image does not exist: {image_path}")]
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=float)
    channel_range = float(np.ptp(pixels.reshape(-1, 3), axis=0).max())
    if channel_range < min_channel_range:
        return [
            FigureIssue(
                "blank_image",
                f"Image channel range {channel_range:.1f} is below {min_channel_range:.1f}.",
            )
        ]
    return []


def validate_word_readability(
    fig: plt.Figure,
    *,
    width_inches: float = 6.27,
    min_font_size: float = 8.0,
) -> list[FigureIssue]:
    """Check basic Word/A4 readability constraints."""

    issues: list[FigureIssue] = []
    figure_width = float(fig.get_size_inches()[0])
    if figure_width > width_inches + 0.05:
        issues.append(
            FigureIssue(
                "too_wide",
                f"Figure width {figure_width:.2f}in exceeds Word width {width_inches:.2f}in.",
            )
        )
    for text in fig.findobj(match=Text):
        if text.get_text().strip() and text.get_fontsize() < min_font_size:
            issues.append(
                FigureIssue(
                    "small_text",
                    f"Text '{text.get_text()[:30]}' is below {min_font_size:.1f}pt.",
                )
            )
            break
    return issues
