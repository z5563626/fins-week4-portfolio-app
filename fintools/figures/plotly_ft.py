"""Optional FT-style helpers for Plotly figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .theme import FT_BACKGROUND, FT_COLOR_CYCLE, FT_COLORS

PLOTLY_INSTALL_MESSAGE = (
    "Optional Plotly figure support requires plotly>=6.1.1 and kaleido>=1.0. "
    "Install with the repo interpreter using -m pip install -r requirements-plotly.txt. "
    "Static PNG/PDF export also requires Chrome or Chromium for Kaleido v1."
)


def _plotly_modules() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(PLOTLY_INSTALL_MESSAGE) from exc
    return go, pio


def ft_plotly_template(*, ft_background: bool = False) -> object:
    """Return a Plotly template with FT-inspired colors and typography."""

    go, _pio = _plotly_modules()
    background = FT_BACKGROUND if ft_background else "white"
    return go.layout.Template(
        layout=go.Layout(
            colorway=FT_COLOR_CYCLE,
            font={"family": "Arial, sans-serif", "color": FT_COLORS["charcoal"], "size": 13},
            paper_bgcolor=background,
            plot_bgcolor=background,
            title={
                "font": {"color": FT_COLORS["charcoal"], "size": 18},
                "x": 0,
                "xanchor": "left",
            },
            xaxis={
                "showgrid": False,
                "showline": True,
                "linecolor": FT_COLORS["axis"],
                "tickcolor": FT_COLORS["axis"],
                "ticks": "outside",
                "title": None,
            },
            yaxis={
                "gridcolor": FT_COLORS["grid"],
                "showgrid": True,
                "zerolinecolor": FT_COLORS["axis"],
                "title": None,
            },
            legend={"title": None, "orientation": "h", "x": 0, "y": 1.08},
            margin={"l": 62, "r": 24, "t": 70, "b": 52},
        )
    )


def apply_ft_plotly_layout(
    fig: object,
    *,
    title: str | None = None,
    ft_background: bool = False,
    showlegend: bool = True,
) -> object:
    """Apply the FT-inspired Plotly template to an existing figure."""

    _go, pio = _plotly_modules()
    template = ft_plotly_template(ft_background=ft_background)
    template_name = "fins_ft_background" if ft_background else "fins_ft"
    pio.templates[template_name] = template
    fig.update_layout(template=template_name, showlegend=showlegend)
    if title:
        fig.update_layout(title=title)
    return fig


def export_plotly_image(
    fig: object,
    output_path: str | Path,
    *,
    width: int = 1100,
    height: int = 700,
    scale: float = 2.0,
) -> Path:
    """Export a Plotly figure to a static image with actionable dependency errors."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(path, width=width, height=height, scale=scale)
    except Exception as exc:  # pragma: no cover - depends on Kaleido/Chrome runtime
        raise RuntimeError(PLOTLY_INSTALL_MESSAGE) from exc
    return path
