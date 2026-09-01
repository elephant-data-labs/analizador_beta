"""Gráficos interactivos (Plotly) del análisis: dispersión retorno acción vs.
retorno mercado con recta de regresión, y precio con días atípicos marcados.

Sigue siendo cálculo/gráfico determinístico y sin IA: Plotly solo dibuja los
números que ya calcularon `regression.py` y `outliers.py`; no hay ningún
modelo de lenguaje generando ni el contenido ni la interpretación del
gráfico. El interactivo (zoom, hover con el detalle de cada punto) es una
característica del motor de gráficos, no una nueva fuente de cálculo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import CaseParameters
from .regression import RegressionResult
from .returns import ReturnSeries

_PRIMARY = "#157a8a"
_TEXT = "#0b3d62"
_POINT = "#3c6b82"
_OUTLIER_POINT = "#d64545"


def scatter_with_regression(
    returns: ReturnSeries, result: RegressionResult, case: CaseParameters
) -> go.Figure:
    x = returns.market.to_numpy()
    y = returns.stock.to_numpy()
    fechas = returns.market.index

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observaciones",
            marker={"size": 7, "color": _POINT, "opacity": 0.6, "line": {"width": 0}},
            customdata=np.stack([fechas.strftime("%Y-%m-%d"), y * 100, x * 100], axis=-1),
            hovertemplate=(
                "Fecha: %{customdata[0]}<br>"
                f"Retorno {case.company_name}: " + "%{customdata[1]:.2f}%<br>"
                f"Retorno {case.index_name}: " + "%{customdata[2]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = result.alpha + result.beta * x_line
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Recta de regresión OLS",
            line={"color": _PRIMARY, "width": 2.5},
            hoverinfo="skip",
        )
    )

    sign = "+" if result.alpha >= 0 else "-"
    equation = (
        f"R_i = {sign}{abs(result.alpha):.5f} "
        f"{'+' if result.beta >= 0 else '-'} {abs(result.beta):.4f}·R_m"
    )
    annotation = f"{equation}<br>β = {result.beta:.4f}<br>R² = {result.r_squared:.4f}"
    fig.add_annotation(
        text=annotation,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        font={"size": 12, "color": _TEXT},
        bgcolor="white",
        bordercolor="#d7e6ee",
        borderwidth=1,
        borderpad=6,
    )

    fig.add_hline(y=0, line_width=1, line_color="#c9d8e0")
    fig.add_vline(x=0, line_width=1, line_color="#c9d8e0")

    fig.update_layout(
        title={
            "text": f"{case.company_name} vs. {case.index_name}<br>"
            f"<sub>{result.start_date} — {result.end_date}</sub>",
            "font": {"color": _TEXT, "size": 15},
        },
        xaxis_title=f"Retorno {case.index_name} ({case.frequency_label.lower()})",
        yaxis_title=f"Retorno {case.company_name} ({case.frequency_label.lower()})",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"color": _TEXT},
        plot_bgcolor="white",
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
        hovermode="closest",
    )
    fig.update_xaxes(tickformat=".1%", gridcolor="#eef3f6", zeroline=False)
    fig.update_yaxes(tickformat=".1%", gridcolor="#eef3f6", zeroline=False)
    return fig


def price_with_outliers(
    price: pd.Series, outliers: pd.DataFrame, case: CaseParameters
) -> go.Figure:
    """Precio de la acción (a la misma frecuencia que el análisis) con los
    días atípicos detectados en `outliers.py` marcados encima. Interactivo
    (Plotly): al pasar el mouse sobre un punto atípico se ve la fecha, el
    precio y su z-score."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            mode="lines",
            name=case.company_name,
            line={"color": _PRIMARY, "width": 1.6},
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>",
        )
    )

    if not outliers.empty:
        marked = price.reindex(outliers.index).dropna()
        z_scores = outliers.loc[marked.index, "z_score"]
        fig.add_trace(
            go.Scatter(
                x=marked.index,
                y=marked.values,
                mode="markers",
                name="Días atípicos",
                marker={
                    "color": _OUTLIER_POINT,
                    "size": 10,
                    "line": {"color": "white", "width": 1},
                },
                customdata=z_scores.to_numpy(),
                hovertemplate=(
                    "Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<br>"
                    "Z-score: %{customdata:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={
            "text": f"Precio de {case.company_name} — días atípicos marcados "
            f"({case.frequency_label.lower()})",
            "font": {"color": _TEXT, "size": 14},
        },
        yaxis_title="Precio",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"color": _TEXT},
        plot_bgcolor="white",
        margin={"l": 60, "r": 20, "t": 70, "b": 40},
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="#eef3f6")
    fig.update_yaxes(gridcolor="#eef3f6")
    return fig
