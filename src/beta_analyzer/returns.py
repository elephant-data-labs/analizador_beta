"""Cálculo de retornos y alineación acción/índice por fecha y frecuencia.

Metodología principal (ver README): retorno simple, R_t = P_t / P_{t-1} - 1,
sobre el precio ajustado por dividendos. El retorno logarítmico queda
disponible como alternativa explícita (`kind="log"`), no como default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CaseParameters, FREQUENCY_RESAMPLE_RULE


@dataclass(frozen=True)
class ReturnSeries:
    """Retornos de acción e índice ya alineados por fecha, listos para regresión."""

    stock: pd.Series
    market: pd.Series
    frequency: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    @property
    def n_obs(self) -> int:
        return len(self.stock)


def _resample_prices(prices: pd.Series, rule: str) -> pd.Series:
    """Precio de cierre de cada período (último valor disponible del período)."""
    return prices.resample(rule).last().dropna()


def resample_price_field(prices: pd.DataFrame, field: str, case: CaseParameters) -> pd.Series:
    """Precio de una acción (o índice) remuestreado a la frecuencia del caso.

    Usa exactamente la misma regla de remuestreo que `compute_returns`
    (`FREQUENCY_RESAMPLE_RULE`), expuesta aparte para poder graficar el
    precio en la misma frecuencia/fechas que los retornos ya calculados —
    por ejemplo, para marcar sobre el precio los días que `outliers.py`
    identificó como atípicos en `ReturnSeries`.
    """
    rule = FREQUENCY_RESAMPLE_RULE[case.frequency]
    resolved_field = field if field in prices.columns else "close"
    return _resample_prices(prices[resolved_field], rule)


def _price_returns(prices: pd.Series, kind: str) -> pd.Series:
    if kind == "simple":
        return prices.pct_change()
    if kind == "log":
        return np.log(prices / prices.shift(1))
    raise ValueError(f"Tipo de retorno no soportado: {kind!r}")


def compute_returns(
    stock_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    case: CaseParameters,
    kind: str = "simple",
) -> ReturnSeries:
    """Retornos de acción e índice, remuestreados a la frecuencia del caso,
    alineados por fecha (intersección) y sin los NaN iniciales del cálculo
    de retornos.
    """
    rule = FREQUENCY_RESAMPLE_RULE[case.frequency]

    stock_field = case.price_field if case.price_field in stock_prices.columns else "close"
    market_field = "close"  # los índices no tienen adj_close propio (ver data_sources)

    stock_close = _resample_prices(stock_prices[stock_field], rule)
    market_close = _resample_prices(market_prices[market_field], rule)

    stock_ret = _price_returns(stock_close, kind)
    market_ret = _price_returns(market_close, kind)

    aligned = pd.concat({"stock": stock_ret, "market": market_ret}, axis=1, join="inner")
    aligned = aligned.dropna(how="any")

    if aligned.empty:
        raise ValueError(
            "No quedaron observaciones tras alinear acción e índice por fecha. "
            "Revise que ambas series cubran la misma ventana."
        )

    return ReturnSeries(
        stock=aligned["stock"],
        market=aligned["market"],
        frequency=case.frequency,
        start_date=aligned.index.min(),
        end_date=aligned.index.max(),
    )
