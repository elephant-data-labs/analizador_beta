from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from beta_analyzer.config import CaseParameters
from beta_analyzer.returns import compute_returns


def _price_frame(prices: list[float], start: str) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "adj_close": prices,
            "volume": [1_000] * len(prices),
        },
        index=index,
    )


def _daily_case() -> CaseParameters:
    return CaseParameters(
        company_name="Empresa de Prueba",
        stock_ticker="TEST-A.SN",
        index_name="Índice de Prueba",
        index_ticker="^TEST",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        frequency="D",
    )


def test_simple_return_formula() -> None:
    stock = _price_frame([100.0, 110.0, 99.0], "2024-01-01")
    market = _price_frame([50.0, 55.0, 49.5], "2024-01-01")

    result = compute_returns(stock, market, _daily_case())

    assert result.n_obs == 2
    assert result.stock.iloc[0] == pytest.approx(0.10)
    assert result.market.iloc[0] == pytest.approx(0.10)
    assert result.stock.iloc[1] == pytest.approx(-0.10, abs=1e-9)


def test_alignment_drops_dates_without_both_series() -> None:
    stock = _price_frame([100.0, 102.0, 104.0, 106.0], "2024-01-01")
    # Al índice le falta un día que la acción sí tiene (huevo típico de datos reales).
    market = pd.concat([_price_frame([50.0, 51.0], "2024-01-01"), _price_frame([53.0], "2024-01-04")])

    result = compute_returns(stock, market, _daily_case())

    # Solo deben sobrevivir los retornos donde ambas series tienen ambos días.
    assert result.n_obs <= 2
    assert result.stock.index.equals(result.market.index)


def test_empty_intersection_raises() -> None:
    stock = _price_frame([100.0, 101.0], "2024-01-01")
    market = _price_frame([50.0, 51.0], "2030-01-01")

    with pytest.raises(ValueError):
        compute_returns(stock, market, _daily_case())
