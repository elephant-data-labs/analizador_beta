from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from beta_analyzer import data_sources, pipeline
from beta_analyzer.config import CaseParameters
from beta_analyzer.data_sources import DataSourceError


def _synthetic_price_frame(n: int, start_price: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = start_price * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    index = pd.bdate_range("2024-01-02", periods=n)
    frame = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "adj_close": prices, "volume": 1000},
        index=index,
    )
    frame.index.name = "date"
    return frame


def _case() -> CaseParameters:
    return CaseParameters(
        company_name="Empresa de Prueba",
        stock_ticker="TEST-A.SN",
        index_name="Índice de Prueba",
        index_ticker="^TEST",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        frequency="D",
    )


def test_run_beta_case_falls_back_to_replicated_index_when_stooq_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock_prices = _synthetic_price_frame(40, 100.0, seed=1)
    replicated_prices = _synthetic_price_frame(40, 100.0, seed=2)

    monkeypatch.setattr(data_sources, "fetch_stock_prices", lambda *a, **k: stock_prices)
    monkeypatch.setattr(
        data_sources,
        "fetch_index_prices",
        lambda *a, **k: (_ for _ in ()).throw(DataSourceError("Stooq bloqueado (anti-bot)")),
    )
    monkeypatch.setattr(
        data_sources, "build_replicated_index_prices", lambda *a, **k: replicated_prices
    )

    result = pipeline.run_beta_case(_case())

    assert "replicado" in result.index_source_label.lower()
    assert "Stooq bloqueado" in result.index_source_label
    assert result.regression.n_obs > 0


def test_run_beta_case_uses_stooq_label_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    stock_prices = _synthetic_price_frame(40, 100.0, seed=3)
    index_prices = _synthetic_price_frame(40, 100.0, seed=4)

    monkeypatch.setattr(data_sources, "fetch_stock_prices", lambda *a, **k: stock_prices)
    monkeypatch.setattr(data_sources, "fetch_index_prices", lambda *a, **k: index_prices)

    result = pipeline.run_beta_case(_case())

    assert "Stooq" in result.index_source_label
    assert "replicado" not in result.index_source_label.lower()


def test_run_beta_case_without_fallback_reraises_stooq_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stock_prices = _synthetic_price_frame(40, 100.0, seed=5)

    monkeypatch.setattr(data_sources, "fetch_stock_prices", lambda *a, **k: stock_prices)
    monkeypatch.setattr(
        data_sources,
        "fetch_index_prices",
        lambda *a, **k: (_ for _ in ()).throw(DataSourceError("Stooq bloqueado")),
    )

    with pytest.raises(DataSourceError):
        pipeline.run_beta_case(_case(), allow_replicated_index_fallback=False)


def test_run_beta_case_with_yahoo_index_source_uses_yahoo_not_stooq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con `index_source='yahoo'` (p. ej. S&P 500 / ^GSPC, ver
    catalog.KNOWN_INDICES) el índice se baja de Yahoo Finance como cualquier
    otro ticker — nunca debería llamar a Stooq ni al índice replicado."""
    stock_prices = _synthetic_price_frame(40, 100.0, seed=10)
    index_prices = _synthetic_price_frame(40, 100.0, seed=11)

    calls: dict[str, list[str]] = {"fetch_stock_prices": []}

    def fake_fetch_stock_prices(ticker: str, *a: object, **k: object) -> pd.DataFrame:
        calls["fetch_stock_prices"].append(ticker)
        return stock_prices if ticker == "TEST-A.SN" else index_prices

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("index_source='yahoo' no debería llamar a Stooq")

    monkeypatch.setattr(data_sources, "fetch_stock_prices", fake_fetch_stock_prices)
    monkeypatch.setattr(data_sources, "fetch_index_prices", _boom)
    monkeypatch.setattr(data_sources, "build_replicated_index_prices", _boom)

    case = CaseParameters(
        company_name="Empresa de Prueba",
        stock_ticker="TEST-A.SN",
        index_name="S&P 500",
        index_ticker="^GSPC",
        index_source="yahoo",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        frequency="D",
    )
    result = pipeline.run_beta_case(case)

    assert calls["fetch_stock_prices"] == ["TEST-A.SN", "^GSPC"]
    assert "Yahoo Finance" in result.index_source_label
    assert "Stooq" not in result.index_source_label
    assert result.regression.n_obs > 0


def test_manual_csv_takes_priority_even_with_yahoo_index_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """El CSV manual en data/raw/ manda sea cual sea `index_source` — ver
    `pipeline._fetch_market_prices`."""
    stock_prices = _synthetic_price_frame(40, 100.0, seed=12)
    manual_index_prices = _synthetic_price_frame(40, 100.0, seed=13)

    csv_path = tmp_path / "gspc_manual.csv"
    manual_index_prices.reset_index().rename(
        columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    ).drop(columns=["adj_close"]).to_csv(csv_path, index=False)

    monkeypatch.setattr(pipeline, "_default_manual_csv_path", lambda ticker: csv_path)
    monkeypatch.setattr(data_sources, "fetch_stock_prices", lambda *a, **k: stock_prices)

    case = CaseParameters(
        company_name="Empresa de Prueba",
        stock_ticker="TEST-A.SN",
        index_name="S&P 500",
        index_ticker="^GSPC",
        index_source="yahoo",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        frequency="D",
    )
    result = pipeline.run_beta_case(case)

    assert "CSV descargado a mano" in result.index_source_label


def test_run_beta_case_auto_detects_manual_csv_without_touching_the_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Si existe `data/raw/<ticker>_manual.csv` (descargado a mano una vez,
    ver README), el pipeline lo usa automáticamente — sin pasar
    `manual_index_csv` a mano y sin llamar a Stooq/Yahoo para el índice."""
    stock_prices = _synthetic_price_frame(40, 100.0, seed=8)
    manual_index_prices = _synthetic_price_frame(40, 100.0, seed=9)

    csv_path = tmp_path / "test_manual.csv"
    manual_index_prices.reset_index().rename(
        columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    ).drop(columns=["adj_close"]).to_csv(csv_path, index=False)

    monkeypatch.setattr(pipeline, "_default_manual_csv_path", lambda ticker: csv_path)
    monkeypatch.setattr(data_sources, "fetch_stock_prices", lambda *a, **k: stock_prices)

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("no debería llamar a Stooq si ya hay un CSV manual disponible")

    monkeypatch.setattr(data_sources, "fetch_index_prices", _boom)

    result = pipeline.run_beta_case(_case())

    assert "CSV descargado a mano" in result.index_source_label
    assert result.regression.n_obs > 0


def test_replicated_index_excludes_the_stock_being_analyzed(monkeypatch: pytest.MonkeyPatch) -> None:
    """La acción en análisis no debe quedar dentro de su propio índice de
    mercado (evita comparar la acción contra un mercado que la incluye)."""
    stock_prices = _synthetic_price_frame(40, 100.0, seed=6)
    replicated_prices = _synthetic_price_frame(40, 100.0, seed=7)
    captured: dict[str, list[str]] = {}

    def fake_build(constituent_tickers: list[str], *a: object, **k: object) -> pd.DataFrame:
        captured["tickers"] = constituent_tickers
        return replicated_prices

    monkeypatch.setattr(data_sources, "fetch_stock_prices", lambda *a, **k: stock_prices)
    monkeypatch.setattr(
        data_sources,
        "fetch_index_prices",
        lambda *a, **k: (_ for _ in ()).throw(DataSourceError("Stooq bloqueado")),
    )
    monkeypatch.setattr(data_sources, "build_replicated_index_prices", fake_build)

    case = CaseParameters(
        company_name="Aguas Andinas S.A.",
        stock_ticker="AGUAS-A.SN",  # está en el catálogo IPSA
        index_name="Índice de Prueba",
        index_ticker="^TEST",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        frequency="D",
    )
    pipeline.run_beta_case(case)

    assert "AGUAS-A.SN" not in captured["tickers"]
