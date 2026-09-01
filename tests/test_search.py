from __future__ import annotations

import pytest

from beta_analyzer.data_sources import DataSourceError, search_tickers


class _FakeSearch:
    """Reemplaza yfinance.Search en los tests: mismo `.quotes`, sin red.

    Las respuestas usadas acá son las que efectivamente devolvió Yahoo
    Finance al probar el buscador a mano (ver README) para 'banco de chile'
    y 'asdfqwerzxcv' — se fijan como datos de prueba en vez de generarlas,
    para no depender de la red al correr los tests.
    """

    def __init__(self, query: str, **kwargs: object) -> None:
        self.query = query
        if query == "banco de chile":
            self.quotes = [
                {"symbol": "BCH", "shortname": "Banco De Chile", "exchDisp": "NYSE", "quoteType": "EQUITY"},
                {
                    "symbol": "CHILE.SN",
                    "shortname": "BANCO DE CHILE",
                    "exchDisp": "Santiago Stock Exchange",
                    "quoteType": "EQUITY",
                },
                {
                    "symbol": "CHILECO.CL",
                    "shortname": "BANCO DE CHILE",
                    "exchDisp": "BVC",
                    "quoteType": "EQUITY",
                },
            ]
        elif query == "asdfqwerzxcv":
            self.quotes = []
        elif query == "raise":
            raise RuntimeError("Yahoo no responde")
        else:
            self.quotes = [
                {
                    "symbol": f"{query.upper()}.SN",
                    "shortname": query.upper(),
                    "exchDisp": "Santiago Stock Exchange",
                    "quoteType": "EQUITY",
                }
            ]


@pytest.fixture(autouse=True)
def _patch_yfinance_search(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "Search", _FakeSearch)


def test_empty_query_returns_empty_without_calling_yahoo() -> None:
    assert search_tickers("   ") == []


def test_no_results_returns_empty_list() -> None:
    assert search_tickers("asdfqwerzxcv") == []


def test_santiago_result_is_prioritized_over_other_exchanges() -> None:
    matches = search_tickers("banco de chile")
    assert [m.symbol for m in matches] == ["CHILE.SN", "BCH", "CHILECO.CL"]
    assert matches[0].is_santiago is True
    assert matches[0].label == "CHILE.SN — BANCO DE CHILE (Santiago Stock Exchange)"


def test_search_failure_raises_data_source_error() -> None:
    with pytest.raises(DataSourceError):
        search_tickers("raise")
