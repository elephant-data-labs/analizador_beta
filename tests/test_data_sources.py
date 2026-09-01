from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from beta_analyzer.data_sources import DataSourceError, _exclusive_end_iso, build_replicated_index_prices


def test_exclusive_end_iso_adds_one_day() -> None:
    """Regresión: `date + pd.Timedelta` devuelve un `date` plano (sin
    `.date()`), así que sumar un día debe hacerse con `datetime.timedelta`.
    Este caso reprodujo un AttributeError en producción con fechas elegidas
    desde `st.date_input` (que entrega `datetime.date`, no `Timestamp`)."""
    assert _exclusive_end_iso(date(2025, 12, 31)) == "2026-01-01"
    assert _exclusive_end_iso(date(2024, 2, 28)) == "2024-02-29"  # año bisiesto
    assert _exclusive_end_iso(date(2021, 1, 1)) == "2021-01-02"


def _fake_multi_ticker_download(tickers: list[str]) -> pd.DataFrame:
    """Simula lo que devuelve `yfinance.download` para varios tickers a la
    vez: columnas MultiIndex (campo, ticker)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    series_by_ticker = {
        "AAA.SN": [100.0, 102.0, 101.0, 105.0, 106.0],
        "BBB.SN": [50.0, 49.0, 51.0, 52.0, 52.0],
    }
    data = {}
    for field in ["Open", "High", "Low", "Close", "Adj Close"]:
        for ticker in tickers:
            data[(field, ticker)] = series_by_ticker[ticker]
    for ticker in tickers:
        data[("Volume", ticker)] = [1_000] * 5
    frame = pd.DataFrame(data, index=dates)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_build_replicated_index_prices_is_equal_weighted(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance

    tickers = ["AAA.SN", "BBB.SN"]
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _fake_multi_ticker_download(tickers))

    frame = build_replicated_index_prices(tickers, date(2024, 1, 2), date(2024, 1, 8))

    assert list(frame.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert frame["close"].iloc[0] == pytest.approx(100.0)  # nivel base, sin retorno previo

    # Día 2: AAA +2% (102/100), BBB -2% (49/50) -> promedio equiponderado = 0%.
    assert frame["close"].iloc[1] == pytest.approx(100.0, abs=1e-9)

    # El nivel se acumula (no se resetea) y el "open/high/low/adj_close" del
    # índice replicado replican el "close" (no hay una fuente propia de OHLC
    # para una cartera sintética).
    assert (frame["open"] == frame["close"]).all()
    assert (frame["adj_close"] == frame["close"]).all()


def test_build_replicated_index_prices_empty_tickers_raises() -> None:
    with pytest.raises(DataSourceError):
        build_replicated_index_prices([], date(2024, 1, 1), date(2024, 1, 10))


def test_build_replicated_index_prices_empty_download_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(DataSourceError):
        build_replicated_index_prices(["AAA.SN"], date(2024, 1, 1), date(2024, 1, 10))
