from __future__ import annotations

from beta_analyzer.catalog import (
    KNOWN_COMPANIES,
    KNOWN_INDICES,
    OTHER_INDEX_LABEL,
    OTHER_LABEL,
    company_labels,
    find_by_name,
    find_index_by_label,
    index_labels,
)


def test_catalog_has_no_duplicate_names_or_tickers() -> None:
    names = [c.name for c in KNOWN_COMPANIES]
    tickers = [c.ticker for c in KNOWN_COMPANIES]
    assert len(names) == len(set(names))
    assert len(tickers) == len(set(tickers))


def test_all_tickers_use_santiago_suffix() -> None:
    # Todas las entradas del catálogo son acciones chilenas verificadas en
    # Yahoo Finance; ahí siempre llevan el sufijo .SN (ver README).
    assert all(c.ticker.endswith(".SN") for c in KNOWN_COMPANIES)


def test_company_labels_includes_manual_option_at_the_end() -> None:
    labels = company_labels()
    assert labels[-1] == OTHER_LABEL
    assert len(labels) == len(KNOWN_COMPANIES) + 1
    assert len(KNOWN_COMPANIES) == 27  # componentes IPSA verificados (ver catalog.py)


def test_find_by_name_returns_none_for_manual_option() -> None:
    assert find_by_name(OTHER_LABEL) is None
    assert find_by_name("no existe") is None
    assert find_by_name("Aguas Andinas S.A.").ticker == "AGUAS-A.SN"


def test_known_indices_include_ipsa_via_stooq_and_sp500_via_yahoo() -> None:
    by_ticker = {e.ticker: e for e in KNOWN_INDICES}
    assert by_ticker["^IPSA"].source == "stooq"
    assert by_ticker["^GSPC"].source == "yahoo"
    # Fuente admitida por CaseParameters.index_source (config.IndexSource).
    assert all(e.source in ("stooq", "yahoo") for e in KNOWN_INDICES)


def test_index_labels_includes_manual_option_at_the_end() -> None:
    labels = index_labels()
    assert labels[-1] == OTHER_INDEX_LABEL
    assert len(labels) == len(KNOWN_INDICES) + 1


def test_find_index_by_label_returns_none_for_manual_option() -> None:
    assert find_index_by_label(OTHER_INDEX_LABEL) is None
    assert find_index_by_label("no existe") is None
    ipsa = find_index_by_label("S&P/CLX IPSA (^IPSA)")
    assert ipsa is not None
    assert ipsa.ticker == "^IPSA" and ipsa.source == "stooq"
