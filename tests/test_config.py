from __future__ import annotations

from datetime import date

import pytest

from beta_analyzer.config import DEFAULT_CASE, CaseParameters


def _kwargs(**overrides: object) -> dict:
    base = dict(
        company_name="Empresa de Prueba",
        stock_ticker="TEST-A.SN",
        index_name="Índice de Prueba",
        index_ticker="^TEST",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
    )
    base.update(overrides)
    return base


def test_default_case_uses_stooq_as_index_source() -> None:
    # Comportamiento histórico preservado: quien no elige nada explícito
    # (scripts/tests viejos, DEFAULT_CASE) sigue yendo por Stooq -> CSV
    # manual -> índice replicado, como antes de que existiera index_source.
    assert DEFAULT_CASE.index_source == "stooq"


def test_case_parameters_defaults_index_source_to_stooq() -> None:
    case = CaseParameters(**_kwargs())
    assert case.index_source == "stooq"


def test_case_parameters_accepts_yahoo_index_source() -> None:
    case = CaseParameters(**_kwargs(index_source="yahoo"))
    assert case.index_source == "yahoo"


def test_case_parameters_rejects_unknown_index_source() -> None:
    with pytest.raises(ValueError):
        CaseParameters(**_kwargs(index_source="bloomberg"))
