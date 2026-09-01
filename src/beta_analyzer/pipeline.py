"""Orquesta el flujo completo de la primera versión, en el orden pedido:

    Precio acción
          +
    Precio índice
          v
    Retornos (alineados por fecha)
          v
    Regresión OLS  ->  Beta, alpha, estadísticos
          v
    Validación Cov/Var
          (gráfico se arma aparte, en la capa de interfaz)

Una sola función pública, `run_beta_case`, para que `app.py` (o un script /
notebook / test) no tenga que repetir el orden de llamadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data_sources
from .catalog import KNOWN_COMPANIES
from .config import CaseParameters
from .data_sources import DataSourceError
from .regression import RegressionResult, run_ols_regression
from .returns import ReturnSeries, compute_returns
from .validation import ValidationResult, validate_beta


@dataclass(frozen=True)
class BetaCaseResult:
    case: CaseParameters
    stock_prices: pd.DataFrame
    market_prices: pd.DataFrame
    index_source_label: str
    returns: ReturnSeries
    regression: RegressionResult
    validation: ValidationResult


def _default_manual_csv_path(ticker: str) -> Path:
    """Ruta convencional del CSV manual de respaldo para el ticker de un
    índice (ver README, sección "Fuente de datos"): `data/raw/<ticker>_manual.csv`,
    relativa a la raíz del proyecto. Si el archivo existe ahí, `_fetch_market_prices`
    lo usa automáticamente — no hace falta pasar `manual_index_csv` a mano."""
    project_root = Path(__file__).resolve().parent.parent.parent
    slug = ticker.lstrip("^").lower()
    return project_root / "data" / "raw" / f"{slug}_manual.csv"


def _fetch_market_prices(
    case: CaseParameters, allow_replicated_index_fallback: bool
) -> tuple[pd.DataFrame, str]:
    """Precio del índice + una etiqueta explicando de dónde salió realmente.

    Orden de prioridad:
    1. CSV manual en `data/raw/<ticker>_manual.csv`, si existe (ver
       `_default_manual_csv_path`) — descargado a mano una vez porque Stooq
       bloquea las descargas automatizadas (ver README); es el dato real del
       índice, así que se prefiere por sobre reintentar Stooq, Yahoo o
       aproximar con el índice replicado. Aplica sea cual sea `index_source`.
    2. Si `case.index_source == "yahoo"` (p. ej. S&P 500 / `^GSPC`, ver
       `catalog.KNOWN_INDICES`): Yahoo Finance directo, la misma fuente que
       ya se usa para la acción — no pasa por Stooq ni por el índice
       replicado, porque para estos índices Yahoo sí tiene historial
       completo.
    3. Si `case.index_source == "stooq"` (default, y lo que hace falta para
       ^IPSA): Stooq.
    4. Si (3) falla y el respaldo está habilitado, un índice replicado con
       los componentes del IPSA verificados en `catalog.py` (excluyendo la
       acción en análisis, para no comparar la acción contra un mercado que
       la incluye a ella misma) — ver `data_sources.build_replicated_index_prices`.
       Este respaldo asume componentes chilenas: solo tiene sentido para un
       índice tipo IPSA, no para `index_source == "yahoo"`.

    La etiqueta que se devuelve queda visible en la interfaz: nunca debe
    quedar ambiguo qué fuente se usó de verdad.
    """
    manual_path = _default_manual_csv_path(case.index_ticker)
    if manual_path.exists():
        market_prices = data_sources.load_manual_csv(manual_path)
        return (
            market_prices,
            f"{case.index_name} — CSV descargado a mano ({manual_path.relative_to(manual_path.parents[2])}, ver README)",
        )

    if case.index_source == "yahoo":
        market_prices = data_sources.fetch_stock_prices(
            case.index_ticker, case.start_date, case.end_date
        )
        return market_prices, f"{case.index_name} vía Yahoo Finance (ticker {case.index_ticker})"

    try:
        market_prices = data_sources.fetch_index_prices(
            case.index_ticker, case.start_date, case.end_date
        )
        return market_prices, f"{case.index_name} vía Stooq (ticker {case.index_ticker})"
    except DataSourceError as stooq_error:
        if not allow_replicated_index_fallback:
            raise

        constituents = [c.ticker for c in KNOWN_COMPANIES if c.ticker != case.stock_ticker]
        try:
            market_prices = data_sources.build_replicated_index_prices(
                constituents, case.start_date, case.end_date
            )
        except DataSourceError as fallback_error:
            raise DataSourceError(
                f"Stooq falló ({stooq_error}) y el índice replicado de respaldo "
                f"también falló ({fallback_error})."
            ) from fallback_error

        return (
            market_prices,
            f"Índice replicado — promedio equiponderado de {len(constituents)} "
            "componentes del IPSA (Yahoo Finance), porque Stooq no respondió "
            f"({stooq_error})",
        )


def run_beta_case(
    case: CaseParameters,
    manual_stock_csv: Path | None = None,
    manual_index_csv: Path | None = None,
    allow_replicated_index_fallback: bool = True,
) -> BetaCaseResult:
    """Corre el pipeline completo para un `CaseParameters` dado.

    Si se entrega `manual_stock_csv` / `manual_index_csv`, se usa ese archivo
    en vez de llamar a la fuente automática. Si no se entrega
    `manual_index_csv` y Stooq falla, con `allow_replicated_index_fallback`
    (default True) se arma automáticamente un índice replicado desde Yahoo
    Finance en vez de detener el cálculo — ver `_fetch_market_prices`.
    """
    stock_prices = (
        data_sources.load_manual_csv(manual_stock_csv)
        if manual_stock_csv is not None
        else data_sources.fetch_stock_prices(case.stock_ticker, case.start_date, case.end_date)
    )

    if manual_index_csv is not None:
        market_prices = data_sources.load_manual_csv(manual_index_csv)
        index_source_label = f"CSV manual ({manual_index_csv.name})"
    else:
        market_prices, index_source_label = _fetch_market_prices(
            case, allow_replicated_index_fallback
        )

    window_returns = compute_returns(stock_prices, market_prices, case)
    regression_result = run_ols_regression(window_returns)
    validation_result = validate_beta(regression_result.beta, window_returns)

    return BetaCaseResult(
        case=case,
        stock_prices=stock_prices,
        market_prices=market_prices,
        index_source_label=index_source_label,
        returns=window_returns,
        regression=regression_result,
        validation=validation_result,
    )
