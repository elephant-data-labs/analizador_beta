"""Parámetros del caso de análisis: empresa, ticker, índice, ventana y frecuencia.

Todo lo que puede cambiar entre corridas vive acá, como un solo objeto explícito
(`CaseParameters`). El resto del código nunca hardcodea "Aguas Andinas" ni
"AGUAS-A.SN": recibe siempre una instancia de esta clase, así que reutilizar el
proyecto para otra empresa es cambiar estos valores, no el código.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

Frequency = Literal["D", "W", "M"]

# De dónde se descarga el precio del índice de mercado (ver
# `pipeline._fetch_market_prices`):
#   "stooq"  — Stooq, con caída automática a un CSV manual en data/raw/ (si
#              existe) y, si no, a un índice replicado desde Yahoo Finance.
#              Es lo que hace falta para ^IPSA, que Yahoo no tiene completo.
#   "yahoo"  — Yahoo Finance directo (misma fuente que la acción), para
#              índices que sí tienen historial completo ahí, como ^GSPC.
# Un CSV manual en data/raw/<ticker>_manual.csv siempre tiene prioridad,
# sea cual sea `index_source` — ver catalog.KNOWN_INDICES.
IndexSource = Literal["stooq", "yahoo"]

FREQUENCY_LABELS: dict[Frequency, str] = {
    "D": "Diaria",
    "W": "Semanal",
    "M": "Mensual",
}

# Ventanas de resample de pandas para cada frecuencia (fin de período).
FREQUENCY_RESAMPLE_RULE: dict[Frequency, str] = {
    "D": "D",
    "W": "W-FRI",
    "M": "ME",
}

PRICE_FIELD_OPTIONS = ("adj_close", "close")


@dataclass(frozen=True)
class CaseParameters:
    """Define un caso de estimación de Beta: qué empresa, contra qué índice,
    en qué ventana, con qué frecuencia y con qué precio."""

    company_name: str
    stock_ticker: str
    index_name: str
    index_ticker: str
    start_date: date
    end_date: date
    frequency: Frequency = "D"
    price_field: str = "adj_close"
    index_source: IndexSource = "stooq"

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha de término.")
        if self.frequency not in FREQUENCY_LABELS:
            raise ValueError(f"Frecuencia no soportada: {self.frequency!r}")
        if self.price_field not in PRICE_FIELD_OPTIONS:
            raise ValueError(f"Campo de precio no soportado: {self.price_field!r}")
        if self.index_source not in ("stooq", "yahoo"):
            raise ValueError(f"Fuente de índice no soportada: {self.index_source!r}")

    @property
    def frequency_label(self) -> str:
        return FREQUENCY_LABELS[self.frequency]


# --- Caso de prueba: Aguas Andinas S.A. -------------------------------------
#
# Ticker verificado manualmente en Yahoo Finance antes de programar (ver
# README, sección "Fuente de datos"): AGUAS-A.SN, Bolsa de Santiago, cotiza en
# CLP, con historial diario continuo desde 2021 y ajustes por dividendos en la
# columna Adj Close (no se observaron splits en el período).
#
# Índice: S&P/CLX IPSA. Es el índice bursátil de referencia del mercado
# accionario chileno (las acciones más grandes y líquidas de la Bolsa de
# Santiago), el proxy estándar de "mercado" para calcular Beta de una acción
# chilena en la literatura y en la práctica local. Yahoo Finance no entrega
# una serie histórica utilizable para "^IPSA" (solo un punto de dato), así que
# se usa Stooq como fuente consistente para el índice — ver README.
DEFAULT_CASE = CaseParameters(
    company_name="Aguas Andinas S.A.",
    stock_ticker="AGUAS-A.SN",
    index_name="S&P/CLX IPSA",
    index_ticker="^IPSA",
    start_date=date(2021, 1, 1),
    end_date=date(2025, 12, 31),
    frequency="D",
    price_field="adj_close",
)
