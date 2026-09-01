"""Detección de retornos atípicos (outliers) a partir de los residuos de la
regresión de mercado ya calculada en `regression.py`.

Origen: adaptación del proyecto final del Bootcamp de Carlos Alaniz
("Detector de movimientos anómalos", Tkinter + S&P 500 fijo). Acá se
generaliza a cualquier caso — el índice de mercado es el que defina
`CaseParameters` (por defecto S&P/CLX IPSA, no un mercado fijo) — y se separa
la lógica de la interfaz, igual que el resto del proyecto.

Metodología (idéntica a la del Bootcamp):

    residuo_t = R_i,t - (alpha + beta * R_m,t)
    z_t       = (residuo_t - media(residuo)) / desviación_estándar(residuo)

Un día se marca como atípico si |z_t| supera el umbral (por defecto 2.5
desviaciones estándar, ajustable en la interfaz). El residuo es la parte del
retorno del día que el modelo de mercado NO explica: aísla movimientos
propios de la acción, independientes de lo que hizo el índice ese mismo día.

Determinístico y sin IA: solo aritmética sobre lo que ya calculó
`regression.py`, igual que el resto del pipeline (ver README, "Principio de
diseño: sin IA").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regression import RegressionResult
from .returns import ReturnSeries

DEFAULT_Z_THRESHOLD = 2.5


@dataclass(frozen=True)
class OutlierResult:
    threshold: float
    table: pd.DataFrame
    """Todas las observaciones de la ventana, con columnas retorno_activo,
    retorno_mercado, residuo, z_score y z_score_abs (no solo los atípicos)."""
    outliers: pd.DataFrame
    """Subconjunto de `table` que supera el umbral, ordenado por |z| descendente."""
    n_outliers: int
    n_positive: int
    n_negative: int
    residual_mean: float
    residual_std: float

    @property
    def n_obs(self) -> int:
        return len(self.table)


def detect_outliers(
    returns: ReturnSeries,
    regression: RegressionResult,
    threshold: float = DEFAULT_Z_THRESHOLD,
) -> OutlierResult:
    """Calcula el residuo de cada observación frente a la recta de regresión
    ya estimada y marca como atípicas las que superan `threshold` desviaciones
    estándar de z-score.
    """
    if threshold <= 0:
        raise ValueError("El umbral debe ser un número positivo de desviaciones estándar.")

    df = pd.DataFrame(
        {
            "retorno_activo": returns.stock,
            "retorno_mercado": returns.market,
        }
    )

    predicho = regression.alpha + regression.beta * df["retorno_mercado"]
    df["residuo"] = df["retorno_activo"] - predicho

    media = float(df["residuo"].mean())
    desv = float(df["residuo"].std(ddof=1))
    if not desv or np.isnan(desv):
        raise ValueError(
            "La desviación estándar de los residuos es cero o indefinida; no se "
            "puede calcular el z-score (revise que haya suficientes observaciones)."
        )

    df["z_score"] = (df["residuo"] - media) / desv
    df["z_score_abs"] = df["z_score"].abs()

    outliers = df[df["z_score_abs"] > threshold].sort_values(by="z_score_abs", ascending=False)

    n_pos = int((outliers["residuo"] > 0).sum())
    n_neg = int((outliers["residuo"] <= 0).sum())

    return OutlierResult(
        threshold=threshold,
        table=df,
        outliers=outliers,
        n_outliers=len(outliers),
        n_positive=n_pos,
        n_negative=n_neg,
        residual_mean=media,
        residual_std=desv,
    )
