"""Validación independiente del Beta: Cov(Ri,Rm) / Var(Rm) vs. la pendiente OLS.

Ambos métodos deben coincidir salvo error de redondeo de punto flotante: son
la misma fórmula vista desde dos ángulos (la pendiente de una regresión
simple es, por definición, Cov(X,Y)/Var(X)). Si difieren en algo no
despreciable, es señal de un error de alineación de datos, no de que "un
método sea mejor que el otro".
"""

from __future__ import annotations

from dataclasses import dataclass

from .returns import ReturnSeries

# Diferencia máxima tolerada antes de considerar que algo no cuadra.
# Elegido varios órdenes de magnitud por sobre el ruido de punto flotante
# (~1e-12 típico) para no disparar alertas falsas por redondeo.
DIFFERENCE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ValidationResult:
    beta_ols: float
    beta_cov_var: float

    @property
    def difference(self) -> float:
        return self.beta_ols - self.beta_cov_var

    @property
    def is_consistent(self) -> bool:
        return abs(self.difference) < DIFFERENCE_TOLERANCE


def beta_from_covariance(returns: ReturnSeries) -> float:
    """Beta = Cov(Ri, Rm) / Var(Rm), con la misma convención de grados de
    libertad (ddof=1) que usa pandas por defecto — consistente con la
    varianza muestral que implícitamente usa la regresión OLS.
    """
    covariance_matrix = returns.stock.cov(returns.market)
    market_variance = returns.market.var(ddof=1)
    return float(covariance_matrix / market_variance)


def validate_beta(beta_ols: float, returns: ReturnSeries) -> ValidationResult:
    beta_cov_var = beta_from_covariance(returns)
    return ValidationResult(beta_ols=beta_ols, beta_cov_var=beta_cov_var)
