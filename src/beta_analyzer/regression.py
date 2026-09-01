"""Regresión OLS determinística: R_i = alpha + beta * R_m + epsilon.

Usa `statsmodels` para obtener el set completo de estadísticos que exige el
trabajo académico (alpha, beta, error estándar, t-stat, p-value, R²,
intervalo de confianza al 95%, número de observaciones). No hay ningún paso
de este módulo que dependa de un modelo de lenguaje: todo es cálculo
numérico reproducible con la misma librería y la misma versión fijada en
requirements.txt.
"""

from __future__ import annotations

from dataclasses import dataclass

import statsmodels.api as sm

from .returns import ReturnSeries


@dataclass(frozen=True)
class RegressionResult:
    alpha: float
    beta: float
    beta_std_error: float
    beta_t_stat: float
    beta_p_value: float
    r_squared: float
    n_obs: int
    beta_ci_low: float
    beta_ci_high: float
    alpha_p_value: float
    frequency: str
    start_date: str
    end_date: str

    @property
    def is_significant_5pct(self) -> bool:
        return self.beta_p_value < 0.05


def run_ols_regression(returns: ReturnSeries, confidence: float = 0.95) -> RegressionResult:
    """Estima alpha y beta por mínimos cuadrados ordinarios.

    R_i,t = alpha + beta * R_m,t + epsilon_t
    """
    y = returns.stock
    # Se fuerza el nombre de columna a 'market' explícitamente: no depender de
    # que la Series venga con ese nombre (statsmodels usa 'x1' si no lo tiene).
    x = sm.add_constant(returns.market.rename("market"))  # 'const' = alpha, 'market' = beta

    model = sm.OLS(y, x, missing="drop")
    fitted = model.fit()

    alpha_est, beta_est = fitted.params["const"], fitted.params["market"]
    alpha_p, beta_p = fitted.pvalues["const"], fitted.pvalues["market"]
    beta_se = fitted.bse["market"]
    beta_t = fitted.tvalues["market"]

    alpha_level = 1 - confidence
    conf_int = fitted.conf_int(alpha=alpha_level)
    beta_ci_low, beta_ci_high = conf_int.loc["market", 0], conf_int.loc["market", 1]

    return RegressionResult(
        alpha=float(alpha_est),
        beta=float(beta_est),
        beta_std_error=float(beta_se),
        beta_t_stat=float(beta_t),
        beta_p_value=float(beta_p),
        r_squared=float(fitted.rsquared),
        n_obs=int(fitted.nobs),
        beta_ci_low=float(beta_ci_low),
        beta_ci_high=float(beta_ci_high),
        alpha_p_value=float(alpha_p),
        frequency=returns.frequency,
        start_date=str(returns.start_date.date()),
        end_date=str(returns.end_date.date()),
    )


def significance_conclusion(result: RegressionResult) -> str:
    """Conclusión determinística en español, basada solo en el p-value (no en IA)."""
    beta_txt = f"{result.beta:.2f}".replace(".", ",")
    p_txt = f"{result.beta_p_value:.3f}".replace(".", ",")
    if result.is_significant_5pct:
        return (
            f"Beta = {beta_txt}\n"
            f"p-value = {p_txt}\n"
            "Conclusión:\n"
            "Beta estadísticamente significativo al 5%."
        )
    return (
        f"Beta = {beta_txt}\n"
        f"p-value = {p_txt}\n"
        "Conclusión:\n"
        "No existe evidencia estadística suficiente para afirmar\n"
        "que el Beta sea distinto de cero al 5%."
    )


def beta_interpretation(beta: float) -> str:
    """Explicación en lenguaje simple del valor de Beta, por tramos fijos.

    Los umbrales son números fijos definidos en el código (no hay ningún
    modelo de lenguaje generando este texto): es una tabla de casos, igual
    que `significance_conclusion` de arriba.
    """
    if beta > 1.3:
        return (
            f"Beta = {beta:.2f}. La acción es bastante más volátil que el "
            "mercado: cuando el mercado sube o baja 1%, esta acción tiende a "
            f"moverse cerca de {beta:.1f}% en la misma dirección, en promedio."
        )
    if beta > 1.0:
        return (
            f"Beta = {beta:.2f}. La acción es algo más volátil que el "
            "mercado (se mueve un poco más que el índice, en la misma "
            "dirección)."
        )
    if beta > 0.7:
        return (
            f"Beta = {beta:.2f}. La acción se mueve de forma parecida al "
            "mercado, aunque algo más moderada."
        )
    if beta > 0.3:
        return (
            f"Beta = {beta:.2f}. La acción es bastante menos volátil que el "
            "mercado: se mueve en la misma dirección, pero con menos fuerza."
        )
    if beta > -0.3:
        return (
            f"Beta = {beta:.2f}, cercano a 0. El movimiento del mercado casi "
            "no explica el movimiento de esta acción (poca relación lineal "
            "entre ambos)."
        )
    return (
        f"Beta = {beta:.2f}, negativo. La acción tiende a moverse en "
        "dirección contraria al mercado, en promedio."
    )


def r_squared_interpretation(r_squared: float) -> str:
    """Explicación en lenguaje simple del R² (sin IA: tramos fijos)."""
    pct = r_squared * 100
    if r_squared >= 0.6:
        nivel = "alto"
        detalle = "el modelo explica una parte importante del movimiento de la acción."
    elif r_squared >= 0.3:
        nivel = "moderado"
        detalle = (
            "el mercado explica una parte del movimiento de la acción, pero "
            "hay bastante que queda sin explicar (otros factores propios de "
            "la empresa)."
        )
    else:
        nivel = "bajo"
        detalle = (
            "el mercado explica solo una parte pequeña del movimiento de la "
            "acción; la mayoría se debe a otros factores no capturados por "
            "este modelo."
        )
    return f"R² = {pct:.1f}% ({nivel}): {detalle}"


def p_value_interpretation(result: RegressionResult) -> str:
    """Explicación en lenguaje simple del p-value / significancia (sin IA)."""
    p_txt = f"{result.beta_p_value:.3f}".replace(".", ",")
    if result.is_significant_5pct:
        return (
            f"p-value = {p_txt} (menor a 0,05). Esto significa que la "
            "relación entre la acción y el mercado (el Beta estimado) es "
            "estadísticamente significativa: es poco probable que un Beta "
            "así se deba solo al azar."
        )
    return (
        f"p-value = {p_txt} (mayor o igual a 0,05). Esto significa que, con "
        "los datos disponibles, no hay evidencia estadística suficiente "
        "para descartar que el verdadero Beta sea cero (podría no existir "
        "una relación real con el mercado)."
    )


def ci_interpretation(result: RegressionResult) -> str:
    """Explicación en lenguaje simple del intervalo de confianza al 95% (sin IA)."""
    low_txt = f"{result.beta_ci_low:.2f}".replace(".", ",")
    high_txt = f"{result.beta_ci_high:.2f}".replace(".", ",")
    return (
        f"Intervalo de confianza 95%: [{low_txt} , {high_txt}]. Si se "
        "repitiera este análisis muchas veces con datos similares, en "
        "aproximadamente el 95% de los casos el verdadero Beta caería "
        "dentro de este rango."
    )
