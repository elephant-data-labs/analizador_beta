from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beta_analyzer.regression import (
    beta_interpretation,
    ci_interpretation,
    p_value_interpretation,
    r_squared_interpretation,
    run_ols_regression,
    significance_conclusion,
)
from beta_analyzer.returns import ReturnSeries

TRUE_ALPHA = 0.0004
TRUE_BETA = 1.35


def _synthetic_returns(n: int = 250, seed: int = 42) -> ReturnSeries:
    """Retornos sintéticos con relación lineal conocida (semilla fija, 100%
    reproducible) para poder verificar que la regresión recupera Beta."""
    rng = np.random.default_rng(seed)
    market = rng.normal(loc=0.0005, scale=0.012, size=n)
    noise = rng.normal(loc=0.0, scale=0.006, size=n)
    stock = TRUE_ALPHA + TRUE_BETA * market + noise

    index = pd.date_range("2021-01-01", periods=n, freq="B")
    return ReturnSeries(
        stock=pd.Series(stock, index=index),
        market=pd.Series(market, index=index),
        frequency="D",
        start_date=index.min(),
        end_date=index.max(),
    )


def test_ols_recovers_known_beta_within_sampling_error() -> None:
    result = run_ols_regression(_synthetic_returns())

    # Con 250 observaciones y ese nivel de ruido, el error estándar de Beta
    # es chico: exigir que quede dentro de +/- 0.15 del Beta verdadero es un
    # margen holgado que no debería fallar por variaciones de semilla/librería.
    assert result.beta == pytest.approx(TRUE_BETA, abs=0.15)
    assert result.alpha == pytest.approx(TRUE_ALPHA, abs=0.01)
    assert result.n_obs == 250
    assert 0.0 <= result.r_squared <= 1.0
    assert result.beta_ci_low < result.beta < result.beta_ci_high
    assert result.beta_p_value < 0.05  # con esa señal, debería ser significativo
    assert result.is_significant_5pct is True


def test_regression_result_matches_manual_ols_formula() -> None:
    """Beta de la regresión simple = Cov(x,y) / Var(x); se recalcula acá con
    numpy, aparte de statsmodels, como chequeo cruzado independiente."""
    returns = _synthetic_returns(seed=7)
    result = run_ols_regression(returns)

    x = returns.market.to_numpy()
    y = returns.stock.to_numpy()
    beta_manual = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)

    assert result.beta == pytest.approx(beta_manual, rel=1e-9)


def test_significance_conclusion_text_matches_p_value() -> None:
    returns = _synthetic_returns()
    result = run_ols_regression(returns)
    text = significance_conclusion(result)

    assert "Beta = " in text
    assert "p-value = " in text
    if result.beta_p_value < 0.05:
        assert "estadísticamente significativo al 5%" in text
    else:
        assert "No existe evidencia estadística suficiente" in text


@pytest.mark.parametrize(
    "beta,expected_fragment",
    [
        (1.5, "bastante más volátil"),
        (1.15, "algo más volátil"),
        (0.85, "parecida al mercado"),
        (0.5, "bastante menos volátil"),
        (0.0, "cercano a 0"),
        (-0.8, "dirección contraria"),
    ],
)
def test_beta_interpretation_picks_the_right_bucket(beta: float, expected_fragment: str) -> None:
    text = beta_interpretation(beta)
    assert f"{beta:.2f}" in text
    assert expected_fragment in text


@pytest.mark.parametrize(
    "r2,expected_fragment",
    [(0.75, "alto"), (0.45, "moderado"), (0.1, "bajo")],
)
def test_r_squared_interpretation_picks_the_right_bucket(r2: float, expected_fragment: str) -> None:
    text = r_squared_interpretation(r2)
    assert f"{r2 * 100:.1f}%" in text
    assert expected_fragment in text


def test_p_value_interpretation_matches_significance() -> None:
    returns = _synthetic_returns()
    result = run_ols_regression(returns)
    text = p_value_interpretation(result)

    assert f"{result.beta_p_value:.3f}".replace(".", ",") in text
    if result.is_significant_5pct:
        assert "estadísticamente significativa" in text
    else:
        assert "no hay evidencia estadística suficiente" in text


def test_ci_interpretation_reports_both_bounds() -> None:
    returns = _synthetic_returns()
    result = run_ols_regression(returns)
    text = ci_interpretation(result)

    assert f"{result.beta_ci_low:.2f}".replace(".", ",") in text
    assert f"{result.beta_ci_high:.2f}".replace(".", ",") in text
    assert "95%" in text
