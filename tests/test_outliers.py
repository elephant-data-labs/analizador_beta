from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from beta_analyzer.outliers import detect_outliers
from beta_analyzer.regression import RegressionResult
from beta_analyzer.returns import ReturnSeries


def _regression(alpha: float = 0.0, beta: float = 1.0) -> RegressionResult:
    """RegressionResult con alpha/beta fijados a mano y el resto de campos
    (no usados por `detect_outliers`) rellenados con valores dummy."""
    return RegressionResult(
        alpha=alpha,
        beta=beta,
        beta_std_error=0.0,
        beta_t_stat=0.0,
        beta_p_value=0.0,
        r_squared=0.0,
        n_obs=0,
        beta_ci_low=0.0,
        beta_ci_high=0.0,
        alpha_p_value=0.0,
        frequency="D",
        start_date="2024-01-01",
        end_date="2024-01-01",
    )


def _returns(stock_extra: list[float], n: int = 10) -> ReturnSeries:
    dates = pd.bdate_range("2024-01-02", periods=n)
    market = pd.Series(0.001, index=dates)  # constante, arbitraria
    stock = market + pd.Series(stock_extra, index=dates)
    return ReturnSeries(
        stock=stock, market=market, frequency="D", start_date=dates[0], end_date=dates[-1]
    )


def test_detect_outliers_flags_the_single_large_residual() -> None:
    """9 días con residuo exactamente 0 y 1 día con residuo +0.05 (alpha=0,
    beta=1 => residuo = stock - market). A mano: media = 0.005, desv.
    estándar (ddof=1) = 0.05*sqrt(0.1) ≈ 0.015811, z del día atípico ≈ 2.846,
    z de los días en cero ≈ -0.316 — solo el primero supera 2.5."""
    returns = _returns([0.0] * 9 + [0.05])
    result = detect_outliers(returns, _regression(), threshold=2.5)

    assert result.n_outliers == 1
    assert result.n_positive == 1
    assert result.n_negative == 0
    assert result.n_obs == 10

    assert result.residual_mean == pytest.approx(0.005)
    assert result.residual_std == pytest.approx(0.05 * math.sqrt(0.1))

    only_outlier = result.outliers.iloc[0]
    assert only_outlier["z_score"] == pytest.approx(2.846, abs=0.01)
    assert result.outliers.index[0] == returns.stock.index[-1]


def test_detect_outliers_sorts_by_absolute_z_score_descending() -> None:
    """Un residuo negativo grande y uno positivo mediano: ambos atípicos,
    pero el de mayor |z| debe quedar primero, sin importar el signo."""
    returns = _returns([0.0] * 7 + [0.03, -0.08, 0.0])
    result = detect_outliers(returns, _regression(), threshold=1.0)

    assert result.n_outliers >= 2
    z_abs = result.outliers["z_score_abs"].to_numpy()
    assert np.all(np.diff(z_abs) <= 0)  # no creciente = orden descendente
    assert result.n_positive == 1
    assert result.n_negative == 1


def test_detect_outliers_rejects_non_positive_threshold() -> None:
    returns = _returns([0.0] * 9 + [0.05])
    with pytest.raises(ValueError):
        detect_outliers(returns, _regression(), threshold=0)
    with pytest.raises(ValueError):
        detect_outliers(returns, _regression(), threshold=-1.0)


def test_detect_outliers_raises_when_all_residuals_are_identical() -> None:
    """Si el residuo es constante (p.ej. la acción replica exactamente el
    índice), la desviación estándar es cero y el z-score queda indefinido."""
    returns = _returns([0.0] * 10)
    with pytest.raises(ValueError):
        detect_outliers(returns, _regression())


def test_detect_outliers_uses_the_case_regression_not_a_refit() -> None:
    """`detect_outliers` no vuelve a estimar alpha/beta: usa los que ya
    calculó `regression.py`. Cambiar beta cambia el residuo esperado."""
    returns = _returns([0.0] * 9 + [0.05])

    with_beta_1 = detect_outliers(returns, _regression(alpha=0.0, beta=1.0), threshold=2.5)
    with_beta_0 = detect_outliers(returns, _regression(alpha=0.0, beta=0.0), threshold=2.5)

    # Con beta=0, el residuo de cada día es simplemente su retorno (stock -
    # alpha), muy distinto del caso beta=1 (stock - market): los conjuntos
    # de outliers no tienen por qué coincidir.
    assert with_beta_1.residual_mean != pytest.approx(with_beta_0.residual_mean)
