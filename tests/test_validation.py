from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beta_analyzer.regression import run_ols_regression
from beta_analyzer.returns import ReturnSeries
from beta_analyzer.validation import validate_beta


def _synthetic_returns(n: int = 300, seed: int = 11) -> ReturnSeries:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.011, size=n)
    stock = 0.0002 + 0.85 * market + rng.normal(0.0, 0.005, size=n)
    index = pd.date_range("2021-01-01", periods=n, freq="B")
    return ReturnSeries(
        stock=pd.Series(stock, index=index),
        market=pd.Series(market, index=index),
        frequency="D",
        start_date=index.min(),
        end_date=index.max(),
    )


def test_ols_and_cov_var_beta_match() -> None:
    returns = _synthetic_returns()
    reg = run_ols_regression(returns)
    val = validate_beta(reg.beta, returns)

    assert val.beta_ols == pytest.approx(reg.beta)
    assert val.beta_cov_var == pytest.approx(reg.beta, rel=1e-9)
    assert abs(val.difference) < 1e-9
    assert val.is_consistent is True


def test_validation_flags_inconsistency_when_forced() -> None:
    from beta_analyzer.validation import ValidationResult

    forced = ValidationResult(beta_ols=1.20, beta_cov_var=0.90)
    assert forced.is_consistent is False
    assert forced.difference == pytest.approx(0.30)
