"""Tests for GalerkinSARIMA model."""
import numpy as np
import pytest
from galerkin_arima import GalerkinSARIMA


@pytest.fixture
def sample_data():
    """Generate sample time series."""
    np.random.seed(42)
    n = 150
    y = np.cumsum(np.random.randn(n) * 0.3) + np.sin(np.linspace(0, 4 * np.pi, n))
    return y


def test_fit_forecast(sample_data):
    """Test basic fit and forecast."""
    model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1), use_ridge=False)
    model.fit(sample_data)
    pred = model.forecast(steps=5)
    assert pred is not None
    assert len(pred) == 5
    assert np.all(np.isfinite(pred))


def test_rolling_prediction_intervals(sample_data):
    """Test rolling prediction intervals."""
    train, test = sample_data[:100], sample_data[100:120]
    model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
    roll = model.rolling_prediction_intervals(train, 20, actuals=test, method='residual', alpha=0.05)
    assert 'forecasts' in roll
    assert 'pi_lower' in roll
    assert 'pi_upper' in roll
    assert 'ci_lower' in roll
    assert 'ci_upper' in roll
    assert len(roll['forecasts']) == 20


def test_prediction_intervals(sample_data):
    """Test single-step prediction intervals."""
    model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
    result = model.prediction_intervals(sample_data, steps=1, n_bootstrap=50, alpha=0.05)
    assert 'forecast' in result
    assert 'pi' in result
    assert 'ci_mean' in result
    assert len(result['pi']) == 2


def test_bic(sample_data):
    """Test BIC computation."""
    model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
    model.fit(sample_data)
    bic = model.bic(sample_data)
    assert np.isfinite(bic)


def test_summary(sample_data):
    """Test summary generation."""
    model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
    model.fit(sample_data)
    summary = model.summary(sample_data, n_bootstrap=20)
    assert summary is not None
    s = str(summary)
    assert 'GalerkinSARIMA' in s or 'Coefficients' in s


def test_basis_functions(sample_data):
    """Test with custom basis function."""
    model = GalerkinSARIMA(
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 1),
        basis_functions='linear'
    )
    model.fit(sample_data)
    pred = model.forecast(steps=3)
    assert len(pred) == 3
