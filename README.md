# Galerkin-ARIMA

Galerkin-ARIMA (GARIMA) and Galerkin-SARIMA (GARSARIMA): time series models that replace fixed linear AR/MA terms with flexible basis expansions. Estimation is two-stage least squares (Stage 1: AR projection on value lags; Stage 2: MA projection on residual lags), with optional ridge regularization.

## Installation

```bash
pip install -e .
```

Or from PyPI (when published):

```bash
pip install galerkin-arima
```

## Quick Start

```python
from galerkin_arima import GalerkinSARIMA
import numpy as np

# Sample data
y = np.random.randn(200).cumsum()

# Fit model
model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
model.fit(y)

# Forecast
forecast = model.forecast(steps=10)
print(forecast)

# Rolling prediction intervals (PI) and confidence intervals (CI)
roll = model.rolling_prediction_intervals(y, n_forecast=20, method='residual', alpha=0.05)
print(roll['forecasts'], roll['pi_lower'], roll['pi_upper'])
```

## Features

- **Galerkin-SARIMA (GARSARIMA)**: Basis-expanded ARMA with customizable basis functions
- **Ridge regression**: Optional regularization
- **Statistical inference**: Block bootstrap, prediction intervals, factor significance tests
- **Model selection**: BIC-based order selection
- **Rolling PI/CI**: Per-step prediction and confidence intervals

## API Overview

| Class / Function | Description |
|------------------|-------------|
| `GalerkinSARIMA` | Main model (GARSARIMA): fit, forecast, prediction_intervals, rolling_prediction_intervals |
| `SummaryResults` | Regression-style summary (print after `model.summary(endog)`) |
| `select_model_bic` | BIC-based model order selection |
| `build_features` | Default feature builder |
| `BASIS_FUNCTIONS` | Dict of basis functions: linear, quadratic, cubic, trigonometric, etc. |

## Examples

| Notebook | Description |
|----------|-------------|
| `examples/notebooks/synthetic/` | Synthetic data experiments, quick start |
| `examples/notebooks/gdp/` | GDP forecasting |
| `examples/notebooks/hadcrut5/` | Temperature (HadCRUT5) forecasting |
| `examples/notebooks/sp500/` | SP500 forecasting |

## Basis Functions

Use `basis_functions` to customize the expansion:

```python
model = GalerkinSARIMA(
    order=(2, 0, 2),
    seasonal_order=(0, 0, 0, 12),
    basis_functions='quadratic'  # or 'linear', 'cubic', ['quadratic','sigmoid'], etc.
)
```

## Citation

If you use this package in research, please cite the relevant paper.

## License

MIT License
