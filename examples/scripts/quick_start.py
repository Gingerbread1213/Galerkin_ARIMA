"""
Quick start example for Galerkin-ARIMA.
Run from project root: python examples/scripts/quick_start.py
"""
import numpy as np
import matplotlib.pyplot as plt
from galerkin_arima import GalerkinSARIMA

# Generate sample data
np.random.seed(42)
n = 200
y = np.cumsum(np.random.randn(n) * 0.5) + np.sin(np.linspace(0, 4 * np.pi, n))

# Split
train, test = y[:150], y[150:]
n_forecast = len(test)

# Fit and forecast
model = GalerkinSARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1), use_ridge=False)
model.fit(train)

# Rolling forecast with PI and CI
roll = model.rolling_prediction_intervals(train, n_forecast, actuals=test, method='residual', alpha=0.05)

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
t_train = np.arange(len(train))
t_test = np.arange(len(train), len(train) + n_forecast)
ax.plot(t_train, train, 'b-', label='Train', alpha=0.8)
ax.plot(t_test, test, 'g-', label='Actual', alpha=0.8)
ax.plot(t_test, roll['forecasts'], 'r--', label='Forecast', alpha=0.9)
ax.fill_between(t_test, roll['pi_lower'], roll['pi_upper'], alpha=0.2, color='red', label='95% PI')
ax.fill_between(t_test, roll['ci_lower'], roll['ci_upper'], alpha=0.3, color='blue', label='95% CI')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.set_title('Galerkin-ARIMA: Quick Start')
plt.tight_layout()
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / 'quick_start_plot.png'
plt.savefig(_out, dpi=100)
print(f'Saved {_out}')
