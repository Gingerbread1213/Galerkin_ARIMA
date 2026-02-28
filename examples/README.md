# Galerkin-ARIMA Examples

## Structure

| Directory | Description |
|-----------|-------------|
| `notebooks/synthetic/` | Synthetic data experiments, quick start, algorithm comparison |
| `notebooks/gdp/` | GDP (FRED) forecasting |
| `notebooks/hadcrut5/` | HadCRUT5 temperature forecasting |
| `notebooks/sp500/` | S&P 500 forecasting |
| `scripts/` | Standalone Python scripts |

## Running Notebooks

1. Install the package: `pip install -e .` (from project root)
2. For notebooks that need data (GDP, HadCRUT5, SP500), ensure data files are in the project root:
   - `GDP.csv`, `FRED.csv` for GDP
   - `HadCRUT5.1.txt` for temperature
   - `SP500.csv`, `GSPC_5m.csv` for SP500
3. Run Jupyter from project root: `jupyter notebook` or `jupyter lab`
4. Open notebooks from `examples/notebooks/<dataset>/`

## Quick Start Script

```bash
python examples/scripts/quick_start.py
```
