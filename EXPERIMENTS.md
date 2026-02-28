# Experiments Index

Organized experiments for Galerkin-ARIMA.

## Structure

| Experiment | Location | Data | Description |
|------------|----------|------|-------------|
| **Synthetic** | `examples/notebooks/synthetic/synthetic_experiments.ipynb` | Generated in notebook | Algorithm comparison, rolling PI/CI, multiple synthetic datasets |
| **GDP** | `examples/notebooks/gdp/gdp_forecasting.ipynb` | `GDP.csv`, `FRED.csv` | FRED GDP forecasting |
| **HadCRUT5** | `examples/notebooks/hadcrut5/temperature_forecasting.ipynb` | `HadCRUT5.1.txt` | Global temperature forecasting |
| **SP500** | `examples/notebooks/sp500/sp500_forecasting.ipynb` | `SP500.csv`, `GSPC_5m.csv` | S&P 500 forecasting |

## Results Directories

- `synthetic_comparison_plots/`, `syntheticresults/` – Synthetic experiment outputs
- `GDP_comparison_plots/`, `GDP_results/` – GDP experiment outputs
- `HadCRUT5_comparison_plots/`, `HadCRUT5_results/` – Temperature experiment outputs
- `outputs/` – General outputs

## Running Experiments

1. Install: `pip install -e .`
2. Place data files in project root (see `examples/README.md`)
3. Run Jupyter from project root
4. Open notebooks from `examples/notebooks/<experiment>/`
