"""
Backward compatibility: re-export from galerkin_arima package.
Prefer: from galerkin_arima import GalerkinSARIMA
"""
try:
    from galerkin_arima import (
        GalerkinSARIMA,
        SummaryResults,
        select_model_bic,
    )
except ImportError:
    # Fallback when package not installed (e.g. development)
    import sys
    from pathlib import Path
    _src = Path(__file__).resolve().parent / "src"
    if _src.exists():
        sys.path.insert(0, str(_src))
    from galerkin_arima import (
        GalerkinSARIMA,
        SummaryResults,
        select_model_bic,
    )

__all__ = ["GalerkinSARIMA", "SummaryResults", "select_model_bic"]
