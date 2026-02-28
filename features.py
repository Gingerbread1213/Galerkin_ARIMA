"""
Backward compatibility: re-export from galerkin_arima package.
Prefer: from galerkin_arima import build_features, BASIS_FUNCTIONS, ...
"""
try:
    from galerkin_arima import (
        build_features,
        combine_basis_functions,
        BASIS_FUNCTIONS,
        linear_basis,
        quadratic_basis,
        cubic_basis,
        trigonometric_basis,
        exponential_basis,
        log_basis,
        sigmoid_basis,
        abs_basis,
    )
except ImportError:
    import sys
    from pathlib import Path
    _src = Path(__file__).resolve().parent / "src"
    if _src.exists():
        sys.path.insert(0, str(_src))
    from galerkin_arima import (
        build_features,
        combine_basis_functions,
        BASIS_FUNCTIONS,
        linear_basis,
        quadratic_basis,
        cubic_basis,
        trigonometric_basis,
        exponential_basis,
        log_basis,
        sigmoid_basis,
        abs_basis,
    )
