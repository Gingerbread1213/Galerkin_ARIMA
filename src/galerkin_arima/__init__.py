"""
Galerkin-ARIMA: Galerkin method-based SARIMA models for time series forecasting.

This package provides implementations of Galerkin-SARIMA models that use
basis function expansions instead of traditional ARIMA parameter estimation.
"""

from ._version import __version__
from .model import (
    GalerkinSARIMA,
    SummaryResults,
    select_model_bic,
)
from .features import (
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

__all__ = [
    "__version__",
    "GalerkinSARIMA",
    "SummaryResults",
    "select_model_bic",
    "build_features",
    "combine_basis_functions",
    "BASIS_FUNCTIONS",
    "linear_basis",
    "quadratic_basis",
    "cubic_basis",
    "trigonometric_basis",
    "exponential_basis",
    "log_basis",
    "sigmoid_basis",
    "abs_basis",
]
