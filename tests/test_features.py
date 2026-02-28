"""Tests for feature building and basis functions."""
import numpy as np
import pytest
from galerkin_arima import (
    build_features,
    combine_basis_functions,
    BASIS_FUNCTIONS,
    linear_basis,
    quadratic_basis,
)


def test_build_features():
    """Test default build_features."""
    lags = [1.0, 2.0]
    seasonal = [3.0]
    feats = build_features(lags, seasonal, include_sq_lags=True, include_sq_seasonal=False)
    assert feats[0] == 1.0
    assert 1.0 in feats and 2.0 in feats
    assert 1.0 in feats and 4.0 in feats  # squares
    assert 3.0 in feats


def test_linear_basis():
    """Test linear basis."""
    feats = linear_basis([1.0, 2.0], [3.0])
    assert feats == [1.0, 1.0, 2.0, 3.0]


def test_quadratic_basis():
    """Test quadratic basis."""
    feats = quadratic_basis([1.0, 2.0], [3.0])
    assert 1.0 in feats
    assert 4.0 in feats  # 2^2
    assert 9.0 in feats  # 3^2


def test_combine_basis():
    """Test combining basis functions."""
    combined = combine_basis_functions(['linear', 'quadratic'])
    feats = combined([1.0], [2.0])
    assert len(feats) > 0


def test_basis_functions_dict():
    """Test BASIS_FUNCTIONS contains expected keys."""
    assert 'linear' in BASIS_FUNCTIONS
    assert 'quadratic' in BASIS_FUNCTIONS
    assert 'cubic' in BASIS_FUNCTIONS
