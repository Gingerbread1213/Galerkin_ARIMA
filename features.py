
def build_features(lags, seasonal_lags, include_sq_lags: bool = True, include_sq_seasonal: bool = False):
    """
    Build a feature vector:
      [1,
       lags...,
       (lags)^2... if include_sq_lags,
       seasonal_lags...,
       (seasonal_lags)^2... if include_sq_seasonal]
    """
    feats = [1.0]
    # non-seasonal lags
    feats += lags
    if include_sq_lags:
        feats += [x * x for x in lags]
    # seasonal lags
    feats += seasonal_lags
    if include_sq_seasonal:
        feats += [x * x for x in seasonal_lags]
    return feats

# Basis function definitions
def linear_basis(lags, seasonal_lags):
    """Linear basis: [1, lags, seasonal_lags]"""
    return [1.0] + lags + seasonal_lags

def quadratic_basis(lags, seasonal_lags):
    """Quadratic basis: [1, lags, lags^2, seasonal_lags, seasonal_lags^2]"""
    feats = [1.0] + lags
    feats += [x * x for x in lags]
    feats += seasonal_lags
    feats += [x * x for x in seasonal_lags]
    return feats

def cubic_basis(lags, seasonal_lags):
    """Cubic basis: [1, lags, lags^2, lags^3, seasonal_lags, seasonal_lags^2, seasonal_lags^3]"""
    feats = [1.0] + lags
    feats += [x * x for x in lags]
    feats += [x * x * x for x in lags]
    feats += seasonal_lags
    feats += [x * x for x in seasonal_lags]
    feats += [x * x * x for x in seasonal_lags]
    return feats

def trigonometric_basis(lags, seasonal_lags):
    """Trigonometric basis: [1, lags, sin(lags), cos(lags), seasonal_lags, sin(seasonal_lags), cos(seasonal_lags)]"""
    import numpy as np
    feats = [1.0] + lags
    feats += [np.sin(x) for x in lags]
    feats += [np.cos(x) for x in lags]
    feats += seasonal_lags
    feats += [np.sin(x) for x in seasonal_lags]
    feats += [np.cos(x) for x in seasonal_lags]
    return feats

def exponential_basis(lags, seasonal_lags):
    """Exponential basis: [1, lags, exp(lags), seasonal_lags, exp(seasonal_lags)]"""
    import numpy as np
    feats = [1.0] + lags
    feats += [np.exp(x) for x in lags]
    feats += seasonal_lags
    feats += [np.exp(x) for x in seasonal_lags]
    return feats

def log_basis(lags, seasonal_lags):
    """Logarithmic basis: [1, lags, log(1+|lags|), seasonal_lags, log(1+|seasonal_lags|)]"""
    import numpy as np
    feats = [1.0] + lags
    feats += [np.log1p(abs(x)) for x in lags]
    feats += seasonal_lags
    feats += [np.log1p(abs(x)) for x in seasonal_lags]
    return feats

def sigmoid_basis(lags, seasonal_lags):
    """Sigmoid basis: [1, lags, sigmoid(lags), seasonal_lags, sigmoid(seasonal_lags)]"""
    import numpy as np
    feats = [1.0] + lags
    feats += [1.0 / (1.0 + np.exp(-x)) for x in lags]
    feats += seasonal_lags
    feats += [1.0 / (1.0 + np.exp(-x)) for x in seasonal_lags]
    return feats

def abs_basis(lags, seasonal_lags):
    """Absolute value basis: [1, lags, |lags|, seasonal_lags, |seasonal_lags|]"""
    feats = [1.0] + lags
    feats += [abs(x) for x in lags]
    feats += seasonal_lags
    feats += [abs(x) for x in seasonal_lags]
    return feats

# Dictionary of available basis functions
BASIS_FUNCTIONS = {
    'linear': linear_basis,
    'quadratic': quadratic_basis,
    'cubic': cubic_basis,
    'trigonometric': trigonometric_basis,
    'exponential': exponential_basis,
    'log': log_basis,
    'sigmoid': sigmoid_basis,
    'abs': abs_basis
}

def combine_basis_functions(basis_list):
    """
    Combine multiple basis functions into a single function.
    
    Args:
        basis_list: List of basis function names or functions
        
    Returns:
        Combined basis function
    """
    def combined_basis(lags, seasonal_lags):
        feats = []
        for basis in basis_list:
            if isinstance(basis, str):
                if basis in BASIS_FUNCTIONS:
                    feats.extend(BASIS_FUNCTIONS[basis](lags, seasonal_lags))
                else:
                    raise ValueError(f"Unknown basis function: {basis}")
            else:
                # Assume it's a callable function
                feats.extend(basis(lags, seasonal_lags))
        return feats
    
    return combined_basis