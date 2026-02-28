import numpy as np
import scipy.stats
from .features import build_features, combine_basis_functions, BASIS_FUNCTIONS

# ===================== Weighted Ridge Helpers =====================

def _weighted_ridge(X, y, w_diag, lam, standardize=True, jitter=1e-10):
    """
    Solve min ||y - (X beta + b0)||^2 + lam * ||W beta||^2, intercept unpenalized.
    Returns (beta, b0, mu, sig); beta, b0 act on RAW (unstandardized) features.
    """
    n, p = X.shape
    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sig = X.std(axis=0, ddof=0, keepdims=True)
        sig[sig == 0.0] = 1.0
        Xs = (X - mu) / sig
    else:
        Xs = X
        mu = np.zeros((1, p), dtype=float)
        sig = np.ones((1, p), dtype=float)

    w_diag = np.asarray(w_diag, dtype=float)
    if w_diag.shape[0] != p:
        raise ValueError(f"w_diag length {w_diag.shape[0]} must equal n_features {p}.")

    # Augment with intercept column (unpenalized)
    Z = np.column_stack([Xs, np.ones(n)])
    # Build (p+1) x (p+1) ridge matrix with zero penalty on intercept
    H = Z.T @ Z
    P = np.zeros_like(H)
    P[:p, :p] = (w_diag ** 2) * np.eye(p)
    H = H + lam * P + jitter * np.eye(p + 1)
    f = Z.T @ y

    theta = np.linalg.solve(H, f)     # theta = [beta_s ; c]
    beta_s = theta[:p]
    c = float(theta[p])

    # Map back to raw feature space: y ≈ X beta + b0,
    # where beta = beta_s / sig, b0 = c - (mu/sig)·beta_s
    beta = (beta_s / sig.ravel())
    b0 = c - float((mu.ravel() / sig.ravel()) @ beta_s)
    return beta, b0, mu.ravel(), sig.ravel()

# ===================== Time-Independent (existing) =====================

def fit_ar_stage(train, p, P, m, basis_function=None, use_ridge=True,
                 ridge_lambda=1.0, ridge_weights=None, standardize=True, exog=None):
    """
    Fit AR stage with optional exogenous factors.
    
    Args:
        train: Endogenous time series
        p, P, m: AR orders and seasonal period
        basis_function: Basis function for features
        use_ridge: Whether to use ridge regression
        ridge_lambda: Ridge penalty parameter
        ridge_weights: Ridge weight vector
        standardize: Whether to standardize features
        exog: Exogenous factors (N x r array), aligned with train[start_ar:]
    
    Returns:
        (beta, delta, b0, mu, sig), res where:
        - beta: AR basis coefficients
        - delta: Exogenous factor coefficients (None if exog is None)
        - b0: Intercept
        - mu, sig: Standardization parameters
        - res: Residuals
    """
    L = len(train)
    if L == 0:
        return None, np.array([])
    start_ar = max(p, P * m)
    if L <= start_ar:
        return None, np.array([])
    X_ar = []
    y_ar = train[start_ar:]
    for t in range(start_ar, L):
        ns = list(train[t - p : t][::-1]) if p > 0 else []
        ss = [train[t - r * m] for r in range(1, P + 1)] if P > 0 else []
        if basis_function is not None:
            feats = basis_function(ns, ss)
        else:
            # Default to original quadratic basis
            feats = build_features(ns, ss, True, False)
        X_ar.append(feats)
    X_ar = np.array(X_ar, dtype=float)
    y_ar = np.array(y_ar, dtype=float)
    
    # Add exogenous factors if provided
    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        # Align exog with y_ar (should have same length)
        if len(exog) != len(y_ar):
            # If exog is full length, take the appropriate slice
            if len(exog) == L:
                exog = exog[start_ar:]
            else:
                raise ValueError(f"exog length {len(exog)} must match train length {L} or y_ar length {len(y_ar)}")
        X_ar = np.column_stack([X_ar, exog])
    
    # Basic sanity checks
    if X_ar.ndim != 2 or X_ar.shape[0] <= X_ar.shape[1]:
        return None, np.array([])
    if not np.all(np.isfinite(X_ar)) or not np.all(np.isfinite(y_ar)):
        return None, np.array([])

    # Use ridge or OLS based on flag
    try:
        n_features_ar = len(build_features([0]*p if p>0 else [], [0]*P if P>0 else [], True, False)) if basis_function is None else len(basis_function([0]*p if p>0 else [], [0]*P if P>0 else []))
        n_exog = exog.shape[1] if exog is not None else 0
        
        if use_ridge:
            # Weighted ridge regression with intercept
            # Only penalize AR basis, not exogenous factors
            if ridge_weights is None:
                ridge_weights = np.ones(n_features_ar, dtype=float)
            # Extend weights to include exog (unpenalized)
            if n_exog > 0:
                ridge_weights = np.concatenate([ridge_weights, np.zeros(n_exog)])
            
            beta_full, b0, mu, sig = _weighted_ridge(X_ar, y_ar, ridge_weights, ridge_lambda, standardize=standardize)
            if not np.all(np.isfinite(beta_full)) or not np.isfinite(b0):
                return None, np.array([])
            
            # Split coefficients
            beta = beta_full[:n_features_ar]
            delta = beta_full[n_features_ar:] if n_exog > 0 else None
            
            # Compute residuals using full design
            res = y_ar - (X_ar @ beta_full + b0)
            return (beta, delta, b0, mu, sig), res
        else:
            # Standard OLS with intercept
            n, p_full = X_ar.shape
            # Augment with intercept column
            X_aug = np.column_stack([X_ar, np.ones(n)])
            theta, _, _, _ = np.linalg.lstsq(X_aug, y_ar, rcond=None)
            beta_full = theta[:p_full]
            b0 = float(theta[p_full])
            if not np.all(np.isfinite(beta_full)) or not np.isfinite(b0):
                return None, np.array([])
            
            # Split coefficients
            beta = beta_full[:n_features_ar]
            delta = beta_full[n_features_ar:] if n_exog > 0 else None
            
            res = y_ar - (X_ar @ beta_full + b0)
            # Return same format as ridge (with dummy mu, sig)
            mu = np.zeros(p_full, dtype=float)
            sig = np.ones(p_full, dtype=float)
            return (beta, delta, b0, mu, sig), res
    except np.linalg.LinAlgError:
        return None, np.array([])

def forecast_ar(beta_pack, train, p, P, m, basis_function=None, exog_t=None):
    """
    Forecast AR component with optional exogenous factors.
    
    Args:
        beta_pack: (beta_vec, delta, b0, mu, sig) or (beta_vec, b0, mu, sig) for backward compatibility
        train: Training data
        p, P, m: AR orders
        basis_function: Basis function
        exog_t: Exogenous factors at forecast time (1D array of length r)
    
    Returns:
        Forecast value
    """
    if not isinstance(beta_pack, tuple):
        # Handle non-tuple case (shouldn't happen but for safety)
        beta_vec = beta_pack
        delta = None
        b0 = 0.0
    elif len(beta_pack) == 5:
        # Format: (beta_vec, delta, b0, mu, sig)
        beta_vec, delta, b0, _, _ = beta_pack
    elif len(beta_pack) == 4:
        # Format: (beta_vec, b0, mu, sig) - backward compatibility
        beta_vec, b0, _, _ = beta_pack
        delta = None
    else:
        # Fallback
        beta_vec = beta_pack[0] if len(beta_pack) > 0 else beta_pack
        delta = None
        b0 = 0.0
    
    ns = list(train[-p:][::-1]) if p > 0 else []
    ss = [train[-r * m] for r in range(1, P + 1)] if P > 0 else []
    if basis_function is not None:
        feats = basis_function(ns, ss)
    else:
        # Default to original quadratic basis
        feats = build_features(ns, ss, True, False)
    feats = np.asarray(feats, dtype=float)
    
    forecast_val = float(feats @ beta_vec + b0)
    
    # Add exogenous contribution if present
    if delta is not None and exog_t is not None:
        exog_t = np.asarray(exog_t, dtype=float)
        if exog_t.ndim == 0:
            exog_t = exog_t.reshape(1)
        elif exog_t.ndim == 1 and len(exog_t) == len(delta):
            pass  # Already correct shape
        else:
            exog_t = exog_t.ravel()[:len(delta)]
        forecast_val += float(exog_t @ delta)
    
    return forecast_val

def fit_ma_stage(res, q, Q, m, basis_function=None, use_ridge=True,
                 ridge_lambda=1.0, ridge_weights=None, standardize=True):
    L_res = len(res)
    start_ma = max(q, Q * m)
    if L_res <= start_ma:
        return None
    R = []
    y_ma = []
    for t in range(start_ma, L_res):
        nr = list(res[t - q : t][::-1]) if q > 0 else []
        sr = [res[t - r * m] for r in range(1, Q + 1)] if Q > 0 else []
        if basis_function is not None:
            feats = basis_function(nr, sr)
        else:
            # Default to original quadratic basis
            feats = build_features(nr, sr, True, False)
        R.append(feats)
        y_ma.append(res[t])
    R = np.array(R, dtype=float)
    y_ma = np.array(y_ma, dtype=float)

    if R.ndim != 2 or R.shape[0] <= R.shape[1]:
        return None
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(y_ma)):
        return None

    try:
        if use_ridge:
            # Weighted ridge regression with intercept
            if ridge_weights is None:
                ridge_weights = np.ones(R.shape[1], dtype=float)
            alpha_vec, a0, mu, sig = _weighted_ridge(R, y_ma, ridge_weights, ridge_lambda, standardize=standardize)
            if not np.all(np.isfinite(alpha_vec)) or not np.isfinite(a0):
                return None
            return (alpha_vec, a0, mu, sig)
        else:
            # Standard OLS with intercept
            n, p = R.shape
            # Augment with intercept column
            R_aug = np.column_stack([R, np.ones(n)])
            theta, _, _, _ = np.linalg.lstsq(R_aug, y_ma, rcond=None)
            alpha_vec = theta[:p]
            a0 = float(theta[p])
            if not np.all(np.isfinite(alpha_vec)) or not np.isfinite(a0):
                return None
            # Return same format as ridge (with dummy mu, sig)
            mu = np.zeros(p, dtype=float)
            sig = np.ones(p, dtype=float)
            return (alpha_vec, a0, mu, sig)
    except np.linalg.LinAlgError:
        return None

def forecast_ma(alpha_pack, res, q, Q, m, basis_function=None):
    # alpha_pack = (alpha_vec, a0, mu, sig) - we use alpha_vec and a0 on raw features
    alpha_vec, a0, _, _ = alpha_pack
    nr = list(res[-q:][::-1]) if q > 0 else []
    sr = [res[-r * m] for r in range(1, Q + 1)] if Q > 0 else []
    if basis_function is not None:
        feats = basis_function(nr, sr)
    else:
        # Default to original quadratic basis
        feats = build_features(nr, sr, True, False)
    feats = np.asarray(feats, dtype=float)
    return float(feats @ alpha_vec + a0)

class GalerkinSARIMA:
    """
    Galerkin-SARIMA model for forecasting.
    Mimics statsmodels ARIMA interface but uses a Galerkin approximation with OLS.
    Currently supports d=0, D=0 only.
    """
    def __init__(self, endog=None, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1),
                 basis_functions=None, include_sq_lags=True, include_sq_seasonal=False,
                 forecast_method='recursive',
                 # --- Ridge config ---
                 use_ridge=False,             # True: ridge regression, False: OLS
                 ridge_lambda_ar=1.0, ridge_lambda_ma=1.0,
                 ridge_weight_scheme='none',  # 'none' | 'poly' | 'exp' | 'custom'
                 ridge_eta=1.0,               # exponent/shape for schemes
                 ridge_weights_vector=None,   # if 'custom', provide a 1D vector (for AR)
                 ridge_weights_vector_ma=None, # if 'custom', provide a 1D vector (for MA, optional)
                 standardize=True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.p, self.d, self.q = self.order
        self.P, self.D, self.Q, self.m = self.seasonal_order
        if self.d != 0 or self.D != 0:
            raise NotImplementedError("Differencing (d>0 or D>0) is not implemented.")
        
        # Handle basis functions
        if basis_functions is not None:
            if isinstance(basis_functions, list):
                self.basis_function = combine_basis_functions(basis_functions)
            else:
                # Single basis function (string or callable)
                if isinstance(basis_functions, str):
                    if basis_functions in BASIS_FUNCTIONS:
                        self.basis_function = BASIS_FUNCTIONS[basis_functions]
                    else:
                        raise ValueError(f"Unknown basis function: {basis_functions}")
                else:
                    self.basis_function = basis_functions
        else:
            # Default to original behavior
            self.basis_function = None
            self.include_sq_lags = include_sq_lags
            self.include_sq_seasonal = include_sq_seasonal
        
        # Forecast method: 'recursive' or 'direct'
        if forecast_method not in ['recursive', 'direct']:
            raise ValueError("forecast_method must be 'recursive' or 'direct'")
        self.forecast_method = forecast_method
        
        # Ridge regression parameters
        self.use_ridge = bool(use_ridge)
        self.ridge_lambda_ar = float(ridge_lambda_ar)
        self.ridge_lambda_ma = float(ridge_lambda_ma)
        self.ridge_weight_scheme = ridge_weight_scheme
        self.ridge_eta = float(ridge_eta)
        self.ridge_weights_vector = ridge_weights_vector
        self.ridge_weights_vector_ma = ridge_weights_vector_ma
        self.standardize = bool(standardize)
        
        self.endog = None
        self.exog = None
        self.beta = None
        self.delta = None  # Exogenous factor coefficients
        self.res_ar = None
        self.alpha = None
        self.n_obs_used = None  # Number of observations used in fitting
        if endog is not None:
            self.fit(endog, exog=exog)

    def _build_ridge_weights(self, n_features, stage='ar'):
        """Build per-basis weights for ridge regression.
        
        Args:
            n_features: Number of features in the design matrix
            stage: 'ar' or 'ma' to select the appropriate custom weight vector
        """
        # If a custom vector is provided, validate and use it.
        if self.ridge_weight_scheme == 'custom':
            # Select the appropriate custom weight vector
            if stage == 'ar':
                weight_vec = self.ridge_weights_vector
                if weight_vec is None:
                    raise ValueError("Provide ridge_weights_vector for scheme='custom'.")
            else:  # stage == 'ma'
                # Use MA-specific weights if provided, otherwise fall back to AR weights or error
                weight_vec = self.ridge_weights_vector_ma
                if weight_vec is None:
                    # Fall back to using a generated pattern instead of failing
                    # This allows using custom weights for AR but auto-weights for MA
                    return self._build_ridge_weights_auto(n_features)
            
            w = np.asarray(weight_vec, dtype=float)
            if w.shape[0] != n_features:
                raise ValueError(f"ridge_weights_vector{('_ma' if stage=='ma' else '')} length {w.shape[0]} != n_features {n_features}")
            return w
        
        # Otherwise use auto-generated patterns
        return self._build_ridge_weights_auto(n_features)
    
    def _build_ridge_weights_auto(self, n_features):
        """Build automatic ridge weights based on scheme (non-custom)."""

        # Otherwise construct simple patterns. Index from 1…K for readability.
        idx = np.arange(1, n_features + 1, dtype=float)

        # If scheme is 'custom' but we're in auto mode (fallback), use 'none'
        if self.ridge_weight_scheme == 'custom' or self.ridge_weight_scheme == 'none':
            return np.ones(n_features, dtype=float)

        elif self.ridge_weight_scheme == 'poly':
            # Penalize higher-indexed basis more: w_j = j^eta
            return np.power(idx, self.ridge_eta)

        elif self.ridge_weight_scheme == 'exp':
            # Exponential growth in penalty with index: w_j = exp(eta*(j-1))
            return np.exp(self.ridge_eta * (idx - 1.0))

        else:
            raise ValueError(f"Unknown ridge_weight_scheme: {self.ridge_weight_scheme}")

    def fit(self, endog, exog=None):
        """
        Fit the Galerkin-SARIMA model with optional exogenous factors.
        
        Args:
            endog: Endogenous time series (1D array)
            exog: Exogenous factors (2D array, shape (n_obs, n_factors))
        """
        self.endog = np.asarray(endog, dtype=float)
        if exog is not None:
            self.exog = np.asarray(exog, dtype=float)
            if self.exog.ndim == 1:
                self.exog = self.exog.reshape(-1, 1)
            if len(self.exog) != len(self.endog):
                raise ValueError(f"exog length {len(self.exog)} must match endog length {len(self.endog)}")
        else:
            self.exog = None
        
        # Build a temporary design matrix once to get n_features for weights
        # (Do this cheaply using the last row; then rebuild properly inside fit_ar_stage anyway.)
        tmp_ns = list(self.endog[max(0, len(self.endog)-self.p):][::-1]) if self.p > 0 else []
        tmp_ss = [self.endog[-r * self.m] for r in range(1, self.P + 1)] if (self.P > 0 and len(self.endog) >= self.P*self.m+1) else []
        if self.basis_function is not None:
            tmp_feats = self.basis_function(tmp_ns, tmp_ss)
        else:
            tmp_feats = build_features(tmp_ns, tmp_ss, True, False)
        n_features = len(tmp_feats)

        w_ar = self._build_ridge_weights(n_features, stage='ar') if self.use_ridge else None
        
        # Prepare exog for AR stage (aligned with start_ar)
        start_ar = max(self.p, self.P * self.m)
        exog_ar = None
        if self.exog is not None:
            exog_ar = self.exog[start_ar:]
        
        result = fit_ar_stage(
            self.endog, self.p, self.P, self.m, self.basis_function,
            use_ridge=self.use_ridge,
            ridge_lambda=self.ridge_lambda_ar, ridge_weights=w_ar, 
            standardize=self.standardize, exog=exog_ar
        )
        
        if result[0] is None:
            self.beta = None
            self.delta = None
            self.res_ar = np.array([])
        else:
            beta_pack, self.res_ar = result
            if len(beta_pack) == 5:
                beta_vec, self.delta, b0, mu, sig = beta_pack
                # Store in format: (beta_vec, delta, b0, mu, sig) for exog case
                # or (beta_vec, b0, mu, sig) for backward compatibility
                if self.delta is not None:
                    self.beta = (beta_vec, self.delta, b0, mu, sig)
                else:
                    self.beta = (beta_vec, b0, mu, sig)
            else:
                # Backward compatibility (no exog)
                self.beta = beta_pack
                self.delta = None
        
        if self.beta is not None and len(self.res_ar) > 0 and (self.q > 0 or self.Q > 0):
            # Build MA feature dimension (may differ from AR if q != p or Q != P)
            tmp_nr = [0.0] * self.q if self.q > 0 else []
            tmp_sr = [0.0] * self.Q if self.Q > 0 else []
            if self.basis_function is not None:
                tmp_ma_feats = self.basis_function(tmp_nr, tmp_sr)
            else:
                tmp_ma_feats = build_features(tmp_nr, tmp_sr, True, False)
            n_ma_features = len(tmp_ma_feats)
            
            w_ma = self._build_ridge_weights(n_ma_features, stage='ma') if self.use_ridge else None
            self.alpha = fit_ma_stage(
                self.res_ar, self.q, self.Q, self.m, self.basis_function,
                use_ridge=self.use_ridge,
                ridge_lambda=self.ridge_lambda_ma, ridge_weights=w_ma, standardize=self.standardize
            )
        else:
            self.alpha = None
        
        # Store number of observations used
        start_ar = max(self.p, self.P * self.m)
        start_ma = max(self.q, self.Q * self.m) if self.alpha is not None else 0
        self.n_obs_used = len(self.endog) - max(start_ar, start_ma)
        
        return self  # Could return a results wrapper in future

    def forecast(self, steps=1, exog=None):
        """
        Generate forecasts for the specified number of steps.
        
        Args:
            steps: Number of steps to forecast (default=1)
            exog: Exogenous factors for forecast period (2D array, shape (steps, n_factors))
            
        Returns:
            float or array: Forecast value(s)
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        
        if self.endog is None or len(self.endog) == 0:
            if steps == 1:
                return 0.0
            else:
                return np.zeros(steps)
        
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            if len(exog) != steps:
                raise ValueError(f"exog length {len(exog)} must match steps {steps}")
        
        if steps == 1:
            exog_t = exog[0] if exog is not None else None
            return self._forecast_one_step(exog_t=exog_t)
        else:
            if self.forecast_method == 'recursive':
                return self._forecast_recursive(steps, exog=exog)
            else:  # direct
                return self._forecast_direct(steps)
    
    def _forecast_one_step(self, exog_t=None):
        """Generate one-step forecast.
        
        Args:
            exog_t: Exogenous factors at forecast time (1D array)
        """
        last_val = self.endog[-1]
        if self.beta is None:
            return last_val
        
        # Handle beta format (with or without delta)
        if isinstance(self.beta, tuple):
            if len(self.beta) == 5:
                # Format: (beta_vec, delta, b0, mu, sig)
                beta_vec, delta, b0, mu, sig = self.beta
                beta_pack = (beta_vec, delta, b0, mu, sig)
            elif len(self.beta) == 4:
                # Format: (beta_vec, b0, mu, sig) - backward compatibility
                beta_pack = self.beta
            else:
                beta_pack = self.beta
        else:
            # Backward compatibility
            beta_pack = self.beta
        
        forecast_ar_val = forecast_ar(
            beta_pack, self.endog, self.p, self.P, self.m, self.basis_function, exog_t=exog_t
        )
        forecast_ma_val = 0.0
        if self.alpha is not None:
            forecast_ma_val = forecast_ma(
                self.alpha, self.res_ar, self.q, self.Q, self.m, self.basis_function
            )
        out = forecast_ar_val + forecast_ma_val
        return float(np.nan_to_num(out, nan=last_val, posinf=last_val, neginf=last_val))
    
    def _forecast_recursive(self, steps, exog=None):
        """
        Generate multi-step forecasts using recursive approach.
        Each forecast is used as input for the next forecast.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous factors for forecast period (2D array, shape (steps, n_factors))
        """
        forecasts = []
        current_data = self.endog.copy()
        current_res = self.res_ar.copy() if self.res_ar is not None else None
        
        for step in range(steps):
            # Get exog for this step
            exog_t = None
            if exog is not None:
                exog_t = exog[step] if step < len(exog) else None
            
            # Generate one-step forecast
            if self.beta is None:
                forecast_val = current_data[-1]
            else:
                # Handle beta format
                if isinstance(self.beta, tuple):
                    if len(self.beta) == 5:
                        beta_pack = self.beta
                    elif len(self.beta) == 4:
                        beta_pack = self.beta
                    else:
                        beta_pack = self.beta
                else:
                    beta_pack = self.beta
                
                forecast_ar_val = forecast_ar(
                    beta_pack, current_data, self.p, self.P, self.m, 
                    self.basis_function, exog_t=exog_t
                )
                forecast_ma_val = 0.0
                if self.alpha is not None and current_res is not None:
                    forecast_ma_val = forecast_ma(
                        self.alpha, current_res, self.q, self.Q, self.m, self.basis_function
                    )
                forecast_val = forecast_ar_val + forecast_ma_val
            
            # Handle invalid forecasts
            if not np.isfinite(forecast_val):
                forecast_val = current_data[-1]
            
            forecasts.append(forecast_val)
            
            # Update data for next step
            current_data = np.append(current_data, forecast_val)
            
            # Update residuals if MA component exists
            if self.alpha is not None and current_res is not None:
                # Calculate new residual (simplified - in practice this would be more complex)
                new_res = forecast_val - forecast_ar_val
                current_res = np.append(current_res, new_res)
        
        return np.array(forecasts)
    
    def _forecast_direct(self, steps):
        """
        Generate multi-step forecasts using direct approach.
        Train separate models for each forecast horizon.
        """
        forecasts = []
        
        for step in range(1, steps + 1):
            # Create target for this horizon
            if len(self.endog) <= step:
                forecasts.append(self.endog[-1])
                continue
            
            # Create shifted target
            y_target = self.endog[step:]
            X_features = []
            
            # Create features for each time point
            for t in range(step, len(self.endog)):
                ns = list(self.endog[t-step:t][::-1]) if self.p > 0 else []
                ss = [self.endog[t-step-r*self.m] for r in range(1, self.P+1)] if self.P > 0 else []
                
                if self.basis_function is not None:
                    feats = self.basis_function(ns, ss)
                else:
                    feats = build_features(ns, ss, self.include_sq_lags, self.include_sq_seasonal)
                X_features.append(feats)
            
            X_features = np.array(X_features)
            y_target = np.array(y_target)
            
            # Fit model for this horizon
            if X_features.shape[0] <= X_features.shape[1] or not np.all(np.isfinite(X_features)):
                forecasts.append(self.endog[-1])
                continue
            
            try:
                n_features = X_features.shape[1]
                
                if self.use_ridge:
                    # Use weighted ridge for direct forecasting
                    w_h = self._build_ridge_weights(n_features, stage='ar')
                    beta_h, b0_h, mu_h, sig_h = _weighted_ridge(X_features, y_target, w_h, self.ridge_lambda_ar, standardize=self.standardize)
                else:
                    # Use OLS for direct forecasting
                    n, p = X_features.shape
                    X_aug = np.column_stack([X_features, np.ones(n)])
                    theta, _, _, _ = np.linalg.lstsq(X_aug, y_target, rcond=None)
                    beta_h = theta[:p]
                    b0_h = float(theta[p])
                    mu_h = np.zeros(p, dtype=float)
                    sig_h = np.ones(p, dtype=float)
                
                if not np.all(np.isfinite(beta_h)) or not np.isfinite(b0_h):
                    forecasts.append(self.endog[-1])
                    continue
                
                # Generate forecast for this horizon
                ns = list(self.endog[-step:][::-1]) if self.p > 0 else []
                ss = [self.endog[-step-r*self.m] for r in range(1, self.P+1)] if self.P > 0 else []
                
                if self.basis_function is not None:
                    feats = self.basis_function(ns, ss)
                else:
                    feats = build_features(ns, ss, self.include_sq_lags, self.include_sq_seasonal)
                
                feats = np.asarray(feats, dtype=float)
                forecast_val = float(feats @ beta_h + b0_h)
                if not np.isfinite(forecast_val):
                    forecast_val = self.endog[-1]
                forecasts.append(forecast_val)
                
            except np.linalg.LinAlgError:
                forecasts.append(self.endog[-1])
        
        return np.array(forecasts)
    
    def bic(self):
        """
        Calculate Bayesian Information Criterion (BIC) for model selection.
        
        Returns:
            float: BIC value
        """
        if self.n_obs_used is None or self.n_obs_used == 0:
            return np.inf
        
        # Calculate residual sum of squares
        if self.res_ar is None or len(self.res_ar) == 0:
            return np.inf
        
        # Get final residuals after MA stage
        if self.alpha is not None:
            # Use MA residuals if available
            start_ma = max(self.q, self.Q * self.m)
            if len(self.res_ar) > start_ma:
                final_residuals = []
                for t in range(start_ma, len(self.res_ar)):
                    nr = list(self.res_ar[t - self.q : t][::-1]) if self.q > 0 else []
                    sr = [self.res_ar[t - r * self.m] for r in range(1, self.Q + 1)] if self.Q > 0 else []
                    if self.basis_function is not None:
                        feats = self.basis_function(nr, sr)
                    else:
                        feats = build_features(nr, sr, True, False)
                    feats = np.asarray(feats, dtype=float)
                    if isinstance(self.alpha, tuple):
                        alpha_vec, a0, _, _ = self.alpha
                    else:
                        alpha_vec, a0, _, _ = self.alpha, 0.0, None, None
                    pred_ma = float(feats @ alpha_vec + a0)
                    final_residuals.append(self.res_ar[t] - pred_ma)
                rss = np.sum(np.array(final_residuals) ** 2)
                n_used = len(final_residuals)
            else:
                rss = np.sum(self.res_ar ** 2)
                n_used = len(self.res_ar)
        else:
            rss = np.sum(self.res_ar ** 2)
            n_used = len(self.res_ar)
        
        # Count parameters
        # AR parameters
        if self.beta is not None:
            if isinstance(self.beta, tuple):
                if len(self.beta) >= 1:
                    beta_vec = self.beta[0]
                else:
                    beta_vec = np.array([])
            else:
                beta_vec = self.beta
            n_ar_params = len(beta_vec) + 1  # +1 for intercept
        else:
            n_ar_params = 0
        
        # Exogenous parameters
        n_exog_params = len(self.delta) if self.delta is not None else 0
        
        # MA parameters
        if self.alpha is not None:
            if isinstance(self.alpha, tuple):
                if len(self.alpha) >= 1:
                    alpha_vec = self.alpha[0]
                else:
                    alpha_vec = np.array([])
            else:
                alpha_vec = self.alpha
            n_ma_params = len(alpha_vec) + 1  # +1 for intercept
        else:
            n_ma_params = 0
        
        total_params = n_ar_params + n_exog_params + n_ma_params
        
        # BIC = log(RSS/n) + (k * log(n)) / n
        if n_used == 0:
            return np.inf
        
        log_rss_n = np.log(max(rss / n_used, 1e-10))
        penalty = (total_params * np.log(n_used)) / n_used
        
        return log_rss_n + penalty
    
    def block_bootstrap(self, endog, exog=None, block_length=None, n_bootstrap=1000, 
                        random_state=None):
        """
        Perform block bootstrap for inference on exogenous factors and prediction intervals.
        
        Args:
            endog: Endogenous time series
            exog: Exogenous factors (optional)
            block_length: Block length (default: sqrt(n))
            n_bootstrap: Number of bootstrap replications
            random_state: Random seed
        
        Returns:
            dict: Bootstrap results with keys:
                - 'theta_star': Bootstrap coefficient estimates (list of arrays)
                - 'forecasts_star': Bootstrap forecasts (list of arrays)
                - 'residuals_star': Bootstrap residuals (list of arrays)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        endog = np.asarray(endog, dtype=float)
        n = len(endog)
        
        if block_length is None:
            block_length = int(np.sqrt(n))
        
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            if len(exog) != n:
                raise ValueError(f"exog length {len(exog)} must match endog length {n}")
        
        theta_star = []
        forecasts_star = []
        residuals_star = []
        
        # Ensure block_length doesn't exceed n
        if block_length is None:
            block_length = max(1, int(np.sqrt(n)))
        if block_length >= n:
            block_length = max(1, n // 2)
        if n - block_length + 1 <= 0:
            block_length = 1
        
        for b in range(n_bootstrap):
            # Generate bootstrap sample using moving block bootstrap
            indices = []
            if block_length == 1 or n - block_length + 1 <= 0:
                # If block is too large or n is too small, use simple bootstrap
                indices = np.random.choice(n, size=n, replace=True).tolist()
            else:
                while len(indices) < n:
                    start_idx = np.random.randint(0, max(1, n - block_length + 1))
                    block_indices = list(range(start_idx, min(start_idx + block_length, n)))
                    indices.extend(block_indices)
                indices = indices[:n]
            
            endog_star = endog[indices]
            exog_star = exog[indices] if exog is not None else None
            
            # Fit model on bootstrap sample
            # Only pass include_sq_lags/include_sq_seasonal if they exist (when basis_functions is None)
            init_kwargs = {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'basis_functions': self.basis_function,
                'forecast_method': self.forecast_method,
                'use_ridge': self.use_ridge,
                'ridge_lambda_ar': self.ridge_lambda_ar,
                'ridge_lambda_ma': self.ridge_lambda_ma,
                'ridge_weight_scheme': self.ridge_weight_scheme,
                'ridge_eta': self.ridge_eta,
                'ridge_weights_vector': self.ridge_weights_vector,
                'ridge_weights_vector_ma': self.ridge_weights_vector_ma,
                'standardize': self.standardize
            }
            # Only add include_sq_lags/include_sq_seasonal if they exist (for backward compatibility)
            if hasattr(self, 'include_sq_lags'):
                init_kwargs['include_sq_lags'] = self.include_sq_lags
            if hasattr(self, 'include_sq_seasonal'):
                init_kwargs['include_sq_seasonal'] = self.include_sq_seasonal
            
            model_star = GalerkinSARIMA(**init_kwargs)
            
            try:
                model_star.fit(endog_star, exog=exog_star)
                
                # Store coefficients
                theta_b = {
                    'beta': model_star.beta,
                    'delta': model_star.delta,
                    'alpha': model_star.alpha
                }
                theta_star.append(theta_b)
                
                # Generate forecast
                forecast_b = model_star.forecast(steps=1)
                # Ensure it's a scalar
                if isinstance(forecast_b, (list, np.ndarray)):
                    forecast_b = forecast_b[0] if len(forecast_b) > 0 else np.nan
                forecasts_star.append(float(forecast_b))
                
                # Store residuals
                if model_star.res_ar is not None:
                    residuals_star.append(model_star.res_ar.copy())
                else:
                    residuals_star.append(np.array([]))
                    
            except Exception:
                # Skip failed bootstrap samples
                continue
        
        return {
            'theta_star': theta_star,
            'forecasts_star': forecasts_star,
            'residuals_star': residuals_star
        }
    
    def test_factor_significance(self, endog, exog, block_length=None, n_bootstrap=1000,
                                  random_state=None, alpha=0.05):
        """
        Test significance of exogenous factors using block bootstrap.
        
        Args:
            endog: Endogenous time series
            exog: Exogenous factors
            block_length: Block length for bootstrap
            n_bootstrap: Number of bootstrap replications
            random_state: Random seed
            alpha: Significance level
        
        Returns:
            dict: Test results with keys:
                - 'p_value': p-value for H0: delta = 0
                - 'wald_statistic': Wald test statistic (from paper eq:wald_factor)
                - 'delta_hat': Estimated factor coefficients
                - 'ci_lower': Lower confidence bound
                - 'ci_upper': Upper confidence bound
                - 'reject_h0': Whether to reject H0
        """
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        n_factors = exog.shape[1]
        
        # Fit model
        self.fit(endog, exog=exog)
        
        if self.delta is None:
            return {
                'p_value': 1.0,
                'wald_statistic': 0.0,
                'delta_hat': None,
                'ci_lower': None,
                'ci_upper': None,
                'reject_h0': False
            }
        
        delta_hat = self.delta.copy()
        
        # Bootstrap
        bootstrap_results = self.block_bootstrap(
            endog, exog=exog, block_length=block_length,
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        
        # Extract delta estimates from bootstrap
        delta_star = []
        for theta_b in bootstrap_results['theta_star']:
            if theta_b['delta'] is not None:
                delta_star.append(theta_b['delta'])
        
        if len(delta_star) == 0:
            return {
                'p_value': 1.0,
                'wald_statistic': 0.0,
                'delta_hat': delta_hat,
                'ci_lower': None,
                'ci_upper': None,
                'reject_h0': False
            }
        
        delta_star = np.array(delta_star)
        if delta_star.ndim == 1:
            delta_star = delta_star.reshape(-1, 1)
        
        # Test H0: delta = 0 using Wald statistic from paper (eq:wald_factor)
        # W = (R*theta - r0)^T (R*Omega*R^T)^{-1} (R*theta - r0)
        # For H0: delta = 0, R selects the delta block, r0 = 0
        # So: W = delta^T * Omega_delta^{-1} * delta
        # where Omega_delta = R*Omega*R^T is the covariance matrix of delta
        
        # Compute covariance matrix of delta from bootstrap distribution
        # Omega_delta = Cov(delta) estimated from bootstrap samples
        delta_mean = np.mean(delta_star, axis=0, keepdims=True)
        delta_centered = delta_star - delta_mean
        # Unbiased estimate: divide by (n-1) for sample covariance
        n_boot = len(delta_star)
        if n_boot > 1:
            Omega_delta = (delta_centered.T @ delta_centered) / (n_boot - 1)
        else:
            Omega_delta = np.eye(n_factors) * 1e-6
        
        # Handle scalar case
        if Omega_delta.ndim == 0:
            Omega_delta = np.array([[Omega_delta]])
        elif Omega_delta.shape == ():
            Omega_delta = np.array([[float(Omega_delta)]])
        
        # Ensure Omega_delta is positive definite and invertible
        # Add small regularization to diagonal if needed
        jitter = 1e-10 * np.eye(Omega_delta.shape[0])
        Omega_delta = Omega_delta + jitter
        
        # Compute observed Wald statistic: W_obs = delta_hat^T * Omega_delta^{-1} * delta_hat
        delta_hat_vec = np.asarray(delta_hat).reshape(-1, 1)
        try:
            Omega_delta_inv = np.linalg.inv(Omega_delta)
            wald_obs = float(delta_hat_vec.T @ Omega_delta_inv @ delta_hat_vec)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse or sum of squares
            try:
                Omega_delta_inv = np.linalg.pinv(Omega_delta)
                wald_obs = float(delta_hat_vec.T @ Omega_delta_inv @ delta_hat_vec)
            except:
                # Ultimate fallback: sum of squares (not ideal but works)
                wald_obs = float(np.sum(delta_hat ** 2))
                Omega_delta_inv = np.eye(n_factors)
        
        # Compute Wald statistic for each bootstrap sample
        # Using the same covariance matrix (as per paper's bootstrap procedure)
        wald_stats = []
        for d_star in delta_star:
            d_star_vec = np.asarray(d_star).reshape(-1, 1)
            try:
                wald_stat = float(d_star_vec.T @ Omega_delta_inv @ d_star_vec)
            except:
                # Fallback
                wald_stat = float(np.sum(d_star ** 2))
            wald_stats.append(wald_stat)
        
        # p-value: proportion of bootstrap Wald statistics >= observed
        # Under H0: delta = 0, W ~ chi^2(r) asymptotically, but we use bootstrap distribution
        p_value = np.mean(np.array(wald_stats) >= wald_obs)
        
        # Confidence intervals
        ci_lower = np.percentile(delta_star, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(delta_star, 100 * (1 - alpha / 2), axis=0)
        
        return {
            'p_value': p_value,
            'wald_statistic': wald_obs,
            'delta_hat': delta_hat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'reject_h0': p_value < alpha
        }
    
    def prediction_intervals(self, endog, exog=None, steps=1, block_length=None,
                            n_bootstrap=1000, random_state=None, alpha=0.05):
        """
        Compute prediction intervals for forecasts using block bootstrap.
        
        Args:
            endog: Endogenous time series
            exog: Exogenous factors (for forecast period if steps > 1)
            steps: Forecast horizon
            block_length: Block length for bootstrap
            n_bootstrap: Number of bootstrap replications
            random_state: Random seed
            alpha: Significance level (1 - coverage)
        
        Returns:
            dict: Prediction intervals with keys:
                - 'forecast': Point forecast
                - 'ci_mean': Confidence interval for conditional mean
                - 'pi': Prediction interval for future observation
        """
        # Fit model
        if exog is not None:
            exog_full = exog[:len(endog)] if len(exog) > len(endog) else exog
        else:
            exog_full = None
        
        self.fit(endog, exog=exog_full)
        
        # Point forecast
        if exog is not None and len(exog) > len(endog):
            exog_forecast = exog[len(endog):len(endog)+steps]
        else:
            exog_forecast = None
        
        forecast_point = self.forecast(steps=steps, exog=exog_forecast)
        
        # Bootstrap
        bootstrap_results = self.block_bootstrap(
            endog, exog=exog_full, block_length=block_length,
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        
        # Bootstrap forecasts for conditional mean
        forecasts_star = bootstrap_results['forecasts_star']
        if len(forecasts_star) == 0:
            return {
                'forecast': forecast_point,
                'ci_mean': (forecast_point, forecast_point),
                'pi': (forecast_point, forecast_point)
            }
        
        # For multi-step, we need to handle differently
        if steps == 1:
            # Convert forecasts to array and filter finite values
            forecasts_star_clean = []
            for f in forecasts_star:
                if isinstance(f, (list, np.ndarray)):
                    f_val = f[0] if len(f) > 0 else np.nan
                else:
                    f_val = f
                if np.isfinite(f_val):
                    forecasts_star_clean.append(f_val)
            forecasts_star = np.array(forecasts_star_clean)
            
            if len(forecasts_star) == 0:
                return {
                    'forecast': forecast_point,
                    'ci_mean': (forecast_point, forecast_point),
                    'pi': (forecast_point, forecast_point)
                }
            
            # Confidence interval for conditional mean
            ci_mean_lower = np.percentile(forecasts_star, 100 * alpha / 2)
            ci_mean_upper = np.percentile(forecasts_star, 100 * (1 - alpha / 2))
            
            # Prediction interval: add innovation uncertainty
            residuals_star = bootstrap_results['residuals_star']
            residuals_flat = []
            for res in residuals_star:
                if len(res) > 0:
                    residuals_flat.extend(res)
            
            if len(residuals_flat) > 0:
                residuals_flat = np.array(residuals_flat)
                residuals_flat = residuals_flat[np.isfinite(residuals_flat)]
                
                # Sample innovations
                innovations = np.random.choice(
                    residuals_flat, 
                    size=min(len(forecasts_star), n_bootstrap),
                    replace=True
                )
                
                # Predictive draws
                predictive_draws = forecasts_star[:len(innovations)] + innovations
                
                pi_lower = np.percentile(predictive_draws, 100 * alpha / 2)
                pi_upper = np.percentile(predictive_draws, 100 * (1 - alpha / 2))
            else:
                pi_lower = ci_mean_lower
                pi_upper = ci_mean_upper
            
            # PI must be at least as wide as CI (observation = mean + innovation)
            pi_lower = min(pi_lower, ci_mean_lower)
            pi_upper = max(pi_upper, ci_mean_upper)
            
            # Ensure intervals are tuples of floats
            ci_mean_lower = float(ci_mean_lower) if np.isfinite(ci_mean_lower) else float(forecast_point)
            ci_mean_upper = float(ci_mean_upper) if np.isfinite(ci_mean_upper) else float(forecast_point)
            pi_lower = float(pi_lower) if np.isfinite(pi_lower) else float(forecast_point)
            pi_upper = float(pi_upper) if np.isfinite(pi_upper) else float(forecast_point)
            
            return {
                'forecast': float(forecast_point),
                'ci_mean': (ci_mean_lower, ci_mean_upper),
                'pi': (pi_lower, pi_upper)
            }
        else:
            # Multi-step: simplified version
            return {
                'forecast': forecast_point,
                'ci_mean': (forecast_point, forecast_point),  # TODO: implement multi-step CI
                'pi': (forecast_point, forecast_point)  # TODO: implement multi-step PI
            }
    
    def rolling_prediction_intervals(self, train, n_forecast, actuals=None, exog=None,
                                     block_length=None, n_bootstrap=500,
                                     random_state=None, alpha=0.05, method='residual'):
        """
        Compute rolling prediction intervals (PI) and confidence intervals (CI)
        for each forecast step. At each step, the model is refit on train +
        previous actuals (if provided) or forecasts (recursive).
        
        Args:
            train: Training endogenous series
            n_forecast: Number of steps to forecast
            actuals: Optional array of actual values for the forecast period.
                     If provided, use them to expand history (for evaluation/plotting).
                     If None, use recursive forecasts.
            exog: Exogenous factors (optional)
            block_length: Block length for bootstrap
            n_bootstrap: Bootstrap replications (for method='bootstrap')
            random_state: Random seed
            alpha: Significance level (1 - coverage)
            method: 'bootstrap' (accurate, slower) or 'residual' (fast, default)
        
        Returns:
            dict with keys:
                - 'forecasts': array of point forecasts
                - 'pi_lower', 'pi_upper': prediction interval bounds per step
                - 'ci_lower', 'ci_upper': confidence interval for mean per step
        """
        train = np.asarray(train, dtype=float)
        forecasts = np.zeros(n_forecast)
        pi_lower = np.zeros(n_forecast)
        pi_upper = np.zeros(n_forecast)
        ci_lower = np.zeros(n_forecast)
        ci_upper = np.zeros(n_forecast)
        
        hist = list(train)
        use_actuals = actuals is not None and len(actuals) >= n_forecast
        
        for i in range(n_forecast):
            # Fit on current history
            hist_arr = np.array(hist)
            exog_cur = exog[:len(hist_arr)] if exog is not None and len(exog) >= len(hist_arr) else None
            
            self.fit(hist_arr, exog=exog_cur)
            pred = self.forecast(steps=1)
            forecasts[i] = pred
            
            if method == 'residual':
                # Fast residual-based intervals (use full model residuals)
                try:
                    res = self._get_residuals(hist_arr, exog_cur)
                except Exception:
                    res = self.res_ar
                if res is not None and len(res) > 0:
                    res_clean = res[np.isfinite(res)]
                    if len(res_clean) > 0:
                        sigma = np.std(res_clean, ddof=1)
                        z = scipy.stats.norm.ppf(1 - alpha / 2)
                        # PI: forecast +/- z * sigma (observation uncertainty)
                        pi_lower[i] = pred - z * sigma
                        pi_upper[i] = pred + z * sigma
                        # CI: narrower, approximate as forecast +/- z * sigma/sqrt(n)
                        n_eff = max(len(res_clean), 10)
                        ci_half = z * sigma / np.sqrt(n_eff)
                        ci_lower[i] = pred - ci_half
                        ci_upper[i] = pred + ci_half
                    else:
                        pi_lower[i] = pi_upper[i] = ci_lower[i] = ci_upper[i] = pred
                else:
                    pi_lower[i] = pi_upper[i] = ci_lower[i] = ci_upper[i] = pred
            else:
                # Bootstrap-based intervals
                pi_result = self.prediction_intervals(
                    hist_arr, exog=exog_cur, steps=1,
                    block_length=block_length, n_bootstrap=n_bootstrap,
                    random_state=random_state, alpha=alpha
                )
                pi_lower[i] = pi_result['pi'][0]
                pi_upper[i] = pi_result['pi'][1]
                ci_lower[i] = pi_result['ci_mean'][0]
                ci_upper[i] = pi_result['ci_mean'][1]
            
            # Extend history for next step
            if use_actuals:
                hist.append(actuals[i])
            else:
                hist.append(forecasts[i])
        
        return {
            'forecasts': forecasts,
            'pi_lower': pi_lower,
            'pi_upper': pi_upper,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def get_basis_function_info(self):
        """Get information about the current basis function."""
        if self.basis_function is not None:
            return f"Custom basis function: {self.basis_function.__name__ if hasattr(self.basis_function, '__name__') else 'anonymous'}"
        else:
            # Only access include_sq_lags/include_sq_seasonal if they exist
            if hasattr(self, 'include_sq_lags') and hasattr(self, 'include_sq_seasonal'):
                return f"Default basis function (include_sq_lags={self.include_sq_lags}, include_sq_seasonal={self.include_sq_seasonal})"
            else:
                return "Default basis function (legacy mode)"
    
    def get_forecast_method(self):
        """Get the current forecast method."""
        return self.forecast_method
    
    def _get_fitted_values(self, endog, exog=None):
        """Get fitted values for the model."""
        if self.beta is None:
            return np.zeros(len(endog))
        
        fitted = np.zeros(len(endog))
        start_ar = max(self.p, self.P * self.m)
        
        for t in range(start_ar, len(endog)):
            # AR component
            ns = list(endog[t - self.p : t][::-1]) if self.p > 0 else []
            ss = [endog[t - r * self.m] for r in range(1, self.P + 1)] if self.P > 0 else []
            if self.basis_function is not None:
                feats_ar = self.basis_function(ns, ss)
            else:
                feats_ar = build_features(ns, ss, True, False)
            feats_ar = np.asarray(feats_ar, dtype=float)
            
            if isinstance(self.beta, tuple):
                if len(self.beta) >= 1:
                    beta_vec = np.asarray(self.beta[0]).flatten()
                    # (beta_vec, b0, mu, sig) or (beta_vec, delta, b0, mu, sig)
                    b0 = float(self.beta[2]) if len(self.beta) == 5 else (float(self.beta[1]) if len(self.beta) >= 2 else 0.0)
                else:
                    beta_vec = np.array([])
                    b0 = 0.0
            else:
                beta_vec = np.asarray(self.beta).flatten()
                b0 = 0.0

            val = np.asarray(feats_ar).flatten() @ beta_vec + b0
            ar_pred = float(np.asarray(val).flatten()[0]) if np.size(val) > 0 else 0.0
            
            # Exogenous component
            if exog is not None and self.delta is not None:
                exog_t = exog[t] if t < len(exog) else None
                if exog_t is not None:
                    ex_val = np.asarray(exog_t).flatten() @ np.asarray(self.delta).flatten()
                    ar_pred += float(np.asarray(ex_val).flatten()[0]) if np.size(ex_val) > 0 else 0.0
            
            # MA component
            ma_pred = 0.0
            if self.alpha is not None and self.res_ar is not None and t < len(self.res_ar):
                start_ma = max(self.q, self.Q * self.m)
                if t >= start_ma:
                    nr = list(self.res_ar[t - self.q : t][::-1]) if self.q > 0 else []
                    sr = [self.res_ar[t - r * self.m] for r in range(1, self.Q + 1)] if self.Q > 0 else []
                    if self.basis_function is not None:
                        feats_ma = self.basis_function(nr, sr)
                    else:
                        feats_ma = build_features(nr, sr, True, False)
                    feats_ma = np.asarray(feats_ma, dtype=float)
                    
                    if isinstance(self.alpha, tuple):
                        alpha_vec = np.asarray(self.alpha[0]).flatten()
                        a0 = float(self.alpha[1])
                    else:
                        alpha_vec = np.asarray(self.alpha).flatten()
                        a0 = 0.0
                    ma_val = np.asarray(feats_ma).flatten() @ alpha_vec + a0
                    ma_pred = float(np.asarray(ma_val).flatten()[0]) if np.size(ma_val) > 0 else 0.0
            
            fitted[t] = ar_pred + ma_pred
        
        return fitted
    
    def _get_residuals(self, endog, exog=None):
        """Get model residuals."""
        fitted = self._get_fitted_values(endog, exog)
        residuals = endog - fitted
        # Only return residuals for observations used in fitting
        start_ar = max(self.p, self.P * self.m)
        start_ma = max(self.q, self.Q * self.m) if self.alpha is not None else 0
        start = max(start_ar, start_ma)
        return residuals[start:]

    def summary(self, endog, exog=None, block_length=None, n_bootstrap=1000,
                random_state=None, alpha=0.05):
        """
        Generate a comprehensive regression-style summary (similar to R's summary.lm).
        Returns a SummaryResults object; use print(model.summary(...)) to display.
        """
        return _compute_summary(self, endog, exog, block_length, n_bootstrap, random_state, alpha)


# ===================== Summary Results Class =====================

def _compute_summary(model, endog, exog=None, block_length=None, n_bootstrap=1000,
                     random_state=None, alpha=0.05):
    """Internal: compute summary for GalerkinSARIMA. Used by model.summary()."""
    import pandas as pd
    if exog is not None:
        exog_full = exog[:len(endog)] if len(exog) > len(endog) else exog
    else:
        exog_full = None
    model.fit(endog, exog=exog_full)
    residuals = model._get_residuals(endog, exog_full)
    n_obs = len(residuals)
    if n_obs == 0:
        raise ValueError("No observations available for summary")
    residual_stats = {
        'Min': np.min(residuals),
        '1Q': np.percentile(residuals, 25),
        'Median': np.median(residuals),
        '3Q': np.percentile(residuals, 75),
        'Max': np.max(residuals)
    }
    bootstrap_results = model.block_bootstrap(
        endog, exog=exog_full, block_length=block_length,
        n_bootstrap=n_bootstrap, random_state=random_state
    )
    beta_star, alpha_star, delta_star = [], [], []
    for theta_b in bootstrap_results['theta_star']:
        if theta_b['beta'] is not None:
            beta_vec = theta_b['beta'][0] if isinstance(theta_b['beta'], tuple) else theta_b['beta']
            beta_star.append(beta_vec)
        if theta_b['alpha'] is not None:
            alpha_vec = theta_b['alpha'][0] if isinstance(theta_b['alpha'], tuple) else theta_b['alpha']
            alpha_star.append(alpha_vec)
        if theta_b['delta'] is not None:
            delta_star.append(theta_b['delta'])
    var_names, coefs, std_errs, t_stats, p_values = [], [], [], [], []
    if model.beta is not None:
        beta_vec = model.beta[0] if isinstance(model.beta, tuple) else model.beta
        n_ar_feats = len(beta_vec)
        for i in range(n_ar_feats):
            var_names.append(f"AR_b{i+1}")
            coefs.append(beta_vec[i])
        if len(beta_star) > 0:
            beta_arr = np.array(beta_star)
            for i in range(n_ar_feats):
                se = np.std(beta_arr[:, i]) if beta_arr.shape[1] > i else np.nan
                std_errs.append(se)
                t_stat = coefs[len(std_errs)-1] / se if se > 1e-10 else 0.0
                t_stats.append(t_stat)
                from scipy import stats
                p = 2 * (1 - stats.norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
                p_values.append(p)
        else:
            std_errs.extend([np.nan] * n_ar_feats)
            t_stats.extend([np.nan] * n_ar_feats)
            p_values.extend([np.nan] * n_ar_feats)
    if model.delta is not None:
        n_ar = len(coefs)
        delta_vec = np.asarray(model.delta).flatten()
        for i, d in enumerate(delta_vec):
            var_names.append(f"Exog_{i+1}")
            coefs.append(float(d))
        if len(delta_star) > 0:
            delta_arr = np.array(delta_star)
            if delta_arr.ndim == 1:
                delta_arr = delta_arr.reshape(-1, 1)
            for i in range(delta_arr.shape[1]):
                se = np.std(delta_arr[:, i])
                std_errs.append(se)
                t_stat = coefs[n_ar + i] / se if se > 1e-10 else 0.0
                t_stats.append(t_stat)
                from scipy import stats
                p_values.append(2 * (1 - stats.norm.cdf(abs(t_stat))))
        else:
            std_errs.extend([np.nan] * len(delta_vec))
            t_stats.extend([np.nan] * len(delta_vec))
            p_values.extend([np.nan] * len(delta_vec))
    if model.alpha is not None:
        n_before_ma = len(coefs)
        alpha_vec = model.alpha[0] if isinstance(model.alpha, tuple) else model.alpha
        n_ma_feats = len(alpha_vec)
        for i in range(n_ma_feats):
            var_names.append(f"MA_b{i+1}")
            coefs.append(alpha_vec[i])
        if len(alpha_star) > 0:
            alpha_arr = np.array(alpha_star)
            for i in range(n_ma_feats):
                se = np.std(alpha_arr[:, i]) if alpha_arr.shape[1] > i else np.nan
                std_errs.append(se)
                t_stat = coefs[n_before_ma + i] / se if se > 1e-10 else 0.0
                t_stats.append(t_stat)
                from scipy import stats
                p_values.append(2 * (1 - stats.norm.cdf(abs(t_stat))))
        else:
            std_errs.extend([np.nan] * n_ma_feats)
            t_stats.extend([np.nan] * n_ma_feats)
            p_values.extend([np.nan] * n_ma_feats)
    n_coefs = len(coefs)
    while len(std_errs) < n_coefs:
        std_errs.append(np.nan)
    while len(t_stats) < n_coefs:
        t_stats.append(np.nan)
    while len(p_values) < n_coefs:
        p_values.append(np.nan)
    sig_map = [(0.001, '***'), (0.01, '**'), (0.05, '*'), (0.1, '.'), (1.0, ' ')]
    significance = []
    for p in p_values:
        s = ' '
        for thresh, sym in sig_map:
            if not np.isnan(p) and p < thresh:
                s = sym
                break
        significance.append(s)
    coefficients_df = pd.DataFrame({
        'Variable': var_names,
        'Coef': coefs,
        'Std Err': std_errs[:n_coefs],
        't': t_stats[:n_coefs],
        'P>|t|': p_values[:n_coefs],
        'Significance': significance
    })
    rss = np.sum(residuals ** 2)
    df_residual = n_obs - n_coefs
    residual_se = np.sqrt(rss / df_residual) if df_residual > 0 else np.nan
    start_ar = max(model.p, model.P * model.m)
    y_used = endog[start_ar:start_ar + len(residuals)]
    ss_tot = np.sum((y_used - np.mean(y_used)) ** 2)
    r_squared = 1 - rss / ss_tot if ss_tot > 0 else np.nan
    adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / df_residual if df_residual > 0 and not np.isnan(r_squared) else np.nan
    if not np.isnan(r_squared) and n_coefs > 0 and df_residual > 0 and 0 < r_squared < 1:
        f_statistic = (r_squared / n_coefs) / ((1 - r_squared) / df_residual)
        f_pvalue = 1 - scipy.stats.f.cdf(f_statistic, n_coefs, df_residual)
    else:
        f_statistic, f_pvalue = np.nan, np.nan
    bic_val = model.bic()
    model_stats = {
        'residual_se': residual_se, 'df_residual': df_residual,
        'r_squared': r_squared, 'adj_r_squared': adj_r_squared,
        'f_statistic': f_statistic, 'f_df1': n_coefs, 'f_df2': df_residual,
        'f_pvalue': f_pvalue, 'bic': bic_val, 'n_obs': n_obs, 'n_params': n_coefs
    }
    order_str = f"order={model.order}"
    seasonal_str = f"seasonal_order={model.seasonal_order}"
    basis_str = ""
    if model.basis_function is not None:
        basis_str = f", basis_functions='{model.basis_function.__name__}'" if hasattr(model.basis_function, '__name__') else ", basis_functions=custom"
    call_str = f"GalerkinSARIMA({order_str}, {seasonal_str}{basis_str})"
    if exog_full is not None:
        call_str += f" with {exog_full.shape[1]} exogenous factor(s)"
    return SummaryResults(model=model, endog=endog, exog=exog_full, residuals=residuals,
                          coefficients_df=coefficients_df, residual_stats=residual_stats,
                          model_stats=model_stats, call_str=call_str)


class SummaryResults:
    """
    Comprehensive summary results for Galerkin-ARIMA model (similar to R's summary.lm).
    
    This class provides a formatted summary including:
    - Residuals statistics
    - Coefficients table with inference
    - Model fit statistics (R-squared, F-statistic, BIC, etc.)
    """
    def __init__(self, model, endog, exog, residuals, coefficients_df, 
                 residual_stats, model_stats, call_str):
        self.model = model
        self.endog = endog
        self.exog = exog
        self.residuals = residuals
        self.coefficients = coefficients_df
        self.residual_stats = residual_stats
        self.model_stats = model_stats
        self.call_str = call_str
    
    def __repr__(self):
        return self._format_summary()
    
    def __str__(self):
        return self._format_summary()
    
    def _format_summary(self):
        """Format the summary output similar to R's summary.lm."""
        lines = []
        
        # Call
        lines.append("Call:")
        lines.append(f"  {self.call_str}")
        lines.append("")
        
        # Residuals
        lines.append("Residuals:")
        rs = self.residual_stats
        lines.append(f"     Min       1Q   Median       3Q      Max")
        lines.append(f"{rs['Min']:9.4f} {rs['1Q']:9.4f} {rs['Median']:9.4f} {rs['3Q']:9.4f} {rs['Max']:9.4f}")
        lines.append("")
        
        # Coefficients
        lines.append("Coefficients:")
        df = self.coefficients
        # Format header
        header = f"{'':<15} {'Estimate':>12} {'Std. Error':>12} {'t value':>10} {'Pr(>|t|)':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Format each row
        for idx, row in df.iterrows():
            var_name = row['Variable']
            coef = row['Coef']
            std_err = row['Std Err']
            t_val = row['t']
            p_val = row['P>|t|']
            sig = row['Significance']
            
            # Format p-value
            if np.isnan(p_val):
                p_str = "      NA"
            elif p_val < 2e-16:
                p_str = "  <2e-16"
            else:
                p_str = f"{p_val:10.4f}"
            
            # Format values
            coef_str = f"{coef:12.4f}" if not np.isnan(coef) else "          NA"
            se_str = f"{std_err:12.4f}" if not np.isnan(std_err) else "          NA"
            t_str = f"{t_val:10.4f}" if not np.isnan(t_val) else "        NA"
            
            line = f"{var_name:<15} {coef_str} {se_str} {t_str} {p_str} {sig}"
            lines.append(line)
        
        lines.append("-" * len(header))
        lines.append("")
        
        # Significance codes
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")
        
        # Model statistics
        ms = self.model_stats
        lines.append(f"Residual standard error: {ms['residual_se']:.4f} on {ms['df_residual']} degrees of freedom")
        
        if 'r_squared' in ms and not np.isnan(ms['r_squared']):
            r2_line = f"Multiple R-squared:  {ms['r_squared']:.5f},"
            if 'adj_r_squared' in ms and not np.isnan(ms['adj_r_squared']):
                r2_line += f"\tAdjusted R-squared: {ms['adj_r_squared']:.5f}"
            lines.append(r2_line)
        
        if 'f_statistic' in ms and not np.isnan(ms['f_statistic']):
            f_val = ms['f_statistic']
            f_df1 = ms.get('f_df1', 1)
            f_df2 = ms.get('f_df2', ms['df_residual'])
            f_pval = ms.get('f_pvalue', np.nan)
            if not np.isnan(f_pval):
                if f_pval < 2e-16:
                    p_str = "<2e-16"
                else:
                    p_str = f"{f_pval:.4f}"
                lines.append(f"F-statistic: {f_val:.4f} on {f_df1} and {f_df2} DF,  p-value: {p_str}")
        
        if 'bic' in ms and not np.isnan(ms['bic']):
            lines.append(f"BIC: {ms['bic']:.4f}")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print the formatted summary."""
        print(self._format_summary())


# ===================== Model Selection via BIC =====================

def select_model_bic(endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1),
                     basis_candidates=None, **model_kwargs):
    """
    Select optimal model using BIC criterion.
    
    This function fits multiple models with different basis configurations
    and selects the one with the lowest BIC.
    
    Args:
        endog: Endogenous time series
        exog: Exogenous factors (optional)
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, m)
        basis_candidates: List of basis function configurations to try.
                         Each element can be:
                         - None (default basis)
                         - A string (basis function name)
                         - A callable (basis function)
                         - A list of basis functions
                         If None, uses [None] (default basis only)
        **model_kwargs: Additional arguments to pass to GalerkinSARIMA
    
    Returns:
        dict: Results with keys:
            - 'best_model': GalerkinSARIMA instance with lowest BIC
            - 'best_bic': BIC value of best model
            - 'best_basis': Basis configuration of best model
            - 'all_results': List of dicts with 'model', 'bic', 'basis' for each candidate
    """
    if basis_candidates is None:
        basis_candidates = [None]
    
    best_model = None
    best_bic = np.inf
    best_basis = None
    all_results = []
    
    for basis in basis_candidates:
        try:
            model = GalerkinSARIMA(
                order=order,
                seasonal_order=seasonal_order,
                basis_functions=basis,
                **model_kwargs
            )
            model.fit(endog, exog=exog)
            bic = model.bic()
            
            all_results.append({
                'model': model,
                'bic': bic,
                'basis': basis
            })
            
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_basis = basis
        except Exception as e:
            # Skip failed models
            all_results.append({
                'model': None,
                'bic': np.inf,
                'basis': basis,
                'error': str(e)
            })
            continue
    
    return {
        'best_model': best_model,
        'best_bic': best_bic,
        'best_basis': best_basis,
        'all_results': all_results
    }