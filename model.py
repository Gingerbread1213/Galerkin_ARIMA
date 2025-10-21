import numpy as np
from features import build_features, combine_basis_functions, BASIS_FUNCTIONS

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
                 ridge_lambda=1.0, ridge_weights=None, standardize=True):
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
    
    # Basic sanity checks
    if X_ar.ndim != 2 or X_ar.shape[0] <= X_ar.shape[1]:
        return None, np.array([])
    if not np.all(np.isfinite(X_ar)) or not np.all(np.isfinite(y_ar)):
        return None, np.array([])

    # Use ridge or OLS based on flag
    try:
        if use_ridge:
            # Weighted ridge regression with intercept
            if ridge_weights is None:
                ridge_weights = np.ones(X_ar.shape[1], dtype=float)
            beta, b0, mu, sig = _weighted_ridge(X_ar, y_ar, ridge_weights, ridge_lambda, standardize=standardize)
            if not np.all(np.isfinite(beta)) or not np.isfinite(b0):
                return None, np.array([])
            res = y_ar - (X_ar @ beta + b0)
            return (beta, b0, mu, sig), res
        else:
            # Standard OLS with intercept
            n, p = X_ar.shape
            # Augment with intercept column
            X_aug = np.column_stack([X_ar, np.ones(n)])
            theta, _, _, _ = np.linalg.lstsq(X_aug, y_ar, rcond=None)
            beta = theta[:p]
            b0 = float(theta[p])
            if not np.all(np.isfinite(beta)) or not np.isfinite(b0):
                return None, np.array([])
            res = y_ar - (X_ar @ beta + b0)
            # Return same format as ridge (with dummy mu, sig)
            mu = np.zeros(p, dtype=float)
            sig = np.ones(p, dtype=float)
            return (beta, b0, mu, sig), res
    except np.linalg.LinAlgError:
        return None, np.array([])

def forecast_ar(beta_pack, train, p, P, m, basis_function=None):
    # beta_pack = (beta_vec, b0, mu, sig) - we use beta_vec and b0 on raw features
    beta_vec, b0, _, _ = beta_pack
    ns = list(train[-p:][::-1]) if p > 0 else []
    ss = [train[-r * m] for r in range(1, P + 1)] if P > 0 else []
    if basis_function is not None:
        feats = basis_function(ns, ss)
    else:
        # Default to original quadratic basis
        feats = build_features(ns, ss, True, False)
    feats = np.asarray(feats, dtype=float)
    return float(feats @ beta_vec + b0)

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
    def __init__(self, endog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1),
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
        self.beta = None
        self.res_ar = None
        self.alpha = None
        if endog is not None:
            self.fit(endog)

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

    def fit(self, endog):
        self.endog = np.asarray(endog, dtype=float)
        
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
        self.beta, self.res_ar = fit_ar_stage(
            self.endog, self.p, self.P, self.m, self.basis_function,
            use_ridge=self.use_ridge,
            ridge_lambda=self.ridge_lambda_ar, ridge_weights=w_ar, standardize=self.standardize
        )
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
        return self  # Could return a results wrapper in future

    def forecast(self, steps=1):
        """
        Generate forecasts for the specified number of steps.
        
        Args:
            steps: Number of steps to forecast (default=1)
            
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
        
        if steps == 1:
            return self._forecast_one_step()
        else:
            if self.forecast_method == 'recursive':
                return self._forecast_recursive(steps)
            else:  # direct
                return self._forecast_direct(steps)
    
    def _forecast_one_step(self):
        """Generate one-step forecast."""
        last_val = self.endog[-1]
        if self.beta is None:
            return last_val
        
        forecast_ar_val = forecast_ar(
            self.beta, self.endog, self.p, self.P, self.m, self.basis_function
        )
        forecast_ma_val = 0.0
        if self.alpha is not None:
            forecast_ma_val = forecast_ma(
                self.alpha, self.res_ar, self.q, self.Q, self.m, self.basis_function
            )
        out = forecast_ar_val + forecast_ma_val
        return float(np.nan_to_num(out, nan=last_val, posinf=last_val, neginf=last_val))
    
    def _forecast_recursive(self, steps):
        """
        Generate multi-step forecasts using recursive approach.
        Each forecast is used as input for the next forecast.
        """
        forecasts = []
        current_data = self.endog.copy()
        current_res = self.res_ar.copy() if self.res_ar is not None else None
        
        for step in range(steps):
            # Generate one-step forecast
            if self.beta is None:
                forecast_val = current_data[-1]
            else:
                forecast_ar_val = forecast_ar(
                    self.beta, current_data, self.p, self.P, self.m, self.basis_function
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
    
    def get_basis_function_info(self):
        """Get information about the current basis function."""
        if self.basis_function is not None:
            return f"Custom basis function: {self.basis_function.__name__ if hasattr(self.basis_function, '__name__') else 'anonymous'}"
        else:
            return f"Default basis function (include_sq_lags={self.include_sq_lags}, include_sq_seasonal={self.include_sq_seasonal})"
    
    def get_forecast_method(self):
        """Get the current forecast method."""
        return self.forecast_method

# ===================== Time-Dependent (new, EW-RLS) =====================

class TDRLSGalerkinSARIMA:
    """
    Time-Dependent Galerkin-SARIMA using Exponentially-Weighted Recursive Least Squares (EW-RLS)
    for both AR (features on value lags) and MA (features on residual lags) stages.
    
    API mirrors the time-independent model:
      - order=(p, d, q), seasonal_order=(P, D, Q, m) with d=D=0 only
      - basis_functions: same semantics (string | list[str] | callable)
    
    Fit consumes the full series online (one pass) to set state, and forecast returns one-step ahead.
    """
    def __init__(self, endog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1),
                 basis_functions=None, rho=0.98, lambda_beta=1e-3, lambda_alpha=1e-3,
                 standardize=True, use_ridge=False, ridge_weight_scheme='none',
                 ridge_eta=1.0, ridge_weights_vector=None, ridge_weights_vector_ma=None):
        # Orders
        self.order = order
        self.seasonal_order = seasonal_order
        self.p, self.d, self.q = self.order
        self.P, self.D, self.Q, self.m = self.seasonal_order
        if self.d != 0 or self.D != 0:
            raise NotImplementedError("Differencing (d>0 or D>0) is not implemented in TD model.")
        if self.P > 0 and self.m <= 0:
            raise ValueError("Seasonal period m must be > 0 when P > 0")
        if self.Q > 0 and self.m <= 0:
            raise ValueError("Seasonal period m must be > 0 when Q > 0")
        
        # Basis function handling (same as time-independent)
        if basis_functions is not None:
            if isinstance(basis_functions, list):
                self.basis_function = combine_basis_functions(basis_functions)
            else:
                if isinstance(basis_functions, str):
                    if basis_functions in BASIS_FUNCTIONS:
                        self.basis_function = BASIS_FUNCTIONS[basis_functions]
                    else:
                        raise ValueError(f"Unknown basis function: {basis_functions}")
                else:
                    self.basis_function = basis_functions
        else:
            # Default: use original quadratic-like behavior
            self.basis_function = None  # will fall back to build_features
        
        # EW-RLS hyperparameters
        self.rho = float(rho)
        self.lambda_beta = float(lambda_beta)
        self.lambda_alpha = float(lambda_alpha)
        self.standardize = bool(standardize)
        
        # Ridge regression parameters (per-basis weighted ridge)
        self.use_ridge = bool(use_ridge)
        self.ridge_weight_scheme = ridge_weight_scheme
        self.ridge_eta = float(ridge_eta)
        self.ridge_weights_vector = ridge_weights_vector
        self.ridge_weights_vector_ma = ridge_weights_vector_ma
        
        # Model state
        self.endog = None
        self.beta = None
        self.alpha = None
        self.P_beta = None
        self.P_alpha = None
        self.eps_hist = []  # estimated innovations
        
        # Online standardization stats per feature vector
        self._mu_phi = None
        self._sigma_phi = None
        self._mu_psi = None
        self._sigma_psi = None
        
        if endog is not None:
            self.fit(endog)

    # ---------- Feature builders (aligned with basis_functions) ----------
    def _build_phi(self, y, t):
        # Build non-seasonal and seasonal value lags for AR features
        ns = [y[t - i] for i in range(1, self.p + 1)] if self.p > 0 else []
        ss = [y[t - i * self.m] for i in range(1, self.P + 1)] if (self.P > 0 and self.m > 0) else []
        if self.basis_function is not None:
            feats = self.basis_function(ns, ss)
        else:
            feats = build_features(ns, ss, True, False)
        return np.asarray(feats, dtype=float)

    def _build_psi(self):
        # Build non-seasonal and seasonal residual lags for MA features
        # Fill with zeros if insufficient residual history
        nr = []
        if self.q > 0:
            for i in range(1, self.q + 1):
                if i <= len(self.eps_hist):
                    nr.append(self.eps_hist[-i])
                else:
                    nr.append(0.0)
        sr = []
        if self.Q > 0 and self.m > 0:
            for i in range(1, self.Q + 1):
                idx = i * self.m
                if idx <= len(self.eps_hist):
                    sr.append(self.eps_hist[-idx])
                else:
                    sr.append(0.0)
        if self.basis_function is not None:
            feats = self.basis_function(nr, sr)
        else:
            feats = build_features(nr, sr, True, False)
        return np.asarray(feats, dtype=float)

    # ---------- Online standardization ----------
    def _fit_scale(self, phi, psi):
        if self._mu_phi is None:
            self._mu_phi = np.zeros_like(phi)
            self._sigma_phi = np.ones_like(phi)
        if self._mu_psi is None:
            self._mu_psi = np.zeros_like(psi)
            self._sigma_psi = np.ones_like(psi)
        alpha = 1.0 - self.rho
        self._mu_phi = (1 - alpha) * self._mu_phi + alpha * phi
        self._mu_psi = (1 - alpha) * self._mu_psi + alpha * psi
        self._sigma_phi = np.sqrt((1 - alpha) * (self._sigma_phi ** 2) + alpha * (phi - self._mu_phi) ** 2 + 1e-12)
        self._sigma_psi = np.sqrt((1 - alpha) * (self._sigma_psi ** 2) + alpha * (psi - self._mu_psi) ** 2 + 1e-12)

    def _scale_phi(self, phi):
        if not self.standardize:
            return phi
        return (phi - self._mu_phi) / (self._sigma_phi + 1e-12)

    def _scale_psi(self, psi):
        if not self.standardize:
            return psi
        return (psi - self._mu_psi) / (self._sigma_psi + 1e-12)

    # ---------- Ridge weight builder ----------
    def _build_ridge_weights_auto(self, n_features, stage='ar'):
        """
        Build per-basis ridge weights for TD model.
        stage: 'ar' or 'ma'
        """
        if self.ridge_weight_scheme == 'custom':
            if stage == 'ar':
                if self.ridge_weights_vector is not None:
                    w = np.asarray(self.ridge_weights_vector, dtype=float)
                    if w.shape[0] != n_features:
                        raise ValueError(f"ridge_weights_vector length {w.shape[0]} != n_features {n_features}")
                    return w
                else:
                    # Fallback to uniform if custom requested but not provided
                    return np.ones(n_features, dtype=float)
            else:  # MA stage
                if self.ridge_weights_vector_ma is not None:
                    w = np.asarray(self.ridge_weights_vector_ma, dtype=float)
                    if w.shape[0] != n_features:
                        raise ValueError(f"ridge_weights_vector_ma length {w.shape[0]} != n_features {n_features}")
                    return w
                else:
                    # Fallback to uniform for MA if not provided
                    return np.ones(n_features, dtype=float)
        
        # Auto schemes
        if self.ridge_weight_scheme == 'none':
            return np.ones(n_features, dtype=float)
        elif self.ridge_weight_scheme == 'poly':
            # Polynomial decay: w_i = (i+1)^eta
            return np.array([(i+1)**self.ridge_eta for i in range(n_features)], dtype=float)
        elif self.ridge_weight_scheme == 'exp':
            # Exponential decay: w_i = exp(eta * i)
            return np.array([np.exp(self.ridge_eta * i) for i in range(n_features)], dtype=float)
        else:
            raise ValueError(f"Unknown ridge_weight_scheme: {self.ridge_weight_scheme}")

    # ---------- Ensure shapes and initialize RLS state ----------
    def _ensure_shapes(self, y, t):
        phi = self._build_phi(y, t)
        psi = self._build_psi()
        if self.beta is None:
            d_phi = phi.size
            self.beta = np.zeros(d_phi)
            
            # Initialize P_beta with per-basis weighted ridge if use_ridge=True
            if self.use_ridge:
                w_beta = self._build_ridge_weights_auto(d_phi, 'ar')
                W_beta = np.diag(w_beta ** 2)
                R_beta = self.lambda_beta * W_beta + 1e-10 * np.eye(d_phi)
                self.P_beta = np.linalg.inv(R_beta)
            else:
                # Standard RLS initialization (uniform regularization)
                self.P_beta = (1.0 / self.lambda_beta) * np.eye(d_phi)
            
            self._mu_phi = np.zeros(d_phi)
            self._sigma_phi = np.ones(d_phi)
            
        if self.alpha is None:
            d_psi = psi.size
            self.alpha = np.zeros(d_psi)
            
            # Initialize P_alpha with per-basis weighted ridge if use_ridge=True
            if self.use_ridge:
                w_alpha = self._build_ridge_weights_auto(d_psi, 'ma')
                W_alpha = np.diag(w_alpha ** 2)
                R_alpha = self.lambda_alpha * W_alpha + 1e-10 * np.eye(d_psi)
                self.P_alpha = np.linalg.inv(R_alpha)
            else:
                # Standard RLS initialization (uniform regularization)
                self.P_alpha = (1.0 / self.lambda_alpha) * np.eye(d_psi)
            
            self._mu_psi = np.zeros(d_psi)
            self._sigma_psi = np.ones(d_psi)
        return phi, psi

    # ---------- One online update step ----------
    def partial_fit(self, y, t):
        """
        Consume y[t] and update parameters using EW-RLS.
        Assumes t >= max(p, P*m, q, Q*m).
        Returns a dict of diagnostics.
        """
        phi_raw, psi_raw = self._ensure_shapes(y, t)
        self._fit_scale(phi_raw, psi_raw)
        phi = self._scale_phi(phi_raw)
        psi = self._scale_psi(psi_raw)

        # MA-corrected target
        y_t = float(y[t])
        y_t_tilde = y_t - float(psi @ self.alpha)

        # AR update
        e_ar = y_t_tilde - float(phi @ self.beta)
        denom_beta = self.rho + float(phi.T @ (self.P_beta @ phi))
        K_beta = (self.P_beta @ phi) / denom_beta
        self.beta = self.beta + K_beta * e_ar
        self.P_beta = (1.0 / self.rho) * (np.eye(self.P_beta.shape[0]) - np.outer(K_beta, phi)) @ self.P_beta

        # AR residual
        u_t = y_t - float(phi @ self.beta)

        # MA update
        e_ma = u_t - float(psi @ self.alpha)
        denom_alpha = self.rho + float(psi.T @ (self.P_alpha @ psi))
        K_alpha = (self.P_alpha @ psi) / denom_alpha
        self.alpha = self.alpha + K_alpha * e_ma
        self.P_alpha = (1.0 / self.rho) * (np.eye(self.P_alpha.shape[0]) - np.outer(K_alpha, psi)) @ self.P_alpha

        # Innovation
        eps_t = u_t - float(psi @ self.alpha)
        self.eps_hist.append(float(eps_t))

        return {
            'e_ar': float(e_ar),
            'e_ma': float(e_ma),
            'u_t': float(u_t),
            'eps_t': float(eps_t),
        }

    # ---------- Fitting on a full series ----------
    def fit(self, endog):
        self.endog = np.asarray(endog, dtype=float)
        n = len(self.endog)
        maxlag = max(
            self.p,
            self.P * self.m if (self.P > 0 and self.m > 0) else 0,
            self.q,
            self.Q * self.m if (self.Q > 0 and self.m > 0) else 0,
        )
        start = max(maxlag, 1)
        # reset state
        self.beta = None
        self.alpha = None
        self.P_beta = None
        self.P_alpha = None
        self.eps_hist = []
        self._mu_phi = None
        self._sigma_phi = None
        self._mu_psi = None
        self._sigma_psi = None
        
        # online pass
        for t in range(start, n):
            self.partial_fit(self.endog, t)
        return self

    # ---------- Forecasting ----------
    def forecast(self, steps=1):
        if steps != 1:
            raise NotImplementedError("TD model currently supports steps=1 only.")
        if self.endog is None or len(self.endog) == 0:
            return 0.0
        t = len(self.endog) - 1
        # Build features for t+1 using history up to t
        phi_raw = self._build_phi(self.endog, t + 1)
        psi_raw = self._build_psi()
        phi = self._scale_phi(phi_raw)
        psi = self._scale_psi(psi_raw)
        yhat = float(phi @ self.beta) + float(psi @ self.alpha)
        last_val = self.endog[-1]
        return float(np.nan_to_num(yhat, nan=last_val, posinf=last_val, neginf=last_val))

    def get_basis_function_info(self):
        if self.basis_function is not None:
            return f"Custom basis function: {self.basis_function.__name__ if hasattr(self.basis_function, '__name__') else 'anonymous'}"
        else:
            return "Default basis function (quadratic legacy)"