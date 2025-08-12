import numpy as np
from features import build_features, combine_basis_functions, BASIS_FUNCTIONS

# ===================== Time-Independent (existing) =====================

def fit_ar_stage(train, p, P, m, basis_function=None):
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
    X_ar = np.array(X_ar)
    y_ar = np.array(y_ar)
    if X_ar.shape[0] <= X_ar.shape[1] or not np.all(np.isfinite(X_ar)) or not np.all(np.isfinite(y_ar)):
        return None, np.array([])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
        if not np.all(np.isfinite(beta)):
            return None, np.array([])
        res = y_ar - np.dot(X_ar, beta)
        return beta, res
    except np.linalg.LinAlgError:
        return None, np.array([])

def forecast_ar(beta, train, p, P, m, basis_function=None):
    ns = list(train[-p:][::-1]) if p > 0 else []
    ss = [train[-r * m] for r in range(1, P + 1)] if P > 0 else []
    if basis_function is not None:
        feats = basis_function(ns, ss)
    else:
        # Default to original quadratic basis
        feats = build_features(ns, ss, True, False)
    return float(np.dot(beta, feats))

def fit_ma_stage(res, q, Q, m, basis_function=None):
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
    R = np.array(R)
    y_ma = np.array(y_ma)
    if R.shape[0] <= R.shape[1] or not np.all(np.isfinite(R)) or not np.all(np.isfinite(y_ma)):
        return None
    try:
        alpha, _, _, _ = np.linalg.lstsq(R, y_ma, rcond=None)
        if not np.all(np.isfinite(alpha)):
            return None
        return alpha
    except np.linalg.LinAlgError:
        return None

def forecast_ma(alpha, res, q, Q, m, basis_function=None):
    nr = list(res[-q:][::-1]) if q > 0 else []
    sr = [res[-r * m] for r in range(1, Q + 1)] if Q > 0 else []
    if basis_function is not None:
        feats = basis_function(nr, sr)
    else:
        # Default to original quadratic basis
        feats = build_features(nr, sr, True, False)
    return float(np.dot(alpha, feats))

class GalerkinSARIMA:
    """
    Galerkin-SARIMA model for forecasting.
    Mimics statsmodels ARIMA interface but uses a Galerkin approximation with OLS.
    Currently supports d=0, D=0 only.
    """
    def __init__(self, endog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1),
                 basis_functions=None, include_sq_lags=True, include_sq_seasonal=False,
                 forecast_method='recursive'):
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
        
        self.endog = None
        self.beta = None
        self.res_ar = None
        self.alpha = None
        if endog is not None:
            self.fit(endog)

    def fit(self, endog):
        self.endog = np.asarray(endog, dtype=float)
        self.beta, self.res_ar = fit_ar_stage(
            self.endog, self.p, self.P, self.m, self.basis_function
        )
        if self.beta is not None and len(self.res_ar) > 0 and (self.q > 0 or self.Q > 0):
            self.alpha = fit_ma_stage(
                self.res_ar, self.q, self.Q, self.m, self.basis_function
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
                beta_horizon, _, _, _ = np.linalg.lstsq(X_features, y_target, rcond=None)
                if not np.all(np.isfinite(beta_horizon)):
                    forecasts.append(self.endog[-1])
                    continue
                
                # Generate forecast for this horizon
                ns = list(self.endog[-step:][::-1]) if self.p > 0 else []
                ss = [self.endog[-step-r*self.m] for r in range(1, self.P+1)] if self.P > 0 else []
                
                if self.basis_function is not None:
                    feats = self.basis_function(ns, ss)
                else:
                    feats = build_features(ns, ss, self.include_sq_lags, self.include_sq_seasonal)
                
                forecast_val = float(np.dot(beta_horizon, feats))
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
                 standardize=True):
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

    # ---------- Ensure shapes and initialize RLS state ----------
    def _ensure_shapes(self, y, t):
        phi = self._build_phi(y, t)
        psi = self._build_psi()
        if self.beta is None:
            d_phi = phi.size
            self.beta = np.zeros(d_phi)
            self.P_beta = (1.0 / self.lambda_beta) * np.eye(d_phi)
            self._mu_phi = np.zeros(d_phi)
            self._sigma_phi = np.ones(d_phi)
        if self.alpha is None:
            d_psi = psi.size
            self.alpha = np.zeros(d_psi)
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