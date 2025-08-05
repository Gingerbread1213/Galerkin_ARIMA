import numpy as np
from features import build_features, combine_basis_functions, BASIS_FUNCTIONS

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