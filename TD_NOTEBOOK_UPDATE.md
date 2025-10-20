# TD_GARIMA_sythetic.ipynb - Fixed and Enhanced

## Summary

Fixed and enhanced the Time-Dependent Galerkin-SARIMA notebook to run and compare **four algorithms**:
1. **TD-Galerkin-SARIMA** - Time-dependent version using EW-RLS
2. **GARIMA-OLS** - Time-independent GalerkinSARIMA with OLS
3. **GARIMA-Ridge** - Time-independent GalerkinSARIMA with ridge regression
4. **ARIMA** - Statsmodels baseline

---

## Key Fixes Applied

### Cell 2: Added Missing Definition
**Before:**
```python
orders = [(p, q, P, Q) for p in p_values for q in q_values for P in P_values for Q in Q_values]
# Missing forecast_steps!
```

**After:**
```python
orders = [(p, q, P, Q) for p in p_values for q in q_values for P in P_values for Q in Q_values]
forecast_steps = 1  # Add missing forecast_steps definition
```

### Cell 4: Complete Algorithm Comparison
**Before:**
- Only ran TD-Galerkin-SARIMA
- Incomplete experiment structure
- Missing error handling

**After:**
- Runs **four algorithms** for each dataset
- **TD-Galerkin-SARIMA**: Uses EW-RLS with `rho=0.9999`, `lambda_beta=1e-1`, `lambda_alpha=1e-1`
- **GARIMA-OLS**: `use_ridge=False`
- **GARIMA-Ridge**: `use_ridge=True` with polynomial weights
- **ARIMA**: Statsmodels baseline
- Proper error handling and timing

### Cell 5: Enhanced Visualization
**Before:**
- Only showed TD-Galerkin-SARIMA vs True
- Small plots (8×4)

**After:**
- **Four-algorithm comparison** per plot
- Larger plots (14×5) with grid
- Distinct colors/styles:
  - **TD-Galerkin-SARIMA** (purple, solid, diamonds)
  - **GARIMA-OLS** (blue, dashed, circles)
  - **GARIMA-Ridge** (red, dash-dot, squares)
  - **ARIMA** (green, dotted, triangles)
- Prints metrics below each plot

### Cell 6: Comprehensive Summary (NEW)
Added detailed analysis including:
- **Overall performance** (MAE, RMSE, Time) averaged across all datasets
- **Best algorithm** for each dataset type
- **Top 10 combinations** across all algorithms
- **Bar charts** comparing MAE, RMSE, and computational time
- **Performance by dataset type** analysis
- **TD vs TI comparison** (time-dependent vs time-independent)
- **Statistical summary** with describe()

---

## What Makes TD Different

### Time-Dependent vs Time-Independent

**TD-Galerkin-SARIMA:**
- Uses **Exponentially-Weighted Recursive Least Squares (EW-RLS)**
- **Online learning**: Updates parameters as new data arrives
- **Forgetting factor** `rho=0.9999` (high memory)
- **Regularization**: `lambda_beta=1e-1`, `lambda_alpha=1e-1`
- **Adaptive**: Can track changing patterns over time

**Time-Independent GARIMA:**
- Uses **batch learning**: Fits once on full training set
- **Static parameters**: Fixed throughout forecast horizon
- **Ridge/OLS**: Choice of regularization method
- **Efficient**: Faster for stable patterns

### Synthetic Datasets

**Four challenging datasets:**

1. **Noisy_ARMA**: ARMA(2,1) with noise
   - Tests linear autoregressive patterns
   - TD should excel at tracking AR coefficients

2. **Seasonal**: Sine wave + noise (period=20)
   - Tests seasonal pattern recognition
   - Both TD and TI should perform well

3. **Trend_AR**: Linear trend + AR(1)
   - Tests trend + autoregressive combination
   - TD should adapt to changing trend

4. **Nonlinear**: Logistic map + noise
   - Tests nonlinear chaotic dynamics
   - Most challenging for linear models

---

## Experiment Settings

### Parameters
```python
window = 15        # Small training window (tests adaptation)
horizon = 250      # Long forecast horizon (tests stability)
forecast_steps = 1 # One-step-ahead forecast
m_seasonal = 5     # Seasonal period

# Single parameter combination (focused comparison)
p_values = [1]     # AR order
q_values = [1]     # MA order  
P_values = [1]     # Seasonal AR order
Q_values = [1]     # Seasonal MA order
```

### TD Model Configuration
```python
TDRLSGalerkinSARIMA(
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 5),
    basis_functions="quadratic",
    rho=0.9999,           # High memory (slow forgetting)
    lambda_beta=1e-1,      # AR regularization
    lambda_alpha=1e-1,     # MA regularization
    standardize=False      # No standardization (TD handles scaling)
)
```

---

## Expected Results

### Typical Performance Ranking

**Accuracy (RMSE):**
1. **TD-Galerkin-SARIMA** (adaptive, tracks changes)
2. **GARIMA-Ridge** (regularized, handles noise)
3. **GARIMA-OLS** (standard linear)
4. **ARIMA** (limited basis functions)

**Speed:**
1. **GARIMA-OLS** (fastest)
2. **GARIMA-Ridge** (slightly slower)
3. **ARIMA** (moderate)
4. **TD-Galerkin-SARIMA** (slowest, online updates)

### When TD Excels

TD-Galerkin-SARIMA performs best when:
- **Non-stationary patterns** (changing coefficients)
- **Long forecast horizons** (adaptation helps)
- **Small training windows** (online learning advantage)
- **Complex dynamics** (logistic map, trend changes)

### When TI Excels

Time-independent models perform best when:
- **Stationary patterns** (stable coefficients)
- **Large training windows** (more data for batch learning)
- **Simple patterns** (ARMA, seasonal)
- **Speed requirements** (real-time applications)

---

## Key Insights

### TD vs TI Analysis

The notebook includes special analysis comparing:
- **TD advantage**: `((TI_RMSE - TD_RMSE) / TI_RMSE * 100)%`
- **TD time overhead**: `((TD_time - TI_time) / TI_time * 100)%`

### Dataset-Specific Performance

- **Noisy_ARMA**: TD should win (adaptive AR coefficients)
- **Seasonal**: TI might win (stable seasonal pattern)
- **Trend_AR**: TD should win (trend adaptation)
- **Nonlinear**: Mixed results (linear models struggle)

---

## How to Run

### Basic Run
```bash
jupyter notebook TD_GARIMA_sythetic.ipynb
# Run all cells
# Results will show comprehensive 4-way comparison
```

### Customize Parameters

**Change TD forgetting factor:**
```python
# In Cell 4, modify:
rho=0.99,        # Faster forgetting (more adaptive)
# vs
rho=0.9999,      # Slower forgetting (more stable)
```

**Adjust regularization:**
```python
lambda_beta=1e-2,   # More regularization
lambda_alpha=1e-2,   # More regularization
```

**Test different basis functions:**
```python
basis_functions="linear",     # Simpler
# vs
basis_functions="quadratic",  # Current
# vs
basis_functions=["linear", "quadratic", "sigmoid"]  # More complex
```

---

## Interpreting Results

### Look for:

1. **TD adaptation**: Does TD improve over time in long horizons?
2. **Pattern tracking**: Which datasets benefit most from TD?
3. **Speed tradeoffs**: Is TD accuracy worth the computational cost?
4. **Regularization impact**: How do ridge parameters affect TD?
5. **Basis function importance**: Does quadratic help over linear?

### Example Output:
```
Overall Performance:
                    Alg       MAE      RMSE  combo_sec
    TD-Galerkin-SARIMA  0.1234    0.1567      45.67
         GARIMA-Ridge   0.1345    0.1678      12.34
          GARIMA-OLS    0.1456    0.1789       8.90
               ARIMA    0.1567    0.1890      15.67

Time-Dependent vs Time-Independent Analysis:
TD-Galerkin-SARIMA average RMSE: 0.1567
Time-Independent GARIMA average RMSE: 0.1734
TD advantage: 9.6%

TD-Galerkin-SARIMA average time: 45.67s
Time-Independent GARIMA average time: 10.62s
TD time overhead: 330.0%
```

---

## Troubleshooting

### TD model fails to fit
→ Check if `rho` is too close to 1.0 (numerical issues)
→ Try `lambda_beta=1e-2`, `lambda_alpha=1e-2` (more regularization)

### TD performs worse than TI
→ Normal for stationary data! TD overhead not justified
→ Try `rho=0.95` for faster adaptation

### All models struggle on Nonlinear dataset
→ Expected! Logistic map is chaotic and nonlinear
→ Consider nonlinear basis functions or different models

### Memory issues with long horizons
→ Reduce `horizon` from 250 to 100
→ TD stores more state than TI models

---

## Advanced Analysis Ideas

### 1. Forgetting Factor Analysis
```python
# Test different rho values
rho_values = [0.95, 0.99, 0.995, 0.999, 0.9999]
# Compare performance vs adaptation speed
```

### 2. Online vs Batch Learning
```python
# Compare TD (online) vs retraining TI every N steps
# Test different retraining frequencies
```

### 3. Change Point Detection
```python
# Add artificial change points to datasets
# Test TD's ability to adapt vs TI's robustness
```

### 4. Computational Complexity
```python
# Profile TD vs TI for different window sizes
# Analyze O(n) vs O(n²) scaling
```

---

## Files Modified

- ✅ `TD_GARIMA_sythetic.ipynb` - Main experiment notebook (Cells 2, 4, 5, 6)
- ✅ `model.py` - Already has TDRLSGalerkinSARIMA implementation
- ✅ `TD_NOTEBOOK_UPDATE.md` - This guide

---

## Key Takeaways

✅ **Four algorithms** compared on synthetic datasets  
✅ **TD vs TI analysis** with adaptation metrics  
✅ **Challenging datasets** test different capabilities  
✅ **Comprehensive evaluation** (accuracy, speed, adaptation)  
✅ **Production-ready** for time series research  

The TD notebook is now ready for comprehensive evaluation of time-dependent vs time-independent forecasting! ⏰

---

## Next Steps

### For Research:
1. Run notebook with current settings (expect ~5-10 min runtime)
2. Analyze when TD adaptation helps vs hurts
3. Test different forgetting factors and regularization
4. Compare on real non-stationary data

### For Applications:
1. Test TD on financial data (changing volatility)
2. Apply to sensor data (equipment degradation)
3. Use for adaptive control systems
4. Compare with other online learning methods

### For Publication:
1. Use Cell 6 summary for results section
2. Use Cell 5 plots for forecast visualization
3. Highlight TD vs TI tradeoffs
4. Discuss computational complexity implications

---

## Comparison Across Notebooks

| Notebook | Data | Algorithms | Focus | Status |
|----------|------|------------|-------|--------|
| `GARIMA_sythetic.ipynb` | 4 synthetic | 3 (OLS, Ridge, ARIMA) | TI comparison | ✅ Updated |
| `GARIMA_GDP.ipynb` | Real GDP | 3 (OLS, Ridge, ARIMA) | TI on economic data | ✅ Updated |
| `GARIMA_SP.ipynb` | Real S&P500 | 3 (OLS, Ridge, ARIMA) | TI on financial data | ✅ Updated |
| `TD_GARIMA_sythetic.ipynb` | 4 synthetic | 4 (TD, OLS, Ridge, ARIMA) | TD vs TI comparison | ✅ Fixed |

All notebooks now provide comprehensive algorithm evaluation! 🎉
