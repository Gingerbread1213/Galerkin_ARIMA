#!/usr/bin/env python3
"""Refactor GARIMA_sythetic.ipynb to fair comparison (Phase 1 BIC + Phase 2 at best order)."""
import json

with open('GARIMA_sythetic.ipynb') as f:
    nb = json.load(f)

# 1. Update params cell (cell 2) - add matched_orders, forecast_steps
params_src = ''.join(nb['cells'][2]['source'])
if 'matched_orders' not in params_src:
    params_src += "\nforecast_steps = 1\nmatched_orders = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 1, 1)]\n"
    nb['cells'][2]['source'] = [l + '\n' for l in params_src.split('\n')[:-1]] + [params_src.split('\n')[-1] + '\n']

# 2. Replace main experiment cell (cell 3) with Phase 1 + Phase 2 + Phase 2b
main_cell = '''
algorithms = [
    {'name': 'GARIMA-OLS', 'use_ridge': False},
    {'name': 'GARIMA-Ridge', 'use_ridge': True, 'ridge_lambda_ar': 1.5, 'ridge_lambda_ma': 1.0,
     'ridge_weight_scheme': 'poly', 'ridge_eta': 1.5},
]

# ========== PHASE 1: BIC Model Selection (each algorithm selects its own best order per dataset) ==========
best_order_per_alg = {}  # (name, alg_name) -> (p,q,P,Q)
for name, series in datasets.items():
    train = series[:window]
    m = m_seasonal
    for alg_config in algorithms:
        alg_name = alg_config['name']
        best_bic = np.inf
        best_ord = (0, 0, 0, 0)
        for p, q, P, Q in tqdm(orders, desc=f"BIC {alg_name} {name}", leave=False):
            try:
                model = GalerkinSARIMA(train, order=(p, 0, q), seasonal_order=(P, 0, Q, m),
                    basis_functions=["quadratic", "sigmoid", "linear"], forecast_method="direct",
                    use_ridge=alg_config['use_ridge'], ridge_lambda_ar=alg_config.get('ridge_lambda_ar', 2.0),
                    ridge_lambda_ma=alg_config.get('ridge_lambda_ma', 2.0),
                    ridge_weight_scheme=alg_config.get('ridge_weight_scheme', 'none'),
                    ridge_eta=alg_config.get('ridge_eta', 1.0), standardize=True)
                model.fit(train)
                bic = model.bic()
                if bic < best_bic:
                    best_bic = bic
                    best_ord = (p, q, P, Q)
            except Exception:
                pass
        best_order_per_alg[(name, alg_name)] = best_ord
        print(f"  {name} {alg_name}: best (p,q,P,Q)={best_ord} (BIC={best_bic:.2f})")

# ========== PHASE 2: Rolling Forecast at Best Order ==========
tuned_results = []
first_run_preds = {}
first_run_pis = {}
for name, series in datasets.items():
    m = m_seasonal
    y_true = series[window:window + horizon]
    for alg_config in algorithms:
        alg_name = alg_config['name']
        p, q, P, Q = best_order_per_alg[(name, alg_name)]
        preds, iter_times = [], []
        combo_start = time.perf_counter()
        for i in range(window, window + horizon):
            t0 = time.perf_counter()
            train = series[:i]
            try:
                model = GalerkinSARIMA(train, order=(p, 0, q), seasonal_order=(P, 0, Q, m),
                    basis_functions=["quadratic", "sigmoid", "linear"], forecast_method="direct",
                    use_ridge=alg_config['use_ridge'], ridge_lambda_ar=alg_config.get('ridge_lambda_ar', 2.0),
                    ridge_lambda_ma=alg_config.get('ridge_lambda_ma', 2.0),
                    ridge_weight_scheme=alg_config.get('ridge_weight_scheme', 'none'),
                    ridge_eta=alg_config.get('ridge_eta', 1.0), standardize=True)
                model.fit(train)
                pv = model.forecast(steps=forecast_steps)
                pred = pv[-1] if np.ndim(pv) > 0 else float(pv)
                preds.append(float(pred))
            except Exception:
                preds.append(float(train[-1]))
            iter_times.append(time.perf_counter() - t0)
        preds = np.asarray(preds)
        combo_sec = time.perf_counter() - combo_start
        first_run_preds[(name, p, q, P, Q, alg_name)] = preds
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        err_var = np.var(y_true - preds)
        tuned_results.append({'Dataset': name, 'Alg': alg_name, 'Best_order': f"({p},{q},{P},{Q})",
            'MAE': mae, 'RMSE': rmse, 'Runtime_sec': combo_sec, 'Error_var': err_var})
        # Store PI for plotting (compute once)
        try:
            train_init = series[:window]
            model = GalerkinSARIMA(train_init, order=(p, 0, q), seasonal_order=(P, 0, Q, m),
                basis_functions=["quadratic", "sigmoid", "linear"], forecast_method="direct",
                use_ridge=alg_config['use_ridge'], ridge_lambda_ar=alg_config.get('ridge_lambda_ar', 2.0),
                ridge_lambda_ma=alg_config.get('ridge_lambda_ma', 2.0),
                ridge_weight_scheme=alg_config.get('ridge_weight_scheme', 'none'),
                ridge_eta=alg_config.get('ridge_eta', 1.0), standardize=True)
            roll = model.rolling_prediction_intervals(train_init, horizon, actuals=y_true, method='residual', alpha=0.05)
            first_run_pis[(name, p, q, P, Q, alg_name)] = roll
        except Exception:
            pass

# ========== PHASE 2b: Matched Structure ==========
matched_results = []
for name, series in datasets.items():
    m = m_seasonal
    y_true = series[window:window + horizon]
    for p, q, P, Q in matched_orders:
        for alg_config in algorithms:
            alg_name = alg_config['name']
            preds = []
            for i in range(window, window + horizon):
                train = series[:i]
                try:
                    model = GalerkinSARIMA(train, order=(p, 0, q), seasonal_order=(P, 0, Q, m),
                        basis_functions=["quadratic", "sigmoid", "linear"], forecast_method="direct",
                        use_ridge=alg_config['use_ridge'], ridge_lambda_ar=alg_config.get('ridge_lambda_ar', 2.0),
                        ridge_lambda_ma=alg_config.get('ridge_lambda_ma', 2.0),
                        ridge_weight_scheme=alg_config.get('ridge_weight_scheme', 'none'),
                        ridge_eta=alg_config.get('ridge_eta', 1.0), standardize=True)
                    model.fit(train)
                    pv = model.forecast(steps=1)
                    pred = pv[-1] if np.ndim(pv) > 0 else float(pv)
                    preds.append(float(pred))
                except Exception:
                    preds.append(float(train[-1]))
            preds = np.asarray(preds)
            mse = mean_squared_error(y_true, preds)
            matched_results.append({'p,q,P,Q': f"({p},{q},{P},{Q})", 'Dataset': name, 'Alg': alg_name, 'MSE': mse})

# Build df for compatibility
all_results = []
for r in tuned_results:
    p, q, P, Q = best_order_per_alg[(r['Dataset'], r['Alg'])]
    all_results.append([r['Dataset'], p, q, P, Q, 1, r['Alg'], r['MAE'], r['RMSE'],
        np.nan, np.nan, np.nan, r['Runtime_sec'], horizon, horizon/r['Runtime_sec']])
df = pd.DataFrame(all_results, columns=['Dataset','p','q','P','Q','Run','Alg','MAE','RMSE',
    'mean_iter_ms','median_iter_ms','max_iter_ms','combo_sec','num_iters','throughput_iters_per_sec'])
agg = df.groupby(['Dataset','p','q','P','Q','Alg'], as_index=False).mean(numeric_only=True)

print("\\n=== TABLE 1: Tuned Performance (each model at its BIC-best order) ===")
tuned_df = pd.DataFrame(tuned_results)
print(tuned_df[['Dataset','Alg','Best_order','MAE','RMSE','Runtime_sec']].to_string(index=False))
print("\\n=== TABLE 3: Stability (variance of forecast errors) ===")
print(tuned_df[['Dataset','Alg','Error_var']].to_string(index=False))
print("\\n=== TABLE 2: Matched Structure (same p,q for all - supplementary) ===")
match_df = pd.DataFrame(matched_results)
print(match_df.pivot_table(index=['Dataset','p,q,P,Q'], columns='Alg', values='MSE').to_string())
'''

lines = main_cell.strip().split('\n')
nb['cells'][3]['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
nb['cells'][3]['outputs'] = []

# 3. Update plotting cell (cell 4) - use best_order_per_alg, one plot per dataset
# Read current plotting cell to understand structure
plot_src = ''.join(nb['cells'][4]['source'])
# The synthetic plotting uses selected_combos from bic_results. We need to change to best_order_per_alg.
# Find and replace the plotting logic
if 'selected_combos' in plot_src:
    # Replace the selected_combos logic with best_order_per_alg
    old_block = """selected_combos = {}
if len(bic_results) > 0:
    bic_df = pd.DataFrame(bic_results)
    for name in datasets.keys():
        dataset_bic = bic_df[bic_df['Dataset'] == name]
        if not dataset_bic.empty:
            # Find best BIC per algorithm
            best_bic = dataset_bic.loc[dataset_bic.groupby('Alg')['BIC'].idxmin()]
            selected_combos[name] = best_bic[['p','q','P','Q','Alg']].to_dict('records')"""
    
    new_block = """# Use best order per (dataset, alg) from Phase 1
selected_combos = {}
for name in datasets.keys():
    selected_combos[name] = []
    for alg_name in ['GARIMA-OLS', 'GARIMA-Ridge']:
        if (name, alg_name) in best_order_per_alg:
            p, q, P, Q = best_order_per_alg[(name, alg_name)]
            selected_combos[name].append({'p': p, 'q': q, 'P': P, 'Q': Q, 'Alg': alg_name})"""
    
    if old_block in plot_src:
        plot_src = plot_src.replace(old_block, new_block)
    
    # Fix the loop - selected_combos[name] is list of dicts
    # The rest of the code might iterate over selected_combos[name] - need to check
    nb['cells'][4]['source'] = [l + '\n' for l in plot_src.split('\n')[:-1]] + [plot_src.split('\n')[-1] + '\n']

# 4. Update summary cell (cell 5) - add Tables 1,2,3 at top
summary_src = ''.join(nb['cells'][5]['source'])
if 'TABLE 1: Tuned' not in summary_src:
    insert = '''
# Paper-ready tables (from Phase 1+2)
print("="*80)
print("TABLE 1: Tuned Performance (synthetic, BIC-best per model)")
print("="*80)
print(tuned_df[['Dataset','Alg','Best_order','MAE','RMSE']].to_string(index=False))
print()
'''
    idx = summary_src.find('# Summary comparison')
    if idx > 0:
        summary_src = summary_src[:idx] + insert + summary_src[idx:]
    nb['cells'][5]['source'] = [l + '\n' for l in summary_src.split('\n')[:-1]] + [summary_src.split('\n')[-1] + '\n']

# 5. Add model summary cell at end
summary_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": """# ========== Model Summary: Variable Significance & Statistics ==========
for name, series in datasets.items():
    train = series[:window]
    m = m_seasonal
    for alg_config in algorithms:
        alg_name = alg_config['name']
        if (name, alg_name) not in best_order_per_alg:
            continue
        p, q, P, Q = best_order_per_alg[(name, alg_name)]
        print("=" * 80)
        print(f"MODEL SUMMARY: {name} | {alg_name} | order (p,q,P,Q)=({p},{q},{P},{Q})")
        print("=" * 80)
        model = GalerkinSARIMA(order=(p, 0, q), seasonal_order=(P, 0, Q, m),
            use_ridge=alg_config.get('use_ridge', False),
            ridge_lambda_ar=alg_config.get('ridge_lambda_ar', 1.0),
            ridge_lambda_ma=alg_config.get('ridge_lambda_ma', 1.0),
            ridge_weight_scheme=alg_config.get('ridge_weight_scheme', 'poly'),
            ridge_eta=alg_config.get('ridge_eta', 1.0))
        summary = model.summary(train, n_bootstrap=100, random_state=123)
        print(summary)
        print()
""".split('\n')
}
summary_cell['source'] = [l + '\n' for l in summary_cell['source'][:-1]] + [summary_cell['source'][-1]]
nb['cells'].append(summary_cell)

with open('GARIMA_sythetic.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Refactored GARIMA_sythetic.ipynb to fair comparison")