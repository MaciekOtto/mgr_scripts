"""
RF_oos_rw.py - Prognozy out-of-sample: Random Forest

Dla każdej spółki trenuje model Random Forest (100 drzew) do
prognozowania zmienności (realized variance, r²) na podstawie cech
opóźnionych (5 opóźnień r² oraz 5 opóźnień stóp zwrotu, LAG=5) w
schemacie rozszerzającego się okna (TRAIN_SIZE=1250, model
przetrenowywany okresowo co RETRAIN_EVERY kroków, nie po każdej
obserwacji). Obliczenia zrównoleglone, z checkpointami.

Wejście: dane1000stopy.xlsx
Wyjście: rf_prognozy_oos.parquet, rf_rmse_mae.xlsx,
         checkpoints_rf/

----------------------------------------------------------------------

RF_oos_rw.py - Out-of-sample forecasts: Random Forest

For each company, trains a Random Forest model (100 trees) to forecast
volatility (realized variance, r²) using lagged features (5 lags of r²
and 5 lags of returns, LAG=5) in an expanding-window scheme
(TRAIN_SIZE=1250, the model is periodically retrained every
RETRAIN_EVERY steps rather than at every observation). Computation is
parallelized, with checkpoints.

Input: dane1000stopy.xlsx
Output: rf_prognozy_oos.parquet, rf_rmse_mae.xlsx,
        checkpoints_rf/
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing as mp
import time
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ── Ustawienia ───

INPUT_FILE       = 'dane1000stopy.xlsx'
CHECKPOINT_DIR   = 'checkpoints_rf'
OUT_PARQUET      = 'rf_prognozy_oos.parquet'
OUT_EXCEL        = 'rf_rmse_mae.xlsx'

TRAIN_SIZE       = 1250
LAG              = 5
ROLLING_STD_W    = 20
RETRAIN_EVERY    = 10   

RF_PARAMS = dict(
    n_estimators     = 100,
    max_depth        = 10,
    max_features     = 'sqrt',
    min_samples_leaf = 5,
    random_state     = 42,
    n_jobs           = 1,  
)

N_WORKERS        = max(1, mp.cpu_count() - 1)
CHECKPOINT_EVERY = 50

# ── Budowanie features ──

def build_features(returns_arr):
    r  = returns_arr.astype(float)
    r2 = r ** 2
    n  = len(r)

    rows = []
    start = ROLLING_STD_W + LAG

    for t in range(start, n - 1):
        row = {}

        for k in range(1, LAG + 1):
            row[f'r2_lag_{k}']  = r2[t - k]
            row[f'ret_lag_{k}'] = r[t - k]

        row['rolling_std_20'] = float(np.std(r[t - ROLLING_STD_W:t]))
        row['target'] = r2[t + 1]
        row['t_idx']  = t
        rows.append(row)

    if not rows:
        return None, None, None

    df_f = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in df_f.columns if c not in ('target', 't_idx')]

    return df_f[feat_cols].values, df_f['target'].values, df_f['t_idx'].values


# ── Jedna spółka ──

def process_ticker(ticker, returns_arr, train_size, retrain_every, rf_params):
    X, y, t_idx = build_features(returns_arr)

    empty = {
        'ticker':    ticker,
        'forecasts': np.array([]),
        'realized':  np.array([]),
        'metrics':   {'Spółka': ticker, 'RMSE': np.nan,
                      'MAE': np.nan, 'N_valid': 0, 'Model': 'RF'},
        'feat_imp':  None,
    }

    if X is None:
        return empty

    test_mask  = t_idx >= train_size
    train_mask = ~test_mask

    if train_mask.sum() < 50 or test_mask.sum() < 5:
        return empty

    X_test  = X[test_mask]
    y_test  = y[test_mask]
    n_test  = len(X_test)

    forecasts  = np.full(n_test, np.nan)
    last_model = None
    feat_imp   = None

    for step in range(n_test):
        if step % retrain_every == 0:
            cut = train_mask.sum() + step
            model = RandomForestRegressor(**rf_params)
            model.fit(X[:cut], y[:cut])
            last_model = model
            feat_imp   = model.feature_importances_

        if last_model is not None:
            forecasts[step] = last_model.predict(X_test[step:step+1])[0]

    valid = ~np.isnan(forecasts)
    if valid.sum() > 5:
        rmse = float(np.sqrt(mean_squared_error(y_test[valid], forecasts[valid])))
        mae  = float(mean_absolute_error(y_test[valid], forecasts[valid]))
    else:
        rmse = mae = np.nan

    return {
        'ticker':    ticker,
        'forecasts': forecasts,
        'realized':  y_test,
        'metrics':   {'Spółka': ticker, 'RMSE': rmse, 'MAE': mae,
                      'N_valid': int(valid.sum()), 'Model': 'RF'},
        'feat_imp':  feat_imp,
    }


# ── Batch ──

def process_batch(args):
    (batch_idx, tickers_batch, data_dict,
     train_size, retrain_every, rf_params,
     checkpoint_dir, checkpoint_every) = args

    os.makedirs(checkpoint_dir, exist_ok=True)
    results = []

    for i, ticker in enumerate(tickers_batch):
        res = process_ticker(
            ticker,
            data_dict[ticker],
            train_size, retrain_every, rf_params,
        )
        results.append(res)

        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(tickers_batch):
            cp = os.path.join(checkpoint_dir,
                              f'batch_{batch_idx:03d}_step_{i:04d}.pkl')
            pd.to_pickle(results, cp)

    return results


# ── Wczytanie danych ──

def load_data():
    print(f"Wczytuję dane z: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    date_col = next((c for c in df.columns
                     if 'data' in c.lower() or 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'), errors='coerce')

    df.dropna(inplace=True)
    return df


def load_completed(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return set(), []
    done, results = set(), []
    for cp in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pkl'))):
        try:
            for r in pd.read_pickle(cp):
                if r['ticker'] not in done:
                    done.add(r['ticker'])
                    results.append(r)
        except Exception:
            pass
    return done, results


# ── Zapis ──

def save_outputs(all_results, test_size):
    metrics  = pd.DataFrame([r['metrics'] for r in all_results])

    # Prognozy 
    fc_dict  = {r['ticker']: r['forecasts'] for r in all_results
                if len(r['forecasts']) == test_size}
    pd.DataFrame(fc_dict).to_parquet(OUT_PARQUET)
    print(f"Prognozy: {OUT_PARQUET}  ({os.path.getsize(OUT_PARQUET)/1e6:.1f} MB)")

    # Feature importance — średnia po spółkach
    imps = [r['feat_imp'] for r in all_results if r['feat_imp'] is not None]
    fi_df = pd.DataFrame()
    if imps:
        arr = np.array(imps)
        # Nazwy features 
        n_feat = arr.shape[1]
        names = (
            [f'r2_lag_{k}'  for k in range(1, LAG+1)] +
            [f'ret_lag_{k}' for k in range(1, LAG+1)] +
            ['rolling_std_20']
        )
        fi_df = pd.DataFrame({
            'Feature':    names[:n_feat],
            'Importance': arr.mean(axis=0),
        }).sort_values('Importance', ascending=False)

    summary = pd.DataFrame([{
        'Model':        'RF',
        'RMSE_mediana': metrics['RMSE'].median(),
        'RMSE_srednia': metrics['RMSE'].mean(),
        'RMSE_std':     metrics['RMSE'].std(),
        'MAE_mediana':  metrics['MAE'].median(),
        'MAE_srednia':  metrics['MAE'].mean(),
        'N_spółek_OK':  metrics['RMSE'].notna().sum(),
    }])

    with pd.ExcelWriter(OUT_EXCEL, engine='openpyxl') as writer:
        metrics.to_excel(writer,  sheet_name='Surowe',       index=False)
        summary.to_excel(writer,  sheet_name='Podsumowanie', index=False)
        if not fi_df.empty:
            fi_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

    print(f"Metryki:  {OUT_EXCEL}")
    print(f"\n── RF Podsumowanie ──")
    print(f"RMSE mediana: {metrics['RMSE'].median():.8f}")
    print(f"MAE  mediana: {metrics['MAE'].median():.8f}")
    print(f"Spółek OK:    {metrics['RMSE'].notna().sum()}")


if __name__ == '__main__':
    t0 = time.time()

    df = load_data()
    tickers   = df.columns.tolist()
    TEST_SIZE = len(df) - TRAIN_SIZE

    print(f"Spółek: {len(tickers)} | Train: {TRAIN_SIZE} | Test: {TEST_SIZE}")
    print(f"Retrenowanie co {RETRAIN_EVERY} dni | Workery: {N_WORKERS}")

    done, existing = load_completed(CHECKPOINT_DIR)
    remaining = [t for t in tickers if t not in done]

    if done:
        print(f"Checkpointy: {len(done)} gotowych, {len(remaining)} pozostało")

    if not remaining:
        print("Wszystkie spółki gotowe...")
        save_outputs(existing, TEST_SIZE)
        exit()

    data_dict  = {t: df[t].values for t in remaining}
    batches    = [list(b) for b in np.array_split(remaining, N_WORKERS) if len(b)]
    batch_args = [
        (i, batch, data_dict, TRAIN_SIZE,
         RETRAIN_EVERY, RF_PARAMS, CHECKPOINT_DIR, CHECKPOINT_EVERY)
        for i, batch in enumerate(batches)
    ]

    print(f"\nStart RF rolling... (~3-5h)\n")
    with mp.Pool(processes=N_WORKERS) as pool:
        results_list = pool.map(process_batch, batch_args)

    all_results = existing + [r for batch in results_list for r in batch]
    save_outputs(all_results, TEST_SIZE)

    print(f"\nCzas całkowity: {(time.time()-t0)/3600:.2f} h")
    print("GOTOWE.")
