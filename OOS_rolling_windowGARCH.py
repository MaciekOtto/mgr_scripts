import pandas as pd
import numpy as np
from arch import arch_model
import time
import os
import glob
import warnings
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# ── Ustawienia ─
INPUT_FILE       = 'dane1000stopy.xlsx'
CHECKPOINT_DIR   = 'checkpoints_garch'   
OUT_PARQUET      = 'garch_prognozy_oos.parquet'
OUT_EXCEL        = 'garch_rmse_mae.xlsx'

TRAIN_SIZE       = 1250  
SCALE            = 100
CHECKPOINT_EVERY = 50  
N_WORKERS        = None  # wszystkie rdzenie 

MODELS = [
    ('GARCH',     'Garch',  1, 1, {}),
    ('GJR-GARCH', 'Garch',  1, 1, {'o': 1}),
    ('EGARCH',    'EGARCH', 1, 1, {}),
    ('APARCH',    'APARCH', 1, 1, {}),
]

# Funkcja dla jednej spółki

def process_ticker(args):
    ticker, series_raw, train_size, scale, models = args
    series_full = series_raw * scale
    realized    = series_raw ** 2
    test_size   = len(series_raw) - train_size
    result = {'ticker': ticker, 'forecasts': {}, 'metrics': []}
    for model_name, vol, p, q, kwargs in models:
        forecasts = np.full(test_size, np.nan)
        for step in range(test_size):
            train_data = series_full[:train_size + step]
            try:
                am  = arch_model(train_data, mean='Constant', vol=vol,
                                 p=p, q=q, **kwargs)
                res = am.fit(disp='off', show_warning=False)
                fc  = res.forecast(horizon=1, reindex=False)
                forecasts[step] = fc.variance.values[-1, 0] / (scale ** 2)
            except Exception:
                forecasts[step] = np.nan
        result['forecasts'][model_name] = forecasts
        realized_test = realized[train_size:]
        valid = ~np.isnan(forecasts)
        if valid.sum() > 10:
            rmse = float(np.sqrt(np.mean((forecasts[valid] - realized_test[valid]) ** 2)))
            mae  = float(np.mean(np.abs(forecasts[valid] - realized_test[valid])))
        else:
            rmse = mae = np.nan
        result['metrics'].append({
            'Spółka':  ticker,
            'Model':   model_name,
            'RMSE':    rmse,
            'MAE':     mae,
            'N_valid': int(valid.sum()),
        })
    return result

def process_batch(batch_args):
    batch_idx, tickers_batch, data_dict, train_size, scale, models, \
        checkpoint_dir, checkpoint_every = batch_args
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_results = []
    for i, ticker in enumerate(tickers_batch):
        series_raw = data_dict[ticker]
        res = process_ticker((ticker, series_raw, train_size, scale, models))
        batch_results.append(res)
        # Checkpoint co N spółek
        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(tickers_batch):
            cp_path = os.path.join(checkpoint_dir,
                                   f'batch_{batch_idx:03d}_up_to_{i:04d}.pkl')
            pd.to_pickle(batch_results, cp_path)
    return batch_results

# ── Wczytanie danych ────

def load_data(input_file):
    print(f"Wczytuję dane z: {input_file}")
    df = pd.read_excel(input_file)
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


# ── Wykrywanie ukończonych spółek z checkpointów ──

def load_completed_tickers(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return set(), []
    completed = set()
    all_results = []
    cp_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pkl')))
    for cp_file in cp_files:
        try:
            batch = pd.read_pickle(cp_file)
            for res in batch:
                if res['ticker'] not in completed:
                    completed.add(res['ticker'])
                    all_results.append(res)
        except Exception as e:
            print(f"  Uwaga: nie mogę wczytać {cp_file}: {e}")
    return completed, all_results

# ── Scalanie wyników ───

def build_outputs(all_results, test_index, out_parquet, out_excel):
    model_names = [m[0] for m in MODELS]
    forecasts_dict = {m: {} for m in model_names}
    metrics_list   = []
    for res in all_results:
        ticker = res['ticker']
        for model_name, fc_array in res['forecasts'].items():
            forecasts_dict[model_name][ticker] = fc_array
        metrics_list.extend(res['metrics'])
    frames = []
    for model_name in model_names:
        df_fc = pd.DataFrame(forecasts_dict[model_name], index=test_index)
        df_fc.columns = pd.MultiIndex.from_product([[model_name], df_fc.columns])
        frames.append(df_fc)
    df_all = pd.concat(frames, axis=1)
    df_all.to_parquet(out_parquet)
    print(f"Prognozy zapisane: {out_parquet}  "
          f"({os.path.getsize(out_parquet)/1e6:.1f} MB)")


    df_metrics   = pd.DataFrame(metrics_list)
    pivot_rmse   = df_metrics.pivot(index='Spółka', columns='Model', values='RMSE')
    pivot_mae    = df_metrics.pivot(index='Spółka', columns='Model', values='MAE')
    summary_rows = []
    for mn in model_names:
        rv = pivot_rmse[mn].dropna()
        mv = pivot_mae[mn].dropna()
        summary_rows.append({
            'Model':        mn,
            'RMSE_mediana': rv.median(),
            'RMSE_srednia': rv.mean(),
            'RMSE_std':     rv.std(),
            'MAE_mediana':  mv.median(),
            'MAE_srednia':  mv.mean(),
            'N_spółek_OK':  len(rv),
        })

    with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
        df_metrics.to_excel(writer,   sheet_name='Surowe',       index=False)
        pivot_rmse.to_excel(writer,   sheet_name='RMSE_pivot')
        pivot_mae.to_excel(writer,    sheet_name='MAE_pivot')
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name='Podsumowanie', index=False)

    print(f"Metryki zapisane:  {out_excel}")

    print("\n── Podsumowanie RMSE (mediana po spółkach) ──")
    print(pivot_rmse.median().round(8).to_string())
    print("\n── Podsumowanie MAE (mediana po spółkach) ──")
    print(pivot_mae.median().round(8).to_string())

if __name__ == '__main__':

    total_start = time.time()

    # 1. Dane
    df      = load_data(INPUT_FILE)
    tickers = df.columns.tolist()
    T       = len(df)
    TEST_SIZE = T - TRAIN_SIZE
    test_index = df.index[TRAIN_SIZE:]

    print(f"Spółek: {len(tickers)} | Dni: {T} | "
          f"Train: {TRAIN_SIZE} | Test: {TEST_SIZE}")

    # 2. Checkpointy
    completed_tickers, existing_results = load_completed_tickers(CHECKPOINT_DIR)

    if completed_tickers:
        print(f"\nZnaleziono checkpointy: {len(completed_tickers)} spółek już gotowych.")
        remaining = [t for t in tickers if t not in completed_tickers]
        print(f"Pozostało do policzenia: {len(remaining)} spółek.")
    else:
        print("\nBrak checkpointów - od nowa.")
        remaining = tickers

    if not remaining:
        print("Wszystkie spółki już policzone! Scalanie wyników...")
        build_outputs(existing_results, test_index, OUT_PARQUET, OUT_EXCEL)
        exit()

    # 3. Słownik danych 
    data_dict = {t: df[t].values for t in remaining}

    # 4. Podział na batche
    n_workers = N_WORKERS or max(1, mp.cpu_count() - 1)
    print(f"\nLiczba workerów: {n_workers} (z {mp.cpu_count()} dostępnych rdzeni)")

    # Dzieli remaining na n_workers równych porcji
    batches = np.array_split(remaining, n_workers)
    batches = [list(b) for b in batches if len(b) > 0]

    batch_args = [
        (idx, batch, data_dict, TRAIN_SIZE, SCALE, MODELS,
         CHECKPOINT_DIR, CHECKPOINT_EVERY)
        for idx, batch in enumerate(batches)
    ]

    print(f"Podział: {len(batches)} batchy, "
          f"~{len(remaining)//len(batches)} spółek na batch")
    print(f"\nStart obliczeń... \n")

    # 5. Równoległe obliczenia
    with mp.Pool(processes=n_workers) as pool:
        batch_results_list = pool.map(process_batch, batch_args)

    # Spłaszcz wyniki
    new_results = [res for batch in batch_results_list for res in batch]
    all_results = existing_results + new_results

    print(f"\nObliczenia zakończone. Łącznie spółek: {len(all_results)}")

    # 6. Zapis końcowy
    build_outputs(all_results, test_index, OUT_PARQUET, OUT_EXCEL)

    elapsed = time.time() - total_start
    print(f"\nCałkowity czas: {elapsed/3600:.2f} godz.")
    print("GOTOWE.")
