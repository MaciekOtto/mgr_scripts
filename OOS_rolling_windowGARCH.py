"""
Rolling Window Out-of-Sample Forecasting — 4 modele GARCH
=========================================================
Dane:    dane1000stopy.xlsx  (kolumna z datą + 1000 kolumn ze stopami zwrotu)
Wynik:   garch_prognozy_oos.parquet  — prognozy σ² dla każdej spółki i modelu
         garch_rmse_mae.xlsx         — RMSE i MAE per spółka per model

Metodologia:
  - Okno treningowe: pierwsze TRAIN_SIZE dni (domyślnie 1000)
  - Okno testowe:    pozostałe TEST_SIZE dni (~500)
  - Rolling: co krok przesuwamy okno o 1 dzień i prognozujemy σ²_t+1
  - Zmienna celu (proxy zmienności): r²_t (realized variance)
"""

import pandas as pd
import numpy as np
from arch import arch_model
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ── Ustawienia ────────────────────────────────────────────────────────────────

INPUT_FILE  = 'dane1000stopy.xlsx'
OUT_PARQUET = 'garch_prognozy_oos.parquet'   # główny plik z prognozami
OUT_EXCEL   = 'garch_rmse_mae.xlsx'          # RMSE / MAE per spółka

TRAIN_SIZE  = 1000   # liczba dni w oknie treningowym
SCALE       = 100    # arch lubi dane przeskalowane (zwroty × 100)

# Modele do estymacji — (nazwa, vol, p, q, dodatkowe kwargs)
MODELS = [
    ('GARCH',     'Garch',  1, 1, {}),
    ('GJR-GARCH', 'Garch',  1, 1, {'o': 1}),          # o=1 → GJR
    ('EGARCH',    'EGARCH', 1, 1, {}),
    ('APARCH',    'APARCH', 1, 1, {}),
]

# ── Wczytanie danych ──────────────────────────────────────────────────────────

print(f"Wczytuję dane z: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)

# Wykrycie kolumny z datą
date_col = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

# Konwersja (obsługa przecinka jako separatora dziesiętnego)
for col in df.columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df.dropna(inplace=True)
tickers = df.columns.tolist()
T = len(df)
TEST_SIZE = T - TRAIN_SIZE

print(f"Spółek: {len(tickers)} | Dni łącznie: {T} | Train: {TRAIN_SIZE} | Test: {TEST_SIZE}")
assert TEST_SIZE > 0, "Za mało danych — zmniejsz TRAIN_SIZE"

# ── Rolling Window OOS ────────────────────────────────────────────────────────

# Słownik wynikowy: {model_name: DataFrame(index=daty_testu, columns=tickers)}
all_forecasts = {name: pd.DataFrame(index=df.index[TRAIN_SIZE:], columns=tickers, dtype=float)
                 for name, *_ in MODELS}

rmse_records = []   # lista słowników do zbiorczego RMSE/MAE

total_start = time.time()

for t_idx, ticker in enumerate(tickers):
    series_full = df[ticker].values * SCALE   # przeskalowanie
    realized    = (df[ticker].values) ** 2    # r²_t — proxy zmienności (NIE skalowane)

    ticker_forecasts = {name: np.full(TEST_SIZE, np.nan) for name, *_ in MODELS}

    for model_name, vol, p, q, kwargs in MODELS:
        forecasts_model = np.full(TEST_SIZE, np.nan)

        for step in range(TEST_SIZE):
            train_end = TRAIN_SIZE + step
            train_data = series_full[:train_end]

            try:
                am  = arch_model(train_data, mean='Constant', vol=vol, p=p, q=q, **kwargs)
                res = am.fit(disp='off', show_warning=False)

                # 1-krokowa prognoza wariancji (h.1 jest w jednostkach SCALE²)
                fc  = res.forecast(horizon=1, reindex=False)
                var_scaled = fc.variance.values[-1, 0]

                # Przeliczenie z powrotem na oryginalne jednostki (r²)
                forecasts_model[step] = var_scaled / (SCALE ** 2)

            except Exception:
                # Przy błędzie konwergencji zostawiamy NaN
                forecasts_model[step] = np.nan

        all_forecasts[model_name][ticker] = forecasts_model

        # ── Metryki per spółka per model ──────────────────────────────────────
        realized_test = realized[TRAIN_SIZE:]
        valid = ~np.isnan(forecasts_model)

        if valid.sum() > 10:
            rmse = np.sqrt(np.mean((forecasts_model[valid] - realized_test[valid]) ** 2))
            mae  = np.mean(np.abs(forecasts_model[valid] - realized_test[valid]))
        else:
            rmse = mae = np.nan

        rmse_records.append({
            'Spółka':  ticker,
            'Model':   model_name,
            'RMSE':    rmse,
            'MAE':     mae,
            'N_valid': int(valid.sum()),
        })

    # Postęp co 50 spółek
    if (t_idx + 1) % 50 == 0:
        elapsed = time.time() - total_start
        eta = elapsed / (t_idx + 1) * (len(tickers) - t_idx - 1)
        print(f"  [{t_idx+1}/{len(tickers)}]  czas: {elapsed/60:.1f} min  |  ETA: {eta/60:.1f} min")

# ── Zapis wyników ─────────────────────────────────────────────────────────────

print("\nZapisuję prognozy...")

# Parquet — szybki format kolumnowy, idealny do dalszej analizy
# Struktura: MultiIndex kolumn (model, ticker)
frames = []
for model_name, df_fc in all_forecasts.items():
    df_fc.columns = pd.MultiIndex.from_product([[model_name], df_fc.columns])
    frames.append(df_fc)

df_all = pd.concat(frames, axis=1)
df_all.to_parquet(OUT_PARQUET)
print(f"Prognozy zapisane: {OUT_PARQUET}  ({os.path.getsize(OUT_PARQUET)/1e6:.1f} MB)")

# Excel z RMSE / MAE
df_metrics = pd.DataFrame(rmse_records)

# Tabela przestawna — wiersze: spółki, kolumny: modele
pivot_rmse = df_metrics.pivot(index='Spółka', columns='Model', values='RMSE')
pivot_mae  = df_metrics.pivot(index='Spółka', columns='Model', values='MAE')

with pd.ExcelWriter(OUT_EXCEL, engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='Surowe', index=False)
    pivot_rmse.to_excel(writer, sheet_name='RMSE_pivot')
    pivot_mae.to_excel(writer, sheet_name='MAE_pivot')

    # Arkusz ze statystykami podsumowującymi
    summary_rows = []
    for model_name in [m[0] for m in MODELS]:
        rmse_vals = pivot_rmse[model_name].dropna()
        mae_vals  = pivot_mae[model_name].dropna()
        summary_rows.append({
            'Model':          model_name,
            'RMSE_mediana':   rmse_vals.median(),
            'RMSE_srednia':   rmse_vals.mean(),
            'RMSE_std':       rmse_vals.std(),
            'MAE_mediana':    mae_vals.median(),
            'MAE_srednia':    mae_vals.mean(),
            'N_spółek_OK':    len(rmse_vals),
        })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Podsumowanie', index=False)

print(f"Metryki zapisane:  {OUT_EXCEL}")
print(f"\nCałkowity czas: {(time.time()-total_start)/60:.1f} min")
print("GOTOWE.")

# ── Szybki podgląd wyników ────────────────────────────────────────────────────

print("\n── Podsumowanie RMSE (mediana po spółkach) ──")
print(pivot_rmse.median().round(8).to_string())
print("\n── Podsumowanie MAE (mediana po spółkach) ──")
print(pivot_mae.median().round(8).to_string())
