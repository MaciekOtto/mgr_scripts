"""
Test Diebolda-Mariano (1995) przeprowadzony niezależnie dla każdej spółki,
dla wybranych par modeli (np. GARCH vs Random Forest, GARCH vs LSTM,
GJR-GARCH vs SVR itd.). Funkcja straty: błąd kwadratowy (squared error)
względem zrealizowanej wariancji (r^2).

Wymagania co do plików:
- Realized variance liczona jest z `dane1000stopy.xlsx` (log-stopy zwrotu)
  jako r_t^2, dla ostatnich TEST_SIZE obserwacji każdej spółki.
- Prognozy modeli: jeden plik parquet na model, np.:
    garch_forecasts.parquet, gjrgarch_forecasts.parquet,
    rf_forecasts.parquet, lstm_forecasts.parquet, svr_forecasts.parquet
  Skrypt akceptuje DWA możliwe layouty pliku parquet i wykrywa go
  automatycznie:
    (A) "long"  - kolumny: ticker, date, forecast
    (B) "wide"  - wiersze = daty testowe, kolumny = tickery (TICKER_Close
                  lub po prostu TICKER), tak jak w dane1000stopy.xlsx

Wynik: zbiorczy DataFrame z liczbą/procentem spółek, dla których dany
model istotnie przewyższa drugi (p-value < 0.05), zapisany do Excela.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ============================== KONFIGURACJA ==============================

RETURNS_FILE = "dane1000stopy.xlsx"
TEST_SIZE = 250  # zgodnie ze schematem expanding window TRAIN=1250/TEST=250
ALPHA = 0.05

# Pliki z prognozami - dopasuj nazwy do swoich plików
FORECAST_FILES = {
    "GARCH": "garch_forecasts.parquet",
    "GJR-GARCH": "gjrgarch_forecasts.parquet",
    "RF": "rf_forecasts.parquet",
    "LSTM": "lstm_forecasts.parquet",
    "SVR": "svr_forecasts.parquet",
}

# Pary modeli do porównania (model_1, model_2)
# Dodatnia statystyka DM => model_1 ma WYŻSZY błąd (czyli model_2 lepszy)
# (zgodnie z konwencją Diebold-Mariano: d_t = L(e1_t) - L(e2_t))
MODEL_PAIRS = [
    ("GARCH", "RF"),
    ("GARCH", "LSTM"),
    ("GARCH", "SVR"),
    ("GJR-GARCH", "RF"),
    ("GJR-GARCH", "LSTM"),
    ("GJR-GARCH", "SVR"),
]

OUTPUT_FILE = "wyniki_diebold_mariano.xlsx"

# ============================================================================


def load_realized_variance(returns_file: str, test_size: int) -> pd.DataFrame:
    """Wczytuje log-stopy zwrotu i liczy realized variance (r^2) dla
    ostatnich `test_size` obserwacji każdej spółki."""
    df = pd.read_excel(returns_file, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    # ujednolicenie nazw kolumn: usuń sufiks _Close jeśli występuje
    df.columns = [c.replace("_Close", "") for c in df.columns]

    rv = df.iloc[-test_size:] ** 2
    return rv  # wiersze = daty testowe, kolumny = tickery


def load_forecast_wide(path: str, tickers: list, test_size: int) -> pd.DataFrame:
    """Próbuje wczytać plik parquet w formacie 'wide' (daty x tickery)
    lub 'long' (ticker, date, forecast) i sprowadza do formatu wide,
    z wierszami ograniczonymi do ostatnich `test_size` obserwacji."""
    raw = pd.read_parquet(path)

    cols_lower = [str(c).lower() for c in raw.columns]
    is_long = {"ticker", "forecast"}.issubset(set(cols_lower)) or \
              {"symbol", "forecast"}.issubset(set(cols_lower))

    if is_long:
        # normalizacja nazw kolumn
        rename_map = {}
        for c in raw.columns:
            cl = str(c).lower()
            if cl in ("ticker", "symbol"):
                rename_map[c] = "ticker"
            elif cl in ("date", "data"):
                rename_map[c] = "date"
            elif cl in ("forecast", "prediction", "pred", "yhat"):
                rename_map[c] = "forecast"
        raw = raw.rename(columns=rename_map)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        wide = raw.pivot(index="date", columns="ticker", values="forecast")
        wide = wide.sort_index()
    else:
        # format wide - zakładamy że indeks lub pierwsza kolumna to data
        if not isinstance(raw.index, pd.DatetimeIndex):
            first_col = raw.columns[0]
            raw = raw.set_index(first_col)
        raw.index = pd.to_datetime(raw.index, errors="coerce")
        raw.columns = [str(c).replace("_Close", "").replace("_forecast", "") for c in raw.columns]
        wide = raw.sort_index()

    wide = wide.iloc[-test_size:]
    # zachowaj tylko tickery, które faktycznie analizujemy
    common = [t for t in wide.columns if t in tickers]
    return wide[common]


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, loss: str = "squared"):
    """
    Klasyczny test Diebolda-Mariano (bez korekty Harvey-Leybourne-Newbold,
    h=1, brak autokorelacji rzędu wyższego niż 0 zakładany dla prognoz
    jednookresowych - long-run variance estymowana metodą Newey-West
    z lagiem 0 redukuje się do wariancji próbkowej).

    e1, e2: błędy prognozy (forecast - realized) dla modelu 1 i 2.
    loss: 'squared' lub 'absolute'.

    Zwraca: (statystyka DM, p-value)
    Dodatnia statystyka => model 1 ma wyższą stratę (czyli model 2 lepszy).
    """
    if loss == "squared":
        l1 = e1 ** 2
        l2 = e2 ** 2
    elif loss == "absolute":
        l1 = np.abs(e1)
        l2 = np.abs(e2)
    else:
        raise ValueError("loss musi być 'squared' lub 'absolute'")

    d = l1 - l2
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    if d_var == 0 or np.isnan(d_var):
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

    return dm_stat, p_value


def main():
    rv = load_realized_variance(RETURNS_FILE, TEST_SIZE)
    tickers = list(rv.columns)
    print(f"Wczytano realized variance dla {len(tickers)} spółek, "
          f"{rv.shape[0]} obserwacji testowych.")

    # wczytanie wszystkich potrzebnych modeli
    needed_models = sorted({m for pair in MODEL_PAIRS for m in pair})
    forecasts = {}
    for model in needed_models:
        path = FORECAST_FILES[model]
        if not Path(path).exists():
            print(f"UWAGA: plik {path} nie istnieje - pomijam model {model}")
            continue
        forecasts[model] = load_forecast_wide(path, tickers, TEST_SIZE)
        print(f"Wczytano prognozy dla modelu {model}: "
              f"{forecasts[model].shape[1]} spółek, {forecasts[model].shape[0]} obs.")

    results = []

    for model_1, model_2 in MODEL_PAIRS:
        if model_1 not in forecasts or model_2 not in forecasts:
            print(f"Pomijam parę {model_1} vs {model_2} - brak danych.")
            continue

        f1 = forecasts[model_1]
        f2 = forecasts[model_2]

        # spółki dostępne we wszystkich trzech zbiorach (rv, f1, f2)
        common_tickers = sorted(set(rv.columns) & set(f1.columns) & set(f2.columns))

        n_significant_1_better = 0  # model_1 istotnie lepszy (niższa strata)
        n_significant_2_better = 0  # model_2 istotnie lepszy
        n_not_significant = 0
        n_skipped = 0

        for ticker in common_tickers:
            actual = rv[ticker].values
            pred1 = f1[ticker].reindex(rv.index).values
            pred2 = f2[ticker].reindex(rv.index).values

            mask = ~(np.isnan(actual) | np.isnan(pred1) | np.isnan(pred2))
            if mask.sum() < 30:  # zbyt mało wspólnych obserwacji
                n_skipped += 1
                continue

            e1 = pred1[mask] - actual[mask]
            e2 = pred2[mask] - actual[mask]

            dm_stat, p_value = diebold_mariano(e1, e2, loss="squared")

            if np.isnan(p_value):
                n_skipped += 1
                continue

            results.append({
                "model_1": model_1,
                "model_2": model_2,
                "ticker": ticker,
                "dm_stat": dm_stat,
                "p_value": p_value,
                "n_obs": mask.sum(),
            })

            if p_value < ALPHA:
                if dm_stat < 0:
                    n_significant_1_better += 1  # model_1 ma niższą stratę
                else:
                    n_significant_2_better += 1
            else:
                n_not_significant += 1

        total = n_significant_1_better + n_significant_2_better + n_not_significant
        print(f"\n{model_1} vs {model_2} (n={total}, pominięto={n_skipped}):")
        if total:
            print(f"  {model_1} istotnie lepszy: {n_significant_1_better} "
                  f"({100*n_significant_1_better/total:.1f}%)")
            print(f"  {model_2} istotnie lepszy: {n_significant_2_better} "
                  f"({100*n_significant_2_better/total:.1f}%)")
            print(f"  brak istotnej różnicy:    {n_not_significant} "
                  f"({100*n_not_significant/total:.1f}%)")
        else:
            print("  brak danych")

    # zapis szczegółowych wyników (firm-by-firm) do Excela
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            results_df.to_excel(writer, sheet_name="szczegoly", index=False)

            # tabela podsumowująca
            summary_rows = []
            for (m1, m2), grp in results_df.groupby(["model_1", "model_2"]):
                n = len(grp)
                sig_1 = ((grp["p_value"] < ALPHA) & (grp["dm_stat"] < 0)).sum()
                sig_2 = ((grp["p_value"] < ALPHA) & (grp["dm_stat"] > 0)).sum()
                not_sig = n - sig_1 - sig_2
                summary_rows.append({
                    "model_1": m1,
                    "model_2": m2,
                    "n_spolek": n,
                    f"{m1}_istotnie_lepszy": sig_1,
                    f"{m1}_istotnie_lepszy_%": round(100 * sig_1 / n, 1),
                    f"{m2}_istotnie_lepszy": sig_2,
                    f"{m2}_istotnie_lepszy_%": round(100 * sig_2 / n, 1),
                    "brak_istotnej_roznicy": not_sig,
                    "brak_istotnej_roznicy_%": round(100 * not_sig / n, 1),
                })
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="podsumowanie", index=False)

        print(f"\nZapisano wyniki do: {OUTPUT_FILE}")
    else:
        print("\nNie uzyskano żadnych wyników - sprawdź pliki wejściowe.")


if __name__ == "__main__":
    main()
