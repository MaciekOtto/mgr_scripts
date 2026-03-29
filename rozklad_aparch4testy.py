import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import scipy.stats as stats
import os
import time
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_APARCH_4testy.xlsx'
output_folder = 'analiza_statystyczna_aparch4testy1'
os.makedirs(output_folder, exist_ok=True)

try:
    print(f"Wczytuję dane z pliku: {input_file}...")
    df_returns = pd.read_excel(input_file)
    
    col_data = next((c for c in df_returns.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
        
    df_returns = df_returns.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df_returns.dropna(how='all', inplace=True)

    wyniki_raw = []
    start_time = time.time()

    print(f"Szacowanie modeli APARCH(1,1,1) dla {len(df_returns.columns)} spółek...")

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].dropna()
        if len(series) < 150: continue

        try:
            # Model APARCH: o=1 (Asymetria), p=1, q=1
            am = arch_model(series * 100, mean='Constant', vol='APARCH', p=1, o=1, q=1, rescale=True)
            res = am.fit(disp='off', show_warning=False)

            p = res.params
            wyniki_raw.append({
                'Spółka': ticker,
                'Omega (APARCH)': p.get('omega'),
                'Alpha (APARCH)': p.get('alpha[1]'),
                'Gamma (APARCH)': p.get('gamma[1]'),
                'Beta (APARCH)': p.get('beta[1]'),
                'Delta (APARCH)': p.get('delta'),
                'AIC': res.aic,
                'Status': 'OK'
            })
            
        except:
            wyniki_raw.append({'Spółka': ticker, 'Status': 'BŁĄD'})

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} spółek...")

    df_res = pd.DataFrame(wyniki_raw)
    df_ok = df_res[df_res['Status'] == 'OK'].copy()

    ## --- ANALIZA STATYSTYCZNA (4 TESTY K-S) ---
    print("\nPrzeprowadzam testy p-value dla rozkładów: Normalny, t-Studenta, Log-normalny, F...")
    
    # Parametry o które prosiłeś
    parametry = ['Alpha (APARCH)', 'Beta (APARCH)', 'Omega (APARCH)', 'Gamma (APARCH)', 'Delta (APARCH)']
    raport_stat = []

    for param in parametry:
        data = df_ok[param].dropna()
        # Usuwanie outlierów (1% - 99%) dla stabilności testów
        q_low, q_high = data.quantile([0.01, 0.99])
        data_clean = data[(data > q_low) & (data < q_high)]

        # Słownik na wyniki p-value
        p_values = {}

        # 1. Test Rozkładu Normalnego
        params_norm = stats.norm.fit(data_clean)
        _, p_values['p-val Normalny'] = stats.kstest(data_clean, 'norm', args=params_norm)

        # 2. Test Rozkładu t-Studenta
        params_t = stats.t.fit(data_clean)
        _, p_values['p-val t-Student'] = stats.kstest(data_clean, 't', args=params_t)

        # 3. Test Rozkładu Log-normalnego (dla danych > 0)
        # Dodajemy mały epsilon, by uniknąć błędów przy zerach
        data_pos = data_clean + 0.00001 if data_clean.min() <= 0 else data_clean
        params_log = stats.lognorm.fit(data_pos)
        _, p_values['p-val Log-normalny'] = stats.kstest(data_pos, 'lognorm', args=params_log)

        # 4. Test Rozkładu F (Snedecora)
        # Rozkład F wymaga dwóch parametrów stopni swobody i jest zdefiniowany dla x > 0
        params_f = stats.f.fit(data_pos)
        _, p_values['p-val F'] = stats.kstest(data_pos, 'f', args=params_f)

        # Dodanie wyników do raportu
        res_row = {
            'Parametr': param,
            'Średnia': data_clean.mean(),
            'Skośność': data_clean.skew(),
            'Kurtoza': data_clean.kurtosis()
        }
        res_row.update(p_values)
        raport_stat.append(res_row)

        # --- WYBÓR NAJLEPSZEGO ROZKŁADU (Najwyższe p-value) ---
        # Tworzymy mapowanie nazw p-value na kody scipy.stats
        mapping = {
            'p-val Normalny': ('norm', 'Normalny', 'blue'),
            'p-val t-Student': ('t', 't-Student', 'red'),
            'p-val Log-normalny': ('lognorm', 'Log-normalny', 'green'),
            'p-val F': ('f', 'F-Snedecora', 'orange')
        }
        
        # Znajdujemy klucz z najwyższym p-value
        best_key = max(p_values, key=p_values.get)
        best_dist_code, best_name, best_color = mapping[best_key]
        
        res_row['Rekomendowany Rozkład'] = best_name
        res_row['Najwyższe p-value'] = p_values[best_key]

        # --- WYKRES Z NAJLEPSZĄ KRZYWĄ ---
        plt.figure(figsize=(10, 6))
        # Histogram danych
        sns.histplot(data_clean, kde=False, bins=40,stat="density", color='lightgray', label='Dane empiryczne')
        
        # Dopasowanie i rysowanie krzywej
        dist_func = getattr(stats, best_dist_code)
        
        # Specjalna obsługa danych dla log-normalnego i F (muszą być > 0)
        if best_dist_code in ['lognorm', 'f']:
            d_plot = data_clean + 0.00001 if data_clean.min() <= 0 else data_clean
            params = dist_func.fit(d_plot)
            x = np.linspace(d_plot.min(), d_plot.max(), 100)
            plt.plot(x, dist_func.pdf(x, *params), color=best_color, lw=2.5, 
                     label=f'Najlepsze dopasowanie: {best_name}\n(p-val: {p_values[best_key]:.4f})')
        else:
            params = dist_func.fit(data_clean)
            x = np.linspace(data_clean.min(), data_clean.max(), 100)
            plt.plot(x, dist_func.pdf(x, *params), color=best_color, lw=2.5, 
                     label=f'Najlepsze dopasowanie: {best_name}\n(p-val: {p_values[best_key]:.4f})')

        plt.title(f'APARCH: Rozkład parametru {param} (Najlepsze przybliżenie)')
        plt.xlabel('Wartość parametru')
        plt.ylabel('Gęstość')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'APARCH_BEST_{param}.png'))
        plt.close()

    # Zapis do Excela
    df_stat = pd.DataFrame(raport_stat)
    with pd.ExcelWriter(output_excel) as writer:
        df_ok.to_excel(writer, sheet_name='Parametry_APARCH', index=False)
        df_stat.to_excel(writer, sheet_name='Testy_p_value', index=False)

    print(f"\n✅ GOTOWE. Wyniki zapisano w: {output_excel}")
    print("\nPodsumowanie p-values (H0: dane pochodzą z danego rozkładu):")
    print(df_stat[['Parametr', 'p-val Normalny', 'p-val t-Student', 'p-val Log-normalny', 'p-val F']])

except Exception as e:
    print(f"🔴 Wystąpił błąd: {e}")