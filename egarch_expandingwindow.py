import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import os
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_EGARCH_narastajaco.xlsx'
output_folder = 'wykresy_egarch_2x3'
os.makedirs(output_folder, exist_ok=True)

try:
    print(f"Wczytuję dane z pliku: {input_file}...")
    df = pd.read_excel(input_file)
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data])
        df.set_index(col_data, inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df.dropna(inplace=True)

    years = sorted(df.index.year.unique())
    egarch_results = []

    for year in years:
        label = f"2020-{year}" if year > 2020 else "2020"
        subset = df[df.index.year <= year]
        print(f"\n>>> Szacowanie EGARCH dla okresu: {label}")
        
        for j, ticker in enumerate(df.columns):
            try:
                # SKALOWANIE: Mnożymy przez 100, aby uniknąć błędów logarytmu
                series = subset[ticker].dropna() * 100 
                if len(series) < 120: continue

                # Kluczowe: rescale=True pozwala bibliotece samej naprawić skalę danych
                am = arch_model(series, mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist='Normal', rescale=True)
                res = am.fit(disp='off', show_warning=False)
                
                p = res.params
                egarch_results.append({
                    'Okres': label,
                    'Spółka': ticker,
                    'Omega': p.get('omega', np.nan),
                    'Alpha': p.get('alpha[1]', np.nan),
                    'Gamma': p.get('gamma[1]', np.nan),
                    'Beta': p.get('beta[1]', np.nan),
                    'Status': 'OK'
                })
            except:
                egarch_results.append({'Okres': label, 'Spółka': ticker, 'Status': 'BŁĄD'})
            
            if (j + 1) % 250 == 0:
                print(f"   ...przetworzono {j + 1} spółek")

    df_results = pd.DataFrame(egarch_results)
    
    # SPRAWDZENIE: Czy w ogóle mamy jakieś wyniki OK?
    if df_results.empty or 'Omega' not in df_results.columns:
        print("🔴 BŁĄD: Żaden model EGARCH nie zakończył się sukcesem. Sprawdź dane wejściowe!")
    else:
        df_results.to_excel(output_excel, index=False)

        def plot_egarch_2x3(data, metric, title, filename, color):
            # Sprawdzamy czy kolumna istnieje i czy są w niej dane inne niż NaN
            if metric not in data.columns or data[data['Status'] == 'OK'][metric].dropna().empty:
                print(f"⚠️ Pominięto wykres {metric}: Brak poprawnych danych.")
                return

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            labels = data['Okres'].unique()

            for k, lbl in enumerate(labels):
                if k >= 6: break
                period_data = data[(data['Okres'] == lbl) & (data['Status'] == 'OK')][metric].dropna()
                
                if not period_data.empty:
                    # Filtracja skrajnych wartości (outliers) dla stabilności wykresu
                    q_low, q_high = period_data.quantile([0.05, 0.95])
                    period_data = period_data[(period_data >= q_low) & (period_data <= q_high)]
                    
                    if not period_data.empty:
                        sns.histplot(period_data, kde=True, ax=axes[k], color=color)
                        avg_val = period_data.mean()
                        axes[k].axvline(avg_val, color='red', linestyle='--', label=f'Śr: {avg_val:.3f}')
                        if metric == 'Gamma': axes[k].axvline(0, color='black', alpha=0.5)
                        axes[k].set_title(f"{lbl}")
                        axes[k].legend()
                
            plt.suptitle(f"Rozkład parametru EGARCH: {title}", fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_folder, filename), dpi=300)
            plt.close()

        print("\nGenerowanie wykresów...")
        plot_egarch_2x3(df_results, 'Omega', 'Omega', '2x3_EGARCH_Omega.png', 'gray')
        plot_egarch_2x3(df_results, 'Alpha', 'Alpha', '2x3_EGARCH_Alpha.png', 'skyblue')
        plot_egarch_2x3(df_results, 'Gamma', 'Gamma (Asymetria)', '2x3_EGARCH_Gamma.png', 'purple')
        plot_egarch_2x3(df_results, 'Beta', 'Beta', '2x3_EGARCH_Beta.png', 'salmon')

    print(f"\n✅ GOTOWE. Wyniki w {output_excel}")

except Exception as e:
    print(f"🔴 Błąd krytyczny: {e}")