import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import os
import time
import warnings

# Wyciszenie ostrzeżeń optymalizatora
warnings.filterwarnings('ignore')

# --- Ustawienia ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_GJRGARCH_narastajaco.xlsx'
output_folder = 'wykresy_gjrgarch_2x3'
os.makedirs(output_folder, exist_ok=True)

print(f"Wczytuję dane z pliku: {input_file}...")

try:
    df = pd.read_excel(input_file)
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data])
        df.set_index(col_data, inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df.dropna(how='all', inplace=True)

    years = sorted(df.index.year.unique())
    results = []

    # 1. Główna pętla po okresach narastających (Expanding Window)
    for year in years:
        label = f"2020-{year}" if year > 2020 else "2020"
        subset = df[df.index.year <= year]
        print(f"\n>>> Szacowanie GJR-GARCH dla okresu: {label} (Obs: {len(subset)})")
        
        for j, ticker in enumerate(df.columns):
            try:
                series = subset[ticker].dropna()
                if len(series) < 150: continue # Minimum danych dla GJR

                # Model GJR-GARCH(1,1,1)
                # Skalowanie * 100 pomaga uniknąć błędu macierzy osobliwej
                am = arch_model(series * 100, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='Normal')
                res = am.fit(disp='off')
                
                p = res.params
                a = p.get('alpha[1]', 0)
                g = p.get('gamma[1]', 0)
                b = p.get('beta[1]', 0)
                
                # Persystencja w GJR-GARCH przy rozkładzie normalnym
                persystencja = a + b + 0.5 * g
                
                results.append({
                    'Okres': label,
                    'Spółka': ticker,
                    'Omega': p.get('omega'),
                    'Alpha': a,
                    'Gamma': g,
                    'Beta': b,
                    'Persystencja': persystencja,
                    'Status': 'OK'
                })
            except:
                results.append({'Okres': label, 'Spółka': ticker, 'Status': 'BŁĄD'})
            
            if (j + 1) % 250 == 0:
                print(f"   ...przetworzono {j + 1} spółek")

    df_results = pd.DataFrame(results)
    df_results.to_excel(output_excel, index=False)

    # 2. Funkcja do generowania zestawień 2x3 (Poprawiona)
    def plot_gjrgarch_2x3(data, metric, title, filename, color):
        valid_data = data[data['Status'] == 'OK'].copy()
        if valid_data.empty: return

        # Dynamiczny dobór liczby podwykresów (max 6)
        unique_periods = valid_data['Okres'].unique()
        num_plots = min(len(unique_periods), 6)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for k in range(len(axes)):
            if k < num_plots:
                lbl = unique_periods[k]
                period_series = valid_data[valid_data['Okres'] == lbl][metric]
                
                # Usuwanie skrajnych outlierów dla czytelności wykresu (1-99 percentyl)
                q_low, q_high = period_series.quantile([0.01, 0.99])
                filtered_series = period_series[(period_series > q_low) & (period_series < q_high)]
                
                sns.histplot(filtered_series, kde=True, ax=axes[k], color=color)
                
                avg_val = filtered_series.mean()
                axes[k].axvline(avg_val, color='red', linestyle='--', label=f'Śr: {avg_val:.3f}')
                if metric == 'Gamma': axes[k].axvline(0, color='black', alpha=0.5)
                
                axes[k].set_title(f"{lbl}")
                axes[k].legend()
            else:
                axes[k].axis('off') # Ukryj puste osie

        plt.suptitle(f"GJR-GARCH: {title}", fontsize=22)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_folder, filename), dpi=300)
        plt.close()

    # 3. Generowanie wykresów
    print("\nGenerowanie wykresów zbiorczych...")
    plot_gjrgarch_2x3(df_results, 'Omega', 'Omega (Poziom bazowy)', '2x3_GJR_Omega.png', 'gray')
    plot_gjrgarch_2x3(df_results, 'Alpha', 'Alpha (Reakcja na szoki)', '2x3_GJR_Alpha.png', 'skyblue')
    plot_gjrgarch_2x3(df_results, 'Gamma', 'Gamma (Asymetria / Efekt dźwigni)', '2x3_GJR_Gamma.png', 'purple')
    plot_gjrgarch_2x3(df_results, 'Beta', 'Beta (Trwałość zmienności)', '2x3_GJR_Beta.png', 'salmon')
    plot_gjrgarch_2x3(df_results, 'Persystencja', 'Persystencja Modelu', '2x3_GJR_Persystencja.png', 'green')

    print(f"\n✅ ZAKOŃCZONO. Wyniki: {output_excel}\nWykresy w: {output_folder}")

except Exception as e:
    print(f"🔴 Błąd krytyczny: {e}")