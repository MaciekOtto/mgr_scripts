import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import os
import time

# --- Ustawienia ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_GARCH_narastajaco.xlsx'
output_folder = 'wykresy_garch_2x3'
os.makedirs(output_folder, exist_ok=True)

print(f"Wczytuję dane z pliku: {input_file}...")

try:
    df = pd.read_excel(input_file)
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data])
        df.set_index(col_data, inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df.dropna(inplace=True)

    years = sorted(df.index.year.unique())
    garch_results = []

    # 1. Główna pętla po okresach narastających
    for i, year in enumerate(years):
        label = f"2020-{year}" if year > 2020 else "2020"
        subset = df[df.index.year <= year]
        print(f"\n>>> Szacowanie GARCH dla okresu: {label} (Liczba obs: {len(subset)})")
        
        # Pętla po spółkach
        for j, ticker in enumerate(df.columns):
            try:
                # Skalujemy stopy zwrotu (GARCH lepiej konwerguje na większych liczbach, np. % zamiast ułamków)
                series = subset[ticker] * 100 
                
                am = arch_model(series, mean='Constant', vol='Garch', p=1, q=1, dist='Normal')
                res = am.fit(disp='off', show_warning=False)
                
                omega = res.params['omega']
                alpha = res.params['alpha[1]']
                beta = res.params['beta[1]']
                
                garch_results.append({
                    'Okres': label,
                    'Spółka': ticker,
                    'Omega':omega,
                    'Alpha': alpha,
                    'Beta': beta,
                    'Suma_AB': alpha + beta,
                    'Status': 'OK'
                })
            except:
                garch_results.append({'Okres': label, 'Spółka': ticker, 'Status': 'BŁĄD'})
            
            if (j + 1) % 250 == 0:
                print(f"   ...przetworzono {j + 1} spółek")

    df_results = pd.DataFrame(garch_results)
    df_results.to_excel(output_excel, index=False)

    # 2. Funkcja do generowania zestawień 2x3
    def plot_garch_2x3(data, metric, title, filename, color):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        labels = data['Okres'].unique()

        for k, lbl in enumerate(labels):
            if k >= 6: break
            period_data = data[(data['Okres'] == lbl) & (data['Status'] == 'OK')]
            
            sns.histplot(period_data[metric], kde=True, ax=axes[k], color=color)
            axes[k].set_title(f"{lbl}")
            
            # Linia średniej dla parametru
            avg_val = period_data[metric].mean()
            axes[k].axvline(avg_val, color='red', linestyle='--', label=f'Średnia: {avg_val:.3f}')
            axes[k].legend()

        plt.suptitle(f"Rozkład parametru GARCH: {title}", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_folder, filename), dpi=300)
        plt.close()

    # 3. Generowanie wykresów
    print("\nGenerowanie wykresów zbiorczych...")
    plot_garch_2x3(df_results, 'Omega', 'Omega (ARCH - Poziom wariancji bazowej)', '2x3_GARCH_Omega.png', 'gray')
    plot_garch_2x3(df_results, 'Alpha', 'Alpha (ARCH - reakcja na szoki)', '2x3_GARCH_Alpha.png', 'skyblue')
    plot_garch_2x3(df_results, 'Beta', 'Beta (GARCH - trwałość zmienności)', '2x3_GARCH_Beta.png', 'salmon')
    plot_garch_2x3(df_results, 'Suma_AB', 'Suma Alpha+Beta (Stabilność)', '2x3_GARCH_Suma.png', 'green')

    print(f"ZAKOŃCZONO. Wyniki w {output_excel}, wykresy w folderze {output_folder}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")
