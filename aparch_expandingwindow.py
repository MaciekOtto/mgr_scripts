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
output_excel = 'wyniki_APARCH_Finalna_Proba.xlsx'
output_folder = 'wykresy_aparch_2x3'
os.makedirs(output_folder, exist_ok=True)

try:
    print(f"Wczytuję dane: {input_file}...")
    df = pd.read_excel(input_file)
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data])
        df.set_index(col_data, inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce')).dropna(how='all')

    years = sorted(df.index.year.unique())
    all_results = []

    # Definicja pustego szablonu wiersza (gwarantuje istnienie kolumn w Excelu)
    def get_empty_row(ticker, label, status_msg):
        return {
            'Okres': label, 'Spółka': ticker, 'Omega': np.nan, 'Alpha': np.nan, 
            'Gamma': np.nan, 'Beta': np.nan, 'Delta': np.nan, 'Nu (df)': np.nan, 
            'Status': status_msg
        }

    for year in years:
        label = f"2020-{year}" if year > 2020 else "2020"
        subset = df[df.index.year <= year]
        print(f"\n>>> Okres: {label} (Obs: {len(subset)})")
        
        for j, ticker in enumerate(df.columns):
            series = subset[ticker].dropna()
            
            if len(series) < 200: # APARCH potrzebuje min. 200-250 dni
                all_results.append(get_empty_row(ticker, label, 'BŁĄD: Za mało danych'))
                continue

            try:
                # Próba APARCH z t-Studentem
                # rescale=True jest kluczowe dla zbieżności
                am = arch_model(series, mean='Constant', vol='APARCH', p=1, o=1, q=1, dist='studentst', rescale=True)
                res = am.fit(disp='off', show_warnings=False)
                
                if res.convergence_flag == 0:
                    p = res.params
                    all_results.append({
                        'Okres': label, 'Spółka': ticker,
                        'Omega': p.get('omega'), 'Alpha': p.get('alpha[1]'),
                        'Gamma': p.get('gamma[1]'), 'Beta': p.get('beta[1]'),
                        'Delta': p.get('delta'), 'Nu (df)': p.get('nu'),
                        'Status': 'OK'
                    })
                else:
                    all_results.append(get_empty_row(ticker, label, 'BŁĄD: Brak zbieżności'))
            except:
                all_results.append(get_empty_row(ticker, label, 'BŁĄD: Macierz osobliwa'))

            if (j + 1) % 250 == 0:
                print(f"   ...przetworzono {j + 1} spółek")

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_excel, index=False)

    # --- BEZPIECZNE GENEROWANIE WYKRESÓW ---
    print("\nGenerowanie wykresów...")
    
    def safe_plot_2x3(data, metric, title, filename, color):
        # Sprawdzamy czy kolumna istnieje i czy są w niej jakiekolwiek dane OK
        if metric not in data.columns:
            print(f"⚠️ Kolumna {metric} nie istnieje w wynikach.")
            return
            
        df_ok = data[(data['Status'] == 'OK') & (data[metric].notnull())]
        if df_ok.empty:
            print(f"⚠️ Brak sukcesów dla parametru {metric} - pomijam wykres.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for k, lbl in enumerate(years):
            if k >= 6: break
            period_label = f"2020-{lbl}" if lbl > 2020 else "2020"
            period_data = df_ok[df_ok['Okres'] == period_label][metric]
            
            if not period_data.empty:
                # Filtracja 1-99 percentyl (usuwa błędy numeryczne optymalizatora)
                q_low, q_high = period_data.quantile([0.01, 0.99])
                period_data = period_data[(period_data >= q_low) & (period_data <= q_high)]
                
                sns.histplot(period_data, kde=True, ax=axes[k], color=color)
                axes[k].axvline(period_data.mean(), color='red', ls='--', label=f'Śr: {period_data.mean():.3f}')
                axes[k].set_title(period_label)
                axes[k].legend()
        
        plt.suptitle(f"APARCH: {title}", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    # Wywołanie wykresów
    safe_plot_2x3(df_results, 'Alpha', 'Alpha (ARCH)', '2x3_Alpha.png', 'skyblue')
    safe_plot_2x3(df_results, 'Beta', 'Beta (GARCH)', '2x3_Beta.png', 'salmon')
    safe_plot_2x3(df_results, 'Gamma', 'Gamma (Asymetria)', '2x3_Gamma.png', 'purple')
    safe_plot_2x3(df_results, 'Delta', 'Delta (Potęga)', '2x3_Delta.png', 'orange')

    print(f"\n✅ GOTOWE. Sprawdź plik Excel: {output_excel}")

except Exception as e:
    print(f"🔴 Błąd krytyczny: {e}")