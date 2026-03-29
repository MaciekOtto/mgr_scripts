import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import time
import os
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx'
output_file_aparch = 'wyniki_APARCH_1000.xlsx' 
output_folder = 'wykresy_aparch_parametry'
os.makedirs(output_folder, exist_ok=True)

print(f"Wczytuję dane z pliku: {input_file}...")

try:
    df_returns = pd.read_excel(input_file)
    
    # --- Przygotowanie Danych ---
    col_data = next((c for c in df_returns.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
        
    # Konwersja na liczby i czyszczenie
    df_returns = df_returns.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df_returns.dropna(how='all', inplace=True)

    print(f"Liczba spółek do przeanalizowania: {len(df_returns.columns)}")

    wyniki_aparch = []
    start_time = time.time()

    print("\nRozpoczynam szacowanie modeli APARCH(1,1)...")

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].dropna()
        
        # Skalowanie (kluczowe dla APARCH)
        series_scaled = series * 100
        
        if len(series_scaled) < 100: 
            continue

        try:
            # vol='APARCH', p=1, o=1, q=1 -> o=1 odpowiada za parametr Gamma
            # dist='Normal' lub 't' (studenta) - tutaj zostawiamy Normal dla porównywalności
            am = arch_model(series_scaled, mean='Constant', vol='APARCH', p=1, o=1, q=1, rescale=True)
            res = am.fit(disp='off', show_warning=False)

            p = res.params
            
            # W APARCH mamy: omega, alpha, gamma, beta oraz delta (parametr potęgowy)
            omega = p.get('omega', np.nan)
            alpha = p.get('alpha[1]', np.nan)
            gamma = p.get('gamma[1]', np.nan)
            beta = p.get('beta[1]', np.nan)
            delta = p.get('delta', np.nan) # Specyficzne dla APARCH
            
            wyniki_aparch.append({
                'Spółka': ticker,
                'Omega': round(omega, 6) if not np.isnan(omega) else None,
                'Alpha (ARCH)': round(alpha, 6) if not np.isnan(alpha) else None,
                'Gamma (Asymetria)': round(gamma, 6) if not np.isnan(gamma) else None,
                'Beta (GARCH)': round(beta, 6) if not np.isnan(beta) else None,
                'Delta (Potęga)': round(delta, 6) if not np.isnan(delta) else None,
                'AIC': res.aic,
                'Status': 'OK'
            })
            
        except Exception as e:
            wyniki_aparch.append({
                'Spółka': ticker,
                'Status': f'BŁĄD: {str(e)[:50]}'
            })

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} spółek...")

    # 2. Zapisanie Wyników
    df_wyniki = pd.DataFrame(wyniki_aparch)
    df_wyniki.to_excel(output_file_aparch, index=False)
    
    print(f"\nZakończono! Wyniki zapisano w: {output_file_aparch}")

    # 3. Tworzenie Wykresów
    df_plot = df_wyniki[df_wyniki['Status'] == 'OK'].copy()
    
    if not df_plot.empty:
        # Lista parametrów do narysowania
        params = [
            ('Omega','gray','Bazowy poziom wariancji.png')
            ('Alpha (ARCH)', 'skyblue', '01_Rozklad_Alpha.png'),
            ('Beta (GARCH)', 'lightcoral', '02_Rozklad_Beta.png'),
            ('Gamma (Asymetria)', 'purple', '03_Rozklad_Gamma.png'),
            ('Delta (Potęga)', 'orange', '04_Rozklad_Delta.png')
        ]

        for col, color, filename in params:
            plt.figure(figsize=(10, 6))
            # Usuwamy outliery (percentyle 1-99), by wykres był czytelny
            data = df_plot[col].dropna()
            q_low, q_high = data.quantile([0.01, 0.99])
            data_filtered = data[(data > q_low) & (data < q_high)]
            
            sns.histplot(data_filtered, kde=True, color=color)
            plt.axvline(data_filtered.mean(), color='red', linestyle='--', label=f'Średnia: {data_filtered.mean():.3f}')
            
            if 'Gamma' in col:
                plt.axvline(0, color='black', alpha=0.5)
                plt.title('Rozkład parametru Gamma (Asymetria - Efekt dźwigni)')
            elif 'Delta' in col:
                plt.axvline(2, color='blue', linestyle=':', label='Standardowy GARCH (2.0)')
                plt.title('Rozkład parametru Delta (Potęga transformacji)')
            else:
                plt.title(f'Rozkład parametru {col}')
            
            plt.legend()
            plt.savefig(os.path.join(output_folder, filename))
            plt.close()
        
        print(f"Wykresy (Alpha, Beta, Gamma, Delta) zapisano w folderze: {output_folder}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")