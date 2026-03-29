import pandas as pd
from scipy.stats import jarque_bera

# Tu wpisz nazwę swojego pliku Excel (z końcówką .xlsx)
input_file = 'dane1000stopy.xlsx' 
output_file = 'statystyki_opisowe_stopy.xlsx'

def calculate_stats(file_path, output_path):
    try:
        # ZMIANA: Używamy read_excel zamiast read_csv
        # Domyślnie czyta pierwszy arkusz. Jeśli dane są na innym, dodaj: sheet_name='NazwaArkusza'
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return f"Błąd: Nie znaleziono pliku '{file_path}'. Sprawdź nazwę i folder."
    except Exception as e:
        return f"Wystąpił błąd: {e}"

    # Ustawienie kolumny 'Data' jako indeksu (jeśli istnieje)
    if 'Data' in df.columns:
        df['Data'] = pd.to_datetime(df['Data'])
        df.set_index('Data', inplace=True)
    
    results = {}

    for col in df.columns:
        # Pobieramy dane, pomijając puste komórki
        series = df[col].dropna()
        
        # Jeśli kolumna nie ma liczb (np. same napisy lub pusta), pomijamy ją
        if series.empty or not pd.api.types.is_numeric_dtype(series):
            continue
            
        # Obliczenie testu Jarque-Bera
        try:
            jb_stat, jb_p = jarque_bera(series)
        except Exception:
            jb_stat, jb_p = None, None # Zabezpieczenie gdyby liczby były błędne

        results[col] = {
            'Średnia': series.mean(),
            'Min': series.min(),
            'Max': series.max(),
            'Odchylenie Std': series.std(),
            'Wariancja': series.var(),
            'Skośność': series.skew(),
            'Kurtoza': series.kurtosis(),
            'Jarque-Bera Stat': jb_stat,
            'Jarque-Bera p-value': jb_p
        }

    # Zapis wyników
    stats_df = pd.DataFrame(results).T
    stats_df.to_excel(output_path)
    
    return f"Gotowe! Statystyki zapisane w pliku: {output_path}"

# Uruchomienie
print(calculate_stats(input_file, output_file))