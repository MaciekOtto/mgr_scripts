import yfinance as yf
import pandas as pd
from datetime import datetime

# Funkcja do wczytywania tickerów z CSV 
def load_tickers_from_csv(file_path='nasdaq_top5001.csv', max_tickers=1721):
    try:
        # Wczytaj jako zwykły tekst 
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        tickers = []
        for line in lines:
            ticker = line.strip().upper()  # Usuń spacje, zamień na wielkie litery
            if ticker and ticker.isalpha() and 1 <= len(ticker) <= 5:  # Tylko litery, 1-5 znaków
                tickers.append(ticker)
        
        # Usuń duplikaty, zachowując kolejność
        unique_tickers = list(dict.fromkeys(tickers))
        tickers = unique_tickers[:max_tickers]
        print(f"Załadowano {len(tickers)} poprawnych tickerów z pliku (zachowując kolejność): {tickers[:10]}...")  # Pokaż pierwsze 10
        return tickers
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku CSV: {e}")
        return []

# Parametry
min_data_ratio = 0.9  

start_date = '2020-02-12'
end_date = '2025-11-13'  

tickers = load_tickers_from_csv(max_tickers=1721)
if not tickers:
    print("Brak tickerów – użyj przykładowej listy.")
    tickers = ['AAPL', 'MSFT', 'NVDA']  # Fallback

print(f"Pobieranie danych dla {len(tickers)} spółek z okresu {start_date} do 2025-11-12...")
try:
    # Pobierz pełne dane 
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        print(f"Pobrano dane dla {len(data.columns.levels[1])} spółek z {len(tickers)} żądanych.")
    else:
        print(f"Pobrano dane dla 1 spółki.")
except Exception as e:
    print(f"Błąd podczas pobierania danych: {e}")
    data = pd.DataFrame()  # Pusta ramka

if data.empty:
    print("Brak danych – sprawdź tickery lub połączenie internetowe.")
    exit()

# Wyciągamy wyłącznie ceny zamknięcia (Close)
if isinstance(data.columns, pd.MultiIndex):
    close_data = data['Close']
else:
    close_data = pd.DataFrame({tickers[0]: data['Close']})

# Utwórz zakres wszystkich dni roboczych (5-dniowy tydzień: pon-pt) do 12.11.2025
all_business_days = pd.date_range(start=start_date, end='2025-11-12', freq='B')
total_business_days = len(all_business_days)

# Filtruj spółki na podstawie dostępności danych
filtered_tickers = []
excluded_tickers = []
for ticker in tickers:
    if ticker in close_data.columns:
        series = close_data[ticker].dropna()
        data_days = len(series)
        ratio = data_days / total_business_days
        if ratio >= min_data_ratio:
            filtered_tickers.append(ticker)
        else:
            excluded_tickers.append((ticker, ratio))

print(f"Po filtrze: {len(filtered_tickers)} spółek z pełnymi danymi (co najmniej {min_data_ratio*100}% dni). Wykluczonych: {len(excluded_tickers)}.")
if excluded_tickers:
    print(f"Wykluczone spółki (ticker, procent danych): {excluded_tickers[:10]}...")

# Przygotuj DataFrame z uzupełnionymi dniami do pełnych tygodni 5-dniowych
filled_data = pd.DataFrame(index=all_business_days)
for ticker in filtered_tickers:
    # Reindeksowanie do pełnego kalendarza biznesowego i uzupełnienie braków poprzednią wartością (forward fill)
    filled_data[ticker] = close_data[ticker].reindex(all_business_days).ffill()

# Reset indeksu: data jako pierwsza kolumna
filled_data.reset_index(inplace=True)
filled_data.rename(columns={'index': 'Data'}, inplace=True)

# Zapisz do Excel
output_file = 'notowania_nasdaq_uzup.xlsx'
filled_data.to_excel(output_file, index=False)
print(f"Dane zapisane do pliku {output_file}. Liczba kolumn (oprócz daty): {len(filled_data.columns) - 1}")
