import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Funkcja do wczytywania tickerów z CSV z dodatkową filtracją
def load_tickers_from_csv(file_path='nasdaq_top5001.csv', max_tickers=1721):
    try:
        # Wczytaj jako zwykły tekst (bez nagłówków)
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
days_back = 2200  # Ostatnie 90 dni
min_data_ratio = 0.9  # Minimalny procent dni roboczych z danymi
start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')  # Do dziś
tickers = load_tickers_from_csv(max_tickers=1721)
if not tickers:
    print("Brak tickerów – użyj przykładowej listy.")
    tickers = ['AAPL', 'MSFT', 'NVDA']  # Fallback

print(f"Pobieranie danych dla {len(tickers)} spółek z okresu {start_date} do {end_date}...")
try:
    # Pobierz pełne dane OHLCV
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    print(f"Pobrano dane dla {len(data.columns.levels[1])} spółek z {len(tickers)} żądanych.")
except Exception as e:
    print(f"Błąd podczas pobierania danych: {e}")
    data = pd.DataFrame()  # Pusta ramka

if data.empty:
    print("Brak danych – sprawdź tickery lub połączenie internetowe.")
    exit()

# Utwórz zakres wszystkich dni roboczych (5-dniowy tydzień: pon-pt)
all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')
total_business_days = len(all_business_days)

# Filtruj spółki: tylko te z wystarczającą liczbą danych (na podstawie Close)
filtered_tickers = []
excluded_tickers = []
for ticker in tickers:
    if ('Close', ticker) in data.columns:
        series = data['Close'][ticker].dropna()
        data_days = len(series)
        ratio = data_days / total_business_days
        if ratio >= min_data_ratio:
            filtered_tickers.append(ticker)
        else:
            excluded_tickers.append((ticker, ratio))

print(f"Po filtrze: {len(filtered_tickers)} spółek z pełnymi danymi (co najmniej {min_data_ratio*100}% dni). Wykluczonych: {len(excluded_tickers)}.")
if excluded_tickers:
    print(f"Wykluczone spółki (ticker, procent danych): {excluded_tickers[:10]}...")

# Przygotuj DataFrame z wszystkimi wskaźnikami
filled_data = pd.DataFrame(index=all_business_days)
for ticker in filtered_tickers:
    # Pobierz podstawowe dane OHLCV i zreindexuj
    open_price = data['Open'][ticker].reindex(all_business_days).fillna(method='ffill')
    high = data['High'][ticker].reindex(all_business_days).fillna(method='ffill')
    low = data['Low'][ticker].reindex(all_business_days).fillna(method='ffill')
    close = data['Close'][ticker].reindex(all_business_days).fillna(method='ffill')
    volume = data['Volume'][ticker].reindex(all_business_days).fillna(method='ffill')
    
    # Zmiana ceny zamknięcia w procentach
    pct_change = close.pct_change() * 100
    pct_change.iloc[0] = 0  # Ustaw 0 dla pierwszego dnia (brak poprzedniej ceny)
    
    # Proste średnie kroczące 5-dniowe (SMA5 na Close)
    #sma5 = close.rolling(window=5).mean().fillna(method='bfill')
    sma5 = close.rolling(window=5).mean()
    # Wykładnicze średnie kroczące (EMA12 i EMA26 na Close)
    ema12 = close.ewm(span=12).mean().fillna(method='bfill')
    ema26 = close.ewm(span=26).mean().fillna(method='bfill')
    
    # MACD: EMA12 - EMA26
    macd = ema12 - ema26
    # Sygnał MACD: EMA9 na MACD
    #macd_signal = macd.ewm(span=9).mean().fillna(method='bfill')
    
    # Dzień tygodnia: 1=poniedziałek, 2=wtorek, ..., 5=piątek (tylko dni robocze)
    weekday = filled_data.index.weekday + 1  # 0=pon -> 1, ..., 4=pt -> 5
    
    # Dodaj kolumny do DataFrame
    filled_data[f'{ticker}_Open'] = open_price
    filled_data[f'{ticker}_High'] = high
    filled_data[f'{ticker}_Low'] = low
    filled_data[f'{ticker}_Close'] = close
    filled_data[f'{ticker}_Volume'] = volume
    filled_data[f'{ticker}_Pct_Change'] = pct_change
    filled_data[f'{ticker}_SMA5'] = sma5
    filled_data[f'{ticker}_EMA12'] = ema12
    #filled_data[f'{ticker}_EMA26'] = ema26
    filled_data[f'{ticker}_MACD'] = macd
    #filled_data[f'{ticker}_MACD_Signal'] = macd_signal
    filled_data[f'{ticker}_Weekday'] = weekday

# Reset indeksu: data jako pierwsza kolumna
filled_data.reset_index(inplace=True)
filled_data.rename(columns={'index': 'Data'}, inplace=True)

# Zapisz do Excel
output_file = 'notowania_nasdaq_uzup.xlsx'
filled_data.to_excel(output_file, index=False)
print(f"Dane zapisane do pliku {output_file}. Liczba kolumn (oprócz daty): {len(filled_data.columns) - 1}")
