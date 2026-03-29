import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

HORYZONT = 5
LOOKBACK = 10 
FILE_PATH = 'dane1000stopy.xlsx' 

def add_indicators(df):
    # Obliczamy na surowych danych przed skalowaniem
    df['SMA_20'] = df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).mean())
    df['Vol_20'] = df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).std())
    return df.fillna(0)

try:
    print("Wczytywanie i przygotowanie danych...")
    df = pd.read_excel(FILE_PATH)
    ticker_order = [col for col in df.columns[1:]]
    date_col = df.columns[0] # To jest nazwa Twojej kolumny z datą
    
    df_long = df.melt(id_vars=[date_col], var_name='Ticker', value_name='Return')
    df_long[date_col] = pd.to_datetime(df_long[date_col])
    df_long = df_long.sort_values(['Ticker', date_col])
    df_long = add_indicators(df_long)

    # Przygotowanie cech
    features = ['Return', 'SMA_20', 'Vol_20']
    
    # Skalowanie - zapamiętujemy scaler dla 'Return', żeby potem odwrócić wynik
    scaler = StandardScaler()
    df_long[features] = scaler.fit_transform(df_long[features])
    
    # Wyciągamy parametry skalowania samej stopy zwrotu (indeks 0 w features)
    return_mean = scaler.mean_[0]
    return_std = scaler.scale_[0]

    X, Y, info = [], [], []

    print(f"Budowanie sekwencji {LOOKBACK}-dniowych...")
    for ticker in ticker_order:
        ticker_data = df_long[df_long['Ticker'] == ticker].copy()
        if len(ticker_data) < (LOOKBACK + HORYZONT + 1):
            continue
            
        values = ticker_data[features].values
        returns = ticker_data['Return'].values
        
        for i in range(LOOKBACK, len(values) - HORYZONT):
            X.append(values[i-LOOKBACK:i])
            Y.append(returns[i:i+HORYZONT])
            info.append((ticker, ticker_data.iloc[i][date_col]))

    X, Y = np.array(X), np.array(Y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    print("Budowanie i trenowanie LSTM (PRO)...")
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(LOOKBACK, len(features))),
        Dropout(0.1),
        LSTM(16),
        Dense(HORYZONT)
    ])

    model.compile(optimizer='adam', loss='mse')
    monitor = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=128, callbacks=[monitor], verbose=1)

    print("Generowanie wyników końcowych...")
    preds = model.predict(X_test)
    
    results = []
    test_info = info[split:]
    
    for ticker in ticker_order:
        ticker_mask = [i for i, t in enumerate(test_info) if t[0] == ticker]
        if not ticker_mask: continue
        
        t_preds = preds[ticker_mask]
        t_actuals = y_test[ticker_mask]
        
        # Statystyki (na danych skalowanych - R2 i kierunek są na to odporne)
        f_preds = t_preds.flatten()
        f_acts = t_actuals.flatten()
        
        # Obliczamy kierunek
        dir_accuracy = np.mean(np.sign(f_acts) == np.sign(f_preds))
        
        res_row = {
            'Ticker': ticker,
            'Zagregowane_R2': r2_score(f_acts, f_preds),
            'Zagregowane_MAE': mean_absolute_error(f_acts, f_preds),
            'Trafnosc_Kierunku': dir_accuracy
        }
        
        # Ostatnie 5 dni - ODWRACAMY SKALOWANIE dla czytelności w Excelu
        # Wzór: oryginalne = (skalowane * std) + mean
        last_p = (t_preds[-1] * return_std) + return_mean
        last_a = (t_actuals[-1] * return_std) + return_mean
        
        for d in range(HORYZONT):
            res_row[f'Rzeczywista_{d+1}d'] = last_a[d]
            res_row[f'Predykcja_{d+1}d'] = last_p[d]
            
        results.append(res_row)

    final_df = pd.DataFrame(results)
    final_df.to_csv('wyniki_LSTM_PRO_v2.csv', index=False, sep=';', decimal=',')
    print("GOTOWE! Plik 'wyniki_LSTM_PRO_v2.csv' czeka na analizę.")

except Exception as e:
    import traceback
    print(f"Błąd: {e}")
    traceback.print_exc()