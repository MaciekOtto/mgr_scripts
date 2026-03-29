import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem # Triki dla nieliniowości
from sklearn.metrics import r2_score, mean_absolute_error

HORYZONT = 5
FILE_PATH = 'dane1000stopy.xlsx' 

def add_indicators(df):
    # RSI - kluczowe dla SVR, żeby widział poziomy wykupienia
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / loss)))

    df['SMA_20'] = df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).mean())
    df['Vol_20'] = df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).std())
    df['RSI'] = df.groupby('Ticker')['Return'].transform(calc_rsi)
    return df.fillna(0)

try:
    print("Wczytywanie i optymalizacja danych...")
    df = pd.read_excel(FILE_PATH)
    ticker_order = [col for col in df.columns[1:]]
    date_col = df.columns[0]
    
    df_long = df.melt(id_vars=[date_col], var_name='Ticker', value_name='Return')
    df_long[date_col] = pd.to_datetime(df_long[date_col])
    df_long = df_long.sort_values(['Ticker', date_col])
    
    # 1. Winsoryzacja - przycinamy ekstremalne outliery, które mylą SVM
    q_low = df_long['Return'].quantile(0.01)
    q_high = df_long['Return'].quantile(0.99)
    df_long['Return'] = df_long['Return'].clip(q_low, q_high)
    
    df_long = add_indicators(df_long)

    for i in range(1, HORYZONT + 1):
        df_long[f'lag_{i}'] = df_long.groupby('Ticker')['Return'].shift(i)
        df_long[f'target_{i}d'] = df_long.groupby('Ticker')['Return'].shift(-i)
    
    df_long.dropna(inplace=True)

    # Podział danych
    unique_dates = sorted(df_long[date_col].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_df = df_long[df_long[date_col] < split_date].copy()
    test_df = df_long[df_long[date_col] >= split_date].copy()

    X_cols = [f'lag_{i}' for i in range(1, HORYZONT + 1)] + ['SMA_20', 'Vol_20', 'RSI']
    y_cols = [f'target_{i}d' for i in range(1, HORYZONT + 1)]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(train_df[X_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[y_cols])
    X_test_scaled = scaler_x.transform(test_df[X_cols])

    # --- TRIK NYSTROEM (Aproksymacja nieliniowości) ---
    print("Generowanie nieliniowych cech (Nystroem)...")
    nystroem = Nystroem(kernel='rbf', gamma=0.1, n_components=100, random_state=42)
    # Mapujemy dane na przestrzeń nieliniową
    X_train_nys = nystroem.fit_transform(X_train_scaled)
    X_test_nys = nystroem.transform(X_test_scaled)

    # --- TRENOWANIE ---
    print("Trenowanie ulepszonego SVR...")
    # C=0.1 zwiększa regularyzację (chroni przed overfittingiem)
    base_svm = LinearSVR(C=0.1, epsilon=0.01, max_iter=5000, random_state=42)
    model = MultiOutputRegressor(base_svm)
    
    # Trenujemy na 30% danych (więcej niż ostatnio, nystroem jest wydajny)
    idx = np.random.choice(len(X_train_nys), int(len(X_train_nys) * 0.3), replace=False)
    model.fit(X_train_nys[idx], y_train_scaled[idx])

    # Predykcje
    print("Generowanie wyników...")
    preds = scaler_y.inverse_transform(model.predict(X_test_nys))

    for i in range(HORYZONT):
        test_df[f'Pred_{i+1}d'] = preds[:, i]

    results = []
    for ticker in ticker_order:
        ticker_df = test_df[test_df['Ticker'] == ticker]
        if ticker_df.empty: continue
        
        acts = ticker_df[[f'target_{i}d' for i in range(1, 6)]].values.flatten()
        prds = ticker_df[[f'Pred_{i}d' for i in range(1, 6)]].values.flatten()
        
        res_row = {
            'Ticker': ticker,
            'Zagregowane_R2': r2_score(acts, prds),
            'Trafnosc_Kierunku': np.mean(np.sign(acts) == np.sign(prds)),
            'Rzeczywista_1d': ticker_df.iloc[-1]['target_1d'],
            'Predykcja_1d': ticker_df.iloc[-1]['Pred_1d']
        }
        results.append(res_row)

    pd.DataFrame(results).to_csv('wyniki_SVM_TURBO.csv', index=False, sep=';', decimal=',')
    print(f"GOTOWE! Średnia trafność: {pd.DataFrame(results)['Trafnosc_Kierunku'].mean():.4f}")

except Exception as e:
    print(f"Błąd: {e}")