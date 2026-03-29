import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

HORYZONT = 5
FILE_PATH = 'dane1000stopy.xlsx' 

def add_technical_indicators(df_long):
    print("Obliczanie wskaźników technicznych...")
    # SMA 20 i Zmienność (Volatility)
    df_long['SMA_20'] = df_long.groupby('Ticker')['Return'].transform(lambda x: x.rolling(window=20).mean())
    df_long['Vol_20'] = df_long.groupby('Ticker')['Return'].transform(lambda x: x.rolling(window=20).std())
    
    # RSI (Relative Strength Index)
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df_long['RSI'] = df_long.groupby('Ticker')['Return'].transform(calc_rsi)
    return df_long

def prepare_data_excel(file_path):
    print("Wczytywanie danych z Excela...")
    df = pd.read_excel(file_path, sheet_name=0)
    ticker_order = [col for col in df.columns[1:]]
    date_col = df.columns[0]
    
    df_long = df.melt(id_vars=[date_col], var_name='Ticker', value_name='Return')
    df_long[date_col] = pd.to_datetime(df_long[date_col])
    
    # Sortowanie ważne dla wskaźników technicznych
    df_long = df_long.sort_values(['Ticker', date_col])
    
    # DODANIE WSKAŹNIKÓW
    df_long = add_technical_indicators(df_long)

    # Tworzymy lagi i targety
    for i in range(1, HORYZONT + 1):
        df_long[f'lag_{i}'] = df_long.groupby('Ticker')['Return'].shift(i)
        df_long[f'target_{i}d'] = df_long.groupby('Ticker')['Return'].shift(-i)
    
    df_long.dropna(inplace=True)
    return df_long, date_col, ticker_order

try:
    df_panel, date_name, ticker_order = prepare_data_excel(FILE_PATH)
    
    # 1. PODZIAŁ DANYCH
    unique_dates = sorted(df_panel[date_name].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    train_data = df_panel[df_panel[date_name] < split_date].copy()
    test_data = df_panel[df_panel[date_name] >= split_date].copy()

    # Definicja kolumn wejściowych (X)
    X_cols = [f'lag_{i}' for i in range(1, HORYZONT + 1)] + ['SMA_20', 'Vol_20', 'RSI']
    y_cols = [f'target_{i}d' for i in range(1, HORYZONT + 1)]

    # --- OPTYMALIZACJA ---
    print("Rozpoczynam tuning na 10% danych treningowych...")
    train_sample = train_data.sample(frac=0.1, random_state=42)
    
    param_dist = {
        'n_estimators': [20, 50],
        'max_depth': [5, 10, None],
        'max_features': ['sqrt']
    }

    base_rf = RandomForestRegressor(random_state=42, n_jobs=2)
    rf_random = RandomizedSearchCV(
        estimator=base_rf, 
        param_distributions=param_dist, 
        n_iter=5, 
        cv=2, 
        verbose=1, 
        random_state=42, 
        scoring='r2'
    )

    rf_random.fit(train_sample[X_cols], train_sample[y_cols])
    best_params = rf_random.best_params_
    print(f"Najlepsze parametry z tuningu: {best_params}")

    # --- TRENOWANIE FINALNE ---
    print("Trenowanie modelu finalnego...")
    final_model = RandomForestRegressor(
        **best_params, 
        random_state=42, 
        n_jobs=2 
    )
    final_model.fit(train_data[X_cols], train_data[y_cols])

    # --- NOWE: OBLICZANIE WARIANCJI (NIEPEWNOŚCI MODELU) ---
    print("Analiza niepewności modelu (wariancja drzew)...")
    # Pobieramy predykcje z każdego drzewa osobno
    tree_preds = np.array([tree.predict(test_data[X_cols]) for tree in final_model.estimators_])
    # Obliczamy wariancję między drzewami dla każdej prognozy
    # Następnie wyciągamy średnią z 5 dni dla każdego wiersza
    test_data['Ryzyko_Modelu_RF'] = np.mean(np.var(tree_preds, axis=0), axis=1)

    # --- PREDYKCJE ---
    print("Generowanie predykcji...")
    test_preds = final_model.predict(test_data[X_cols])
    for i in range(HORYZONT):
        test_data[f'Pred_{i+1}d'] = test_preds[:, i]

    # --- GENEROWANIE FINALNEJ TABELI ---
    final_results = []
    p_cols = [f'Pred_{i}d' for i in range(1, HORYZONT + 1)]
    t_cols = [f'target_{i}d' for i in range(1, HORYZONT + 1)]
    
    for ticker in ticker_order:
        ticker_df = test_data[test_data['Ticker'] == ticker]
        if ticker_df.empty: continue
        
        actuals_flat = ticker_df[t_cols].values.flatten()
        preds_flat = ticker_df[p_cols].values.flatten()
        
        # OBLICZANIE TRAFNOŚCI KIERUNKU
        same_direction = np.sign(actuals_flat) == np.sign(preds_flat)
        dir_accuracy = np.mean(same_direction)
        
        res_row = {
            'Ticker': ticker,
            'Zagregowane_R2': r2_score(actuals_flat, preds_flat),
            'Zagregowane_MSE': mean_squared_error(actuals_flat, preds_flat),
            'Zagregowane_MAE': mean_absolute_error(actuals_flat, preds_flat),
            'Trafnosc_Kierunku': dir_accuracy,
            'Ryzyko_Modelu_Średnie': ticker_df['Ryzyko_Modelu_RF'].mean() # NOWA KOLUMNA
        }
        
        last_row = ticker_df.iloc[-1]
        for d in range(1, HORYZONT + 1):
            res_row[f'Rzeczywista_{d}d'] = last_row[f'target_{d}d']
            res_row[f'Predykcja_{d}d'] = last_row[f'Pred_{d}d']
        final_results.append(res_row)

    # --- NOWE: ZAPIS ISTOTNOŚCI CECH ---
    importances = pd.DataFrame({'Cecha': X_cols, 'Waga': final_model.feature_importances_})
    importances.sort_values(by='Waga', ascending=False).to_csv('istotnosc_cech_RF.csv', index=False, sep=';', decimal=',')

    final_df = pd.DataFrame(final_results)
    final_df.to_csv('wyniki_magisterka_z_ryzykiem_v5.csv', index=False, sep=';', decimal=',')
    print(f"SUKCES! Wygenerowano pliki: 'wyniki_magisterka_z_ryzykiem_v5.csv' oraz 'istotnosc_cech_RF.csv'.")

except Exception as e:
    print(f"Wystąpił błąd: {e}")
