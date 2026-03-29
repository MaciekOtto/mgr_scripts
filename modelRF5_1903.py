import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

HORYZONT = 5
FILE_PATH = 'dane1000stopy.xlsx' 

def prepare_data_excel(file_path):
    print("Wczytywanie danych z Excela...")
    df = pd.read_excel(file_path, sheet_name=0)
    ticker_order = [col for col in df.columns[1:]]
    date_col = df.columns[0]
    
    df_long = df.melt(id_vars=[date_col], var_name='Ticker', value_name='Return')
    df_long[date_col] = pd.to_datetime(df_long[date_col])
    df_long.dropna(subset=['Return'], inplace=True)
    df_long = df_long.sort_values([date_col, 'Ticker'])

    for i in range(1, HORYZONT + 1):
        df_long[f'lag_{i}'] = df_long.groupby('Ticker')['Return'].shift(i)
        df_long[f'target_{i}d'] = df_long.groupby('Ticker')['Return'].shift(-i)
    
    df_long.dropna(inplace=True)
    return df_long, date_col, ticker_order

try:
    df_panel, date_name, ticker_order = prepare_data_excel(FILE_PATH)
    
    unique_dates = sorted(df_panel[date_name].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    train_data = df_panel[df_panel[date_name] < split_date].copy()
    test_data = df_panel[df_panel[date_name] >= split_date].copy()

    X_cols = [f'lag_{i}' for i in range(1, HORYZONT + 1)]
    y_cols = [f'target_{i}d' for i in range(1, HORYZONT + 1)]

    # --- OPTYMALIZACJA (Tuning na podpróbce) ---
    print("Rozpoczynam tuning na 10% danych treningowych (oszczędność RAM)...")
    train_sample = train_data.sample(frac=0.1, random_state=42) # Pobieramy 10% losowych wierszy
    
    param_dist = {
        'n_estimators': [20, 50], # Mniej drzew do wyboru
        'max_depth': [5, 10, None],
        'max_features': ['sqrt'] # 'sqrt' jest znacznie lżejsze dla RAM niż None
    }

    # n_jobs=2 zostawia wolne zasoby dla Windowsa
    base_rf = RandomForestRegressor(random_state=42, n_jobs=2)
    
    rf_random = RandomizedSearchCV(
        estimator=base_rf, 
        param_distributions=param_dist, 
        n_iter=5, # Mniejsza liczba prób
        cv=2, # Mniej podziałów w walidacji krzyżowej
        verbose=1, 
        random_state=42, 
        scoring='r2'
    )

    rf_random.fit(train_sample[X_cols], train_sample[y_cols])
    best_params = rf_random.best_params_
    print(f"Najlepsze parametry: {best_params}")

    # --- TRENOWANIE FINALNE (Na wszystkich danych, ale z optymalnymi parametrami) ---
    print("Trenowanie modelu finalnego na pełnym zbiorze...")
    final_model = RandomForestRegressor(
        **best_params, 
        random_state=42, 
        n_jobs=2 # Nadal trzymamy limit procesora
    )
    final_model.fit(train_data[X_cols], train_data[y_cols])

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
        
        res_row = {
            'Ticker': ticker,
            'Zagregowane_R2': r2_score(actuals_flat, preds_flat),
            'Zagregowane_MSE': mean_squared_error(actuals_flat, preds_flat),
            'Zagregowane_MAE': mean_absolute_error(actuals_flat, preds_flat)
        }
        
        last_row = ticker_df.iloc[-1]
        for d in range(1, HORYZONT + 1):
            res_row[f'Rzeczywista_{d}d'] = last_row[f'target_{d}d']
            res_row[f'Predykcja_{d}d'] = last_row[f'Pred_{d}d']
        final_results.append(res_row)

    final_df = pd.DataFrame(final_results)
    final_df.to_csv('wyniki_magisterka_tuning_safe.csv', index=False, sep=';', decimal=',')
    print(f"SUKCES! Przetworzono {len(final_df)} spółek.")

except Exception as e:
    print(f"Wystąpił błąd: {e}")