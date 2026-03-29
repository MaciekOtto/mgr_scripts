import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, kurtosis, jarque_bera

# --------------------------------------------
# 1. Wczytanie danych z pliku Excel
# --------------------------------------------
file_path = "dane_do_stat.xlsx"   # <-- zmień na swoją nazwę

df = pd.read_excel(file_path)

# Pierwsza kolumna to data
df['Data'] = pd.to_datetime(df.iloc[:, 0])
df = df.set_index('Data')

# Zostaw tylko kolumny numeryczne (log-returny)
df = df.select_dtypes(include=[np.number])

# --------------------------------------------
# 2. Podział na lata
# --------------------------------------------
dfs_by_year = {year: data for year, data in df.groupby(df.index.year)}

# --------------------------------------------
# 3. Obliczanie statystyk rocznych
# --------------------------------------------
annual_stats = {}

for year, data in dfs_by_year.items():

    stats = pd.DataFrame(index=data.columns)

    stats["min"] = data.min()
    stats["max"] = data.max()

    # Annualizacja
    stats["mean"] = data.mean() * 252
    stats["std"] = data.std() * np.sqrt(252)
    stats["var"] = data.var() * 252

    # Statystyki wyższych rzędów
    stats["skew"] = data.apply(skew)
    stats["kurt"] = data.apply(lambda x: kurtosis(x, fisher=False))  # klasyczna kurtoza

    # Jarque–Bera: zapisujemy tylko statystykę
    stats["jarque_bera"] = data.apply(lambda x: jarque_bera(x)[0])

    annual_stats[year] = stats

# --------------------------------------------
# 4. Tworzenie folderu na wyniki
# --------------------------------------------
output_folder = "wyniki_roczne_statystyki"
os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------
# 5. Zapisywanie statystyk do plików
# --------------------------------------------
for year, stats in annual_stats.items():
    stats.to_csv(f"{output_folder}/stats_{year}.csv")

# --------------------------------------------
# 6. Wykresy roczne dla wszystkich spółek
# --------------------------------------------
plot_folder = f"{output_folder}/wykresy"
os.makedirs(plot_folder, exist_ok=True)

metrics_to_plot = ["mean", "std", "var", "skew", "kurt"]

for year, stats in annual_stats.items():
    for metric in metrics_to_plot:
        plt.figure(figsize=(14, 6))

        for col in stats.index:
            plt.plot(stats.index, stats[metric], marker='o', linestyle='-', alpha=0.6)

        plt.title(f"{metric.capitalize()} – {year}")
        plt.xlabel("Spółki")
        plt.ylabel(metric)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{plot_folder}/{metric}_{year}.png")
        plt.close()

print("GOTOWE! Statystyki i wykresy są w folderze:", output_folder)
