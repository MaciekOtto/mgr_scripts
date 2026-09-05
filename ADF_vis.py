"""
ADF_vis.py - Wizualizacja wyników testu ADF

Wczytuje wyniki testu ADF i rysuje dwa histogramy: rozkład statystyki
ADF oraz rozkład p-value (z linią odcięcia p = 0.05) dla wszystkich
spółek.

Wejście: wyniki_ADF_1000_stp.xlsx
Wyjście: histogramy_adf.png

----------------------------------------------------------------------

ADF_vis.py - ADF test results visualization

Loads the ADF test results and draws two histograms: the distribution
of the ADF statistic and the distribution of p-values (with a p = 0.05
cutoff line) across all companies.

Input: wyniki_ADF_1000_stp.xlsx
Output: histogramy_adf.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "wyniki_ADF_1000_stp.xlsx"
OUTPUT_PLOT = "histogramy_adf.png"

# 1. Wczytanie pliku
df = pd.read_excel(INPUT_FILE)

# czyszczenie kolumn
df.columns = df.columns.astype(str).str.strip()

# 2. Zamiana przecinków na kropki i konwersja na liczby
adf_stat = pd.to_numeric(
    df["Statystyka ADF"].astype(str).str.replace(',', '.'), 
    errors="coerce"
).dropna()

p_value = pd.to_numeric(
    df["p-value"].astype(str).str.replace(',', '.'), 
    errors="coerce"
).dropna()

# 3. Tworzenie histogramów
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Wykres Statystyka ADF 
axes[0].hist(adf_stat, bins=40, color="lightgray", edgecolor="black", alpha=0.8)
axes[0].set_title("Rozkład statystyki ADF")
axes[0].set_xlabel("Statystyka ADF")
axes[0].set_ylabel("Liczba spółek")

# Wykres p-value 
p_bins = np.linspace(0, 1, 41)

axes[1].hist(p_value, bins=p_bins, color="lightgray", edgecolor="black", alpha=0.8)
axes[1].axvline(0.05, color="red", linestyle="--", linewidth=2, label="p = 0.05")
axes[1].set_title("Rozkład p-value testu ADF")
axes[1].set_xlabel("p-value")
axes[1].set_ylabel("Liczba spółek")
axes[1].set_xlim(0, 1)    # Skala p-value od 0
axes[1].legend()

fig.suptitle("Rozkład wyników testu ADF", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_PLOT, dpi=150)

print("Zapisano:", OUTPUT_PLOT)
print(f"Liczba spółek z p-value > 0.05 (brak podstaw do odrzucenia H0 o niestacjonarności): "
      f"{(p_value > 0.05).sum()} / {len(p_value)}")
