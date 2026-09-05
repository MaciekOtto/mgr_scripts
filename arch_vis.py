"""
arch_vis.py - Wizualizacja wyników testu ARCH

Wczytuje wyniki testu ARCH i rysuje histogram rozkładu p-value (z
linią odcięcia p = 0.05) dla wszystkich spółek.

Wejście: wyniki_ARIMA_ARCH_1000_stopyzw.xlsx
Wyjście: histogramy_arch_stp.png

----------------------------------------------------------------------

arch_vis.py - ARCH test results visualization

Loads the ARCH test results and draws a histogram of p-value
distribution (with a p = 0.05 cutoff line) across all companies.

Input: wyniki_ARIMA_ARCH_1000_stopyzw.xlsx
Output: histogramy_arch_stp.png

capm_3factors.py - CAPM / Fama-French (FF3) cross-sectional regression

Goal: test whether a firm's exposure to risk factors explains
cross-sectional differences in forecast errors (RMSE). Two stages:
(1) for each company, CAPM (market only) and FF3 (market, SMB, HML)
betas are estimated from its returns and the Fama-French factors;
(2) a cross-sectional regression of RMSE (from each of the six models)
on the estimated betas is run, with HAC standard errors.

Input: dane1000stopy.xlsx, F-F_Research_Data_Factors_daily.txt,
       garch_rmse_mae.xlsx, rf_rmse_mae.xlsx, lstm_rmse_mae.xlsx,
       svr_rmse_mae.xlsx
Output: ff3_bety_spolek.xlsx, ff3_regresja_przekrojowa.xlsx
"""

import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "wyniki_ARIMA_ARCH_1000_stopyzw.xlsx"
OUTPUT_PLOT = "histogramy_arch_stp.png"

# Wczytanie danych
raw = pd.read_excel(INPUT_FILE)

# Pierwsza kolumna jako indeks
raw = raw.set_index(raw.columns[0])

# Wyciąganie p-value dla testu ARCH
p_value = pd.to_numeric(raw["p_value_ARCH"], errors="coerce").dropna()

# Tworzenie pojedynczego wykresu 
fig, ax = plt.subplots(figsize=(9, 5))

# Rysowanie histogramu
ax.hist(p_value, bins=40, color="lightgray", edgecolor="black", alpha=0.8)
ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="p = 0.05")
ax.set_title("Rozkład p-value testu ARCH")
ax.set_xlabel("p-value")
ax.set_ylabel("Liczba spółek")
ax.legend()

# Tytuł i zapisywanie
fig.suptitle("Rozkład wyników testu ARCH dla stóp zwrotu", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_PLOT, dpi=150)

print("Zapisano:", OUTPUT_PLOT)

print(f"Liczba spółek z p-value > 0.05 (brak podstaw do odrzucenia H0 o braku efektów ARCH): "
      f"{(p_value > 0.05).sum()} / {len(p_value)}")
