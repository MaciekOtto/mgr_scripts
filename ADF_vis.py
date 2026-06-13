
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "wyniki_ADF_1000_spolek.xlsx"
OUTPUT_PLOT = "histogramy_adf_close.png"

#Wczytanie danych
raw = pd.read_excel(INPUT_FILE, header=None)
raw = raw.set_index(0).T 

raw.columns = raw.iloc[0].index 
adf_stat = pd.to_numeric(raw["Statystyka ADF"], errors="coerce").dropna()
p_value = pd.to_numeric(raw["p-value"], errors="coerce").dropna()

#Histogramy
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(adf_stat, bins=40, color="lightgray", edgecolor="black", alpha=0.8)
axes[0].set_title("Rozkład statystyki ADF")
axes[0].set_xlabel("Statystyka ADF")
axes[0].set_ylabel("Liczba spółek")

axes[1].hist(p_value, bins=40, color="lightgray", edgecolor="black", alpha=0.8)
axes[1].axvline(0.05, color="red", linestyle="--", linewidth=2, label="p = 0.05")
axes[1].set_title("Rozkład p-value testu ADF")
axes[1].set_xlabel("p-value")
axes[1].set_ylabel("Liczba spółek")
axes[1].legend()

fig.suptitle("Rozkład wyników testu ADF dla cen zamknięcia", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_PLOT, dpi=150)

print("Zapisano:", OUTPUT_PLOT)
print(f"Liczba spółek z p-value > 0.05 (brak podstaw do odrzucenia H0 o niestacjonarności): "
      f"{(p_value > 0.05).sum()} / {len(p_value)}")
