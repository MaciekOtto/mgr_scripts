import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "wyniki_ARIMA_ARCH_1000_stopyzw.xlsx"
OUTPUT_PLOT = "histogramy_arch_stp.png"

# Wczytanie danych
raw = pd.read_excel(INPUT_FILE)

# Ustawiamy pierwszą kolumnę jako indeks (np. nazwy spółek)
raw = raw.set_index(raw.columns[0])

# Wyciągamy p-value dla testu ARCH
p_value = pd.to_numeric(raw["p_value_ARCH"], errors="coerce").dropna()

# Tworzenie pojedynczym wykresu (zmienione z subplots(1, 2) i zmienione axes na ax)
fig, ax = plt.subplots(figsize=(9, 5))

# Rysowanie histogramu (używamy 'ax' zamiast 'axes[0]')
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

# Poprawiłem też tekst w princie, bo testujesz teraz ARCH pod kątem heteroskedastyczności, a nie ADF pod kątem niestacjonarności ;)
print(f"Liczba spółek z p-value > 0.05 (brak podstaw do odrzucenia H0 o braku efektów ARCH): "
      f"{(p_value > 0.05).sum()} / {len(p_value)}")
