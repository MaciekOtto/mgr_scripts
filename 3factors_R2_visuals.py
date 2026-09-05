"""
3factors_R2_visuals.py - Wykres R² modeli CAPM vs. FF3

Rysuje wykres słupkowy porównujący współczynnik determinacji (R²)
modeli CAPM i Famy-Frencha (FF3) dla poszczególnych zmiennych
zależnych (błędów prognoz z różnych modeli).

UWAGA: wymaga ręcznie przygotowanego pliku wejściowego - należy
wydzielić arkusz „Regresja przekrojowa” z pliku
ff3_regresja_przekrojowa.xlsx (wynik capm_3factors.py) do osobnego
pliku ff3_regresja_przekrojowa_2.xlsx.

Wejście: ff3_regresja_przekrojowa_2.xlsx (przygotowany ręcznie)
Wyjście: porownanie_R2_CAPM_FF3.png

----------------------------------------------------------------------

3factors_R2_visuals.py - R² comparison chart: CAPM vs. FF3

Draws a bar chart comparing the coefficient of determination (R²) of
the CAPM and Fama-French (FF3) models for each dependent variable
(forecast errors from the different models).

NOTE: requires a manually prepared input file - the "Regresja
przekrojowa" sheet must be extracted from ff3_regresja_przekrojowa.xlsx
(the output of capm_3factors.py) into a separate file named
ff3_regresja_przekrojowa_2.xlsx.

Input: ff3_regresja_przekrojowa_2.xlsx (prepared manually)
Output: porownanie_R2_CAPM_FF3.png
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Wczytanie danych z wynikami regresji przekrojowej
df_regr = pd.read_excel('ff3_regresja_przekrojowa_2.xlsx') #proszę stworzyć nowy plik i go tutaj umieścić, tylko z arkuszem 'Regresja przekrojowa' z pliku ff3_regresja_przekrojowa

# 2. Przygotowanie wykresu
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=df_regr, 
    x='Zmienna zależna', 
    y='R²', 
    hue='Model',
    palette=['#ff7f0e', '#1f77b4'],
    edgecolor='black'
)

plt.title('Porównanie zdolności objaśniającej (R²) modeli CAPM i Famy-French (FF3)', fontsize=14, pad=15)
plt.xlabel('Zmienna zależna (Błędy prognoz poszczególnych modeli)', fontsize=12)
plt.ylabel('Współczynnik determinacji R²', fontsize=12)

vals = barplot.get_yticks()
barplot.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
plt.legend(title='Model objaśniający', loc='upper left')

for p in barplot.patches:
    height = p.get_height()
    if height > 0: 
        barplot.annotate(f'{height:.3f}', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', 
                         xytext=(0, 5), textcoords='offset points',
                         fontsize=10)

plt.tight_layout()
plt.savefig('porownanie_R2_CAPM_FF3.png', dpi=300, bbox_inches='tight')
plt.show()
