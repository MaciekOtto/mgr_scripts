import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie stylu
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Wczytanie danych z wynikami regresji
df_regr = pd.read_excel('ff3_regresja_przekrojowa_2.xlsx')

# 2. Przygotowanie wykresu
plt.figure(figsize=(10, 6))

# Zwróć uwagę na dokładną nazwę kolumny "R²" - może różnić się zależnie od kodowania pliku.
# Jeśli wyskoczy błąd o braku kolumny, sprawdź w CSV czy nazywa się "R2" czy "R²".
barplot = sns.barplot(
    data=df_regr, 
    x='Zmienna zależna', 
    y='R²', 
    hue='Model',
    palette=['#ff7f0e', '#1f77b4'], # Pomarańczowy dla FF3, Niebieski dla CAPM (dostosuj zależnie od kolejności w CSV)
    edgecolor='black'
)

# 3. Formatowanie i opisy
plt.title('Porównanie zdolności objaśniającej (R²) modeli CAPM i Famy-French (FF3)', fontsize=14, pad=15)
plt.xlabel('Zmienna zależna (Błędy prognoz poszczególnych modeli)', fontsize=12)
plt.ylabel('Współczynnik determinacji R²', fontsize=12)

# Zamiana osi Y na procenty (np. 0.08 -> 8%)
vals = barplot.get_yticks()
barplot.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

# Dostosowanie legendy
plt.legend(title='Model objaśniający', loc='upper left')

# Dodanie wartości liczbowych nad słupkami (opcjonalnie, ale wygląda profesjonalnie)
for p in barplot.patches:
    height = p.get_height()
    if height > 0: # pomija puste słupki
        barplot.annotate(f'{height:.3f}', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', 
                         xytext=(0, 5), textcoords='offset points',
                         fontsize=10)

plt.tight_layout()
plt.savefig('porownanie_R2_CAPM_FF3.png', dpi=300, bbox_inches='tight')
plt.show()
