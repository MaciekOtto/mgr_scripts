import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Wczytanie danych 
df = pd.read_excel('ff3_regresja_przekrojowa_vis.xlsx')
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

color_garch = '#1f77b4' # Niebieski
color_rf = 'gray'    # Zielony

# Wykres 1: GARCH vs beta_smb
sns.regplot(data=df, x='beta_smb', y='GARCH', ax=axes[0, 0], 
            scatter_kws={'alpha':0.4, 's':15}, line_kws={'color':'red', 'linewidth':2}, color=color_garch)
axes[0, 0].set_title('Błąd modelu GARCH a ekspozycja na czynnik Size (SMB)')
axes[0, 0].set_xlabel('Beta SMB')
axes[0, 0].set_ylabel('RMSE (GARCH)')

# Wykres 2: GARCH vs beta_hml
sns.regplot(data=df, x='beta_hml', y='GARCH', ax=axes[0, 1], 
            scatter_kws={'alpha':0.4, 's':15}, line_kws={'color':'red', 'linewidth':2}, color=color_garch)
axes[0, 1].set_title('Błąd modelu GARCH a ekspozycja na czynnik Value (HML)')
axes[0, 1].set_xlabel('Beta HML')
axes[0, 1].set_ylabel('RMSE (GARCH)')

# Wykres 3: RF vs beta_smb
sns.regplot(data=df, x='beta_smb', y='RF', ax=axes[1, 0], 
            scatter_kws={'alpha':0.4, 's':15}, line_kws={'color':'red', 'linewidth':2}, color=color_rf)
axes[1, 0].set_title('Błąd modelu RF a ekspozycja na czynnik Size (SMB)')
axes[1, 0].set_xlabel('Beta SMB')
axes[1, 0].set_ylabel('RMSE (Random Forest)')

# Wykres 4: RF vs beta_hml
sns.regplot(data=df, x='beta_hml', y='RF', ax=axes[1, 1], 
            scatter_kws={'alpha':0.4, 's':15}, line_kws={'color':'red', 'linewidth':2}, color=color_rf)
axes[1, 1].set_title('Błąd modelu RF a ekspozycja na czynnik Value (HML)')
axes[1, 1].set_xlabel('Beta HML')
axes[1, 1].set_ylabel('RMSE (Random Forest)')

plt.tight_layout()
plt.savefig('scatter_garch_rf.png', dpi=300, bbox_inches='tight')
plt.show()
