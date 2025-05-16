import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

"""
buat dataset, 90 hari,
dan normalisasikan data dengan MinMaxScaler (dari 90 hari itu)
"""

# Load data hasil merge
final_df = pd.read_csv('merged.csv', parse_dates=['Datetime'])

# Pastikan urut waktu
final_df = final_df.sort_values('Datetime')

# Ambil 90 hari terakhir
last_date = final_df['Datetime'].max()
first_date = last_date - pd.Timedelta(days=90)
df90 = final_df[final_df['Datetime'] >= first_date].copy()

# Kolom yang ingin dinormalisasi
cols_to_norm = ['Sentiment', 'Close', 'UpperBolinger', 'LowerBolinger']
scaler = MinMaxScaler()
df90[cols_to_norm] = scaler.fit_transform(
    df90[cols_to_norm]
)

df90.to_csv('d90_normalized.csv', index=False)
print(df90.head())


# Plot each feature

plt.figure(figsize=(12, 6))
for col in ['Sentiment', 'Close', 'UpperBolinger', 'LowerBolinger']:
    plt.plot(df90.index, df90[col], label=col)
plt.title('Scaled Price & Sentiment Features')
plt.xlabel('Sample Index')
plt.ylabel('Scaled Value')
plt.legend()
plt.tight_layout()
plt.show()


sns.heatmap(df90[['Sentiment', 'Close','UpperBolinger', 'LowerBolinger']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()