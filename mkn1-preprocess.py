import json
import requests

import pandas as pd
import yfinance as yf

"""
DATA SENTIMENT
1. Pakai Sentiment dari fear&greed index (daily),
2. kemudian urutkan,  
3. interpolate linear menjadi data per jam

"""
file_fear_raw = 'fear_raw.json'
file_fear_interpolated = 'csv/fear_clean.csv'
file_btc = 'csv/btc.csv'
file_btc_clean = 'csv/btc_clean.csv'
file_merged = 'csv/merged.csv'

print("1. Ambil data Fear & Greed Index harian dari API")
try:
    with open(file_fear_raw) as f:
        data = json.loads(f.read())
    print('file sudah didownload')
except FileNotFoundError:
    url = "https://api.alternative.me/fng/?limit=365"
    resp = requests.get(url)
    data = json.loads(resp.content)
    #save resp to json
    with open(file_fear_raw, 'wb') as f:
        f.write(resp.content)

df = pd.DataFrame(data['data'])

print("2. Ubah kolom timestamp ke datetime (UTC)")
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
df = df.sort_values('timestamp')

print("3. Pilih kolom yang dibutuhkan dan ubah ke float")
df['value'] = df['value'].astype(float)
df = df[['timestamp', 'value']].rename(columns={'timestamp': 'Datetime', 'value': 'Sentiment'})

print("4. Set Datetime sebagai index")
df.set_index('Datetime', inplace=True)

print("5. Buat range per jam dari waktu paling awal ke paling akhir")
full_range = pd.date_range(df.index.min(), df.index.max(), freq='H')

print("6. Reindex dan interpolasi linear per jam")
df_sentiment_hourly = df.reindex(full_range)


"""
Feature engineering:
interpolate hourly fear&greed index from daily
"""
df_sentiment_hourly['Sentiment'] = df_sentiment_hourly['Sentiment'].interpolate(method='linear')
df_sentiment_hourly = df_sentiment_hourly.reset_index().rename(columns={'index': 'Datetime'})

print(df_sentiment_hourly.head())
print(df_sentiment_hourly.tail())

print("7. Simpan ke CSV")
df_sentiment_hourly.to_csv(file_fear_interpolated, index=False)
print(f"Fear & Greed Index per jam berhasil disimpan ke {file_fear_interpolated}")






"""
Ambil start & end datetime yang ada di df_sentiment_hourly
"""

start_datetime = df_sentiment_hourly['Datetime'].iloc[0]
end_datetime = df_sentiment_hourly['Datetime'].iloc[-1]

print(f"Start datetime: {start_datetime}")
print(f"End datetime: {end_datetime}")




"""
Download data BTC-USD dari yfinance
"""

try:
    btc = pd.read_csv(file_btc, index_col=0, parse_dates=True) #index0 adalah DateTime
except FileNotFoundError:
    btc = yf.download( #sudah berbentuk DataFrame
        'BTC-USD',
        start=start_datetime.strftime('%Y-%m-%d'),
        end=end_datetime.strftime('%Y-%m-%d'),
        interval='1h'
    )#DateTime as index

    # If DataFrame has MultiIndex columns, make it one line one idex!
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = ['_'.join([str(i) for i in col if i]) for col in btc.columns.values]
    btc.to_csv(file_btc, index=True)
    print(f"BTC data saved to {file_btc}")

print(btc.columns)
print(btc.head())
"""
Feature Engineering: 
Buat bolinger band dari dataframe btc-usd
"""
window = 20  # Typical window for Bolinger Bands and SMA
btc['SMA'] = btc['Close_BTC-USD'].rolling(window=window).mean()
# Standar Deviasi
btc['STD'] = btc['Close_BTC-USD'].rolling(window=window).std(ddof=1) # Bolinger Bands are designed to reflect recent price volatility. The rolling window is a moving sample, so using ddof=1 (Bessel’s correction) is statistically more appropriate for estimating the "true" volatility of the underlying process.

# Bolinger Bands
btc['UpperBolinger'] = btc['SMA'] + (2 * btc['STD'])
btc['LowerBolinger'] = btc['SMA'] - (2 * btc['STD'])
# Drop kolom standard deviasi
btc.drop(columns=['STD'], inplace=True)


"""
19 data head tidak bisa hitung bolinger bandnya, hapus saja
"""
btc.drop(btc.index[:19], inplace=True)


print(btc.tail(30))


"""
Drop trading volume, karena banyak nilai 0 yang tidak mencerminkan trading volume sebenarnya.
padahal volume biasanya berkaitan dengan kondisi market & sentimen
"""
btc.drop(columns=['Volume_BTC-USD'], inplace=True)


#simpan
btc.to_csv(file_btc_clean, index=True)


"""
gabung semua fitur yang mau dipakai (sentimen dan harga) jadi satu 
"""
print(btc.columns)

# jadikan index datetime menjadi kolom biasa
btc = btc.reset_index()
# Merge on Datetime (inner join to keep only matching timestamps)
merged = pd.merge(
    df_sentiment_hourly,
    btc[['Datetime', 'Close_BTC-USD', 'UpperBolinger', 'LowerBolinger']],
    on='Datetime',
    how='inner'
)
merged = merged.rename(columns={
    'Close_BTC-USD': 'Close',
})
merged.to_csv(file_merged, index=False)
print(merged.head())