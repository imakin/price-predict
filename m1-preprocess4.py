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
file_fear_raw = 'csv/fear_raw.json'
file_fear_interpolated = 'csv/fear_clean.csv'
file_btc = 'csv/btc.csv'
file_btc4 = 'csv/btc4.csv'
file_btc12 = 'csv/btc12.csv'
file_btc24 = 'csv/btc24.csv'
file_merged = 'csv/merged.csv'
file_merged4 = 'csv/merged4.csv'
file_merged12 = 'csv/merged12.csv'
file_merged24 = 'csv/merged24.csv'

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




df_sentiment_4h = df_sentiment_hourly.set_index('Datetime').resample('4H').mean().reset_index()
df_sentiment_12h = df_sentiment_hourly.set_index('Datetime').resample('12H').mean().reset_index()
df_sentiment_24h = df_sentiment_hourly.set_index('Datetime').resample('24H').mean().reset_index()

print(df_sentiment_hourly.head())
print(df_sentiment_hourly.tail())

print("7. Simpan ke CSV")
df_sentiment_hourly.to_csv(file_fear_interpolated, index=False)
print(f"Fear & Greed Index per jam berhasil disimpan ke {file_fear_interpolated}")






"""
Ambil start & end datetime yang ada di df_sentiment_hourly
"""

start_datetime = df_sentiment_hourly['Datetime'].iloc[0]
end_datetime = df_sentiment_hourly['Datetime'].max()

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
        # end=(end_datetime + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        interval='1h'
    )#DateTime as index
    # btc.index = btc.index.tz_convert('Asia/Jakarta') # tetap pakai UTC

    # If DataFrame has MultiIndex columns, make it one line one idex!
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = ['_'.join([str(i) for i in col if i]) for col in btc.columns.values]
    btc.to_csv(file_btc, index=True)
    print(f"BTC data saved to {file_btc}")

"""
Drop trading volume, karena banyak nilai 0 yang tidak mencerminkan trading volume sebenarnya.
padahal volume biasanya berkaitan dengan kondisi market & sentimen
"""
btc.drop(columns=['Volume_BTC-USD'], inplace=True)

print(btc.columns)
print(btc.head())


# menambahkan data pada 
# df_sentiment_hourly,df_sentiment_4h,
# df_sentiment_12h,df_sentiment_24h 
# hingga jam terakhir yang tersedia
# pada dataframe `btc`, isi dengan nilai terakhir sebelumnya
# Ambil jam terakhir dari btc
last_btc_time = btc.index.max() #btc['Datetime'].max()

# Fungsi untuk extend DataFrame sentiment hingga jam terakhir btc
def extend_sentiment(df, freq, last_time):
    # Buat range baru dari waktu awal ke waktu akhir btc
    full_range = pd.date_range(df['Datetime'].min(), last_time, freq=freq)
    # Reindex dan isi nilai baru dengan ffill
    df_ext = df.set_index('Datetime').reindex(full_range).ffill().reset_index().rename(columns={'index': 'Datetime'})
    return df_ext

# Extend masing-masing DataFrame
df_sentiment_hourly = extend_sentiment(df_sentiment_hourly, 'H', last_btc_time)
df_sentiment_4h     = extend_sentiment(df_sentiment_4h, '4H', last_btc_time)
df_sentiment_12h    = extend_sentiment(df_sentiment_12h, '12H', last_btc_time)
df_sentiment_24h    = extend_sentiment(df_sentiment_24h, '24H', last_btc_time)

"""
Feature Engineering: 
Buat bolinger band dari dataframe btc-usd
"""
window = 7  # Typical window for Bolinger Bands and SMA
btc['SMA'] = btc['Close_BTC-USD'].rolling(window=window).mean()
# Standar Deviasi
btc['STD'] = btc['Close_BTC-USD'].rolling(window=window).std(ddof=1) # Bolinger Bands are designed to reflect recent price volatility. The rolling window is a moving sample, so using ddof=1 (Bessel’s correction) is statistically more appropriate for estimating the "true" volatility of the underlying process.
# Bolinger Bands
btc['UpperBolinger'] = btc['SMA'] + (2 * btc['STD'])
btc['LowerBolinger'] = btc['SMA'] - (2 * btc['STD'])
# Drop kolom standard deviasi
btc.drop(columns=['STD'], inplace=True)



btc = btc.reset_index()
btc4 = btc.set_index('Datetime').resample('4H').agg({
    'Close_BTC-USD': 'last',
    'UpperBolinger': 'last',
    'LowerBolinger': 'last',
}).reset_index()
btc4 = btc4.dropna()
btc12 = btc.set_index('Datetime').resample('12H').agg({
    'Close_BTC-USD': 'last',
    'UpperBolinger': 'last',
    'LowerBolinger': 'last',
}).reset_index()
btc12 = btc12.dropna()
btc24 = btc.set_index('Datetime').resample('24H').agg({
    'Close_BTC-USD': 'last',
    'UpperBolinger': 'last',
    'LowerBolinger': 'last',
}).reset_index()
# reset index jadikan index datetime menjadi kolom biasa
btc24 = btc24.dropna()

"""
19 data head tidak bisa hitung bolinger bandnya, hapus saja
"""
btc = btc.dropna(subset=['UpperBolinger', 'LowerBolinger'])
btc4 = btc4.dropna(subset=['UpperBolinger', 'LowerBolinger'])
btc12 = btc12.dropna(subset=['UpperBolinger', 'LowerBolinger'])
btc24 = btc24.dropna(subset=['UpperBolinger', 'LowerBolinger'])

print(btc.tail(30))

"""
gabung semua fitur yang mau dipakai (sentimen dan harga) jadi satu 
"""
print(btc.columns)

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


btc4 = btc4.reset_index()
# Merge on Datetime (inner join to keep only matching timestamps)
merged4 = pd.merge(
    df_sentiment_4h,
    btc4[['Datetime', 'Close_BTC-USD', 'UpperBolinger', 'LowerBolinger']],
    on='Datetime',
    how='inner'
)
merged4 = merged4.rename(columns={
    'Close_BTC-USD': 'Close',
})
merged4.to_csv(file_merged4, index=False)
print(merged4.head())


btc12 = btc12.reset_index()
# Merge on Datetime (inner join to keep only matching timestamps)
merged12 = pd.merge(
    df_sentiment_12h,
    btc12[['Datetime', 'Close_BTC-USD', 'UpperBolinger', 'LowerBolinger']],
    on='Datetime',
    how='inner'
)
merged12 = merged12.rename(columns={
    'Close_BTC-USD': 'Close',
})
merged12.to_csv(file_merged12, index=False)
print(merged12.head())

btc24 = btc24.reset_index()
# Merge on Datetime (inner join to keep only matching timestamps)
merged24 = pd.merge(
    df_sentiment_24h,
    btc24[['Datetime', 'Close_BTC-USD', 'UpperBolinger', 'LowerBolinger']],
    on='Datetime',
    how='inner'
)
merged24 = merged24.rename(columns={
    'Close_BTC-USD': 'Close',
})
merged24.to_csv(file_merged24, index=False)
print(merged24.head())
