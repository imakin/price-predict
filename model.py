from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from datetime import datetime



# Load data hasil merge
df_full1 = pd.read_csv('csv/merged.csv', parse_dates=['Datetime'])
# Pastikan urut waktu
df_full1 = df_full1.sort_values('Datetime')

df_full4 = pd.read_csv('csv/merged4.csv', parse_dates=['Datetime'])
df_full4 = df_full4.sort_values('Datetime')

df_full12 = pd.read_csv('csv/merged12.csv', parse_dates=['Datetime'])
df_full12 = df_full12.sort_values('Datetime')

df_full24 = pd.read_csv('csv/merged24.csv', parse_dates=['Datetime'])
df_full24 = df_full24.sort_values('Datetime')

def create_the_model(window_size, dropout, n_feature):
    the_model = Sequential()
    the_model.add(LSTM(window_size, return_sequences=True,input_shape=(window_size, n_feature)))
    the_model.add(Dropout(rate=dropout))
    the_model.add(Bidirectional(LSTM((window_size * 2), return_sequences=True)))
    the_model.add(Dropout(rate=dropout))
    the_model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    the_model.add(Dense(units=1)) # banyak output neuron
    the_model.add(Activation('linear'))
    the_model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.0005))
    return the_model


def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    x_train = sequences[:n_train, :-1, :]
    x_test = sequences[n_train:, :-1, :]
    y_train = sequences[:n_train, -1, :]
    y_test = sequences[n_train:, -1, :]
    return x_train, y_train, x_test, y_test


# buat dataset initial
# Ambil 90 hari terakhir
def dataset(end,days=90, return_scaler=False, candlehr=1):
    # start dan end nilai hari terakhir
    # misal dataset(0,days=90) maka akan mengambil data 90 hari terakhir sampai hari ini
    # misal dataset(-90,days=90) maka data 180 hari sampai 90 hari terakhir
    if candlehr==1:
        df_full = df_full1
    elif candlehr==4:
        df_full = df_full4
    elif candlehr==12:
        df_full = df_full12
    elif candlehr==24:
        df_full = df_full24
    else:
        raise ValueError("candlehr harus 1, 4, 12, atau 24")

    last_date = df_full['Datetime'].max() - pd.Timedelta(days=-end)
    first_date = last_date - pd.Timedelta(days=(days))
    print(f"from {first_date} to {last_date}")
    df90 = df_full[(df_full['Datetime'] >= first_date) & (df_full['Datetime'] <= last_date)].copy()
    print(df90.tail())

    # Sentimen dinormalisasi ke 0~1 bukan relatif ke dataset min max
    df90['Sentiment'] = (df90['Sentiment'] / 100) - 0.5
    # Kolom lain normalisasi minmax dalam satu dataset
    cols_to_norm = ['Close', 'UpperBolinger', 'LowerBolinger']
    scaler = MinMaxScaler(feature_range=(-1,1))
    try:
        df90[cols_to_norm] = scaler.fit_transform(
            df90[cols_to_norm]
        )
    except Exception as e:
        print(df90.head())
        print(df90.tail())
        raise e
    df90 = df90.drop(columns=['Datetime'])
    print('setelah normalisasi')
    print(df90.head())
    print(df90.tail())
    if return_scaler:
        return df90, scaler
    return df90