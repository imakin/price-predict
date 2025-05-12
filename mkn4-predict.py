import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from model import (
    create_the_model,
    split_into_sequences,
    get_train_test_sets,
    dataset,
)

# Divide the data into shorter-period sequences
seq_len = 60
batch_size = 16
dropout = 0.4
window_size = seq_len - 1
n_features = 4    # Sentiment, Close, UpperBolinger, LowerBolinger
n_steps = 48      # jumlah jam ke depan yang ingin diprediksi


# 1. Load model
the_model = create_the_model(window_size, dropout, n_features)
the_model.load_weights('model.weights.h5') 

# 2. Load dataset normalisasi 90 hari terakhir (tanpa kolom Datetime)
# scaler nanti dipakai untuk inverse
df, scaler = dataset(0, return_scaler=True)

print(df.tail())

# Fit scaler pada data 90 hari terakhir (harus sama dengan saat training)
cols_to_norm = ['Sentiment', 'Close', 'UpperBolinger', 'LowerBolinger']

# 4. Ambil sequence terakhir
sequence = df[cols_to_norm].values[-window_size:]  # shape: (window_size, n_features)

# 5. Simpan hasil prediksi
predictions = []

for step in range(n_steps):
    input_seq = np.expand_dims(sequence, axis=0)  # shape: (1, window_size, n_features)
    y_pred = the_model.predict(input_seq, verbose=0)
    pred_close = y_pred[0, 0]  # prediksi dalam skala normalisasi
    predictions.append(pred_close)
    
    # Buat next_row untuk step berikutnya
    # Fitur lain (Sentiment, UpperBolinger, LowerBolinger) bisa diisi dengan nilai jam terakhir
    next_row = sequence[-1].copy()
    # Update kolom 'Close' (asumsi urutan ke-1)
    next_row[1] = pred_close
    # Geser sequence
    sequence = np.vstack([sequence[1:], next_row])
predictions = np.array(predictions)  # shape: (48,)

# 6. Inverse transform ke harga asli (hanya kolom Close)
# Buat array dummy untuk inverse transform
dummy = np.tile(sequence[-1], (n_steps, 1))  # shape: (n_steps, n_features)
dummy[:, 1] = predictions  # isi kolom Close dengan prediksi
pred_close_real = scaler.inverse_transform(dummy)[:, 1]

# Debug prints
print("Urutan kolom:", cols_to_norm)
print("Nilai scaler min:", scaler.data_min_)
print("Nilai scaler max:", scaler.data_max_)
print("Prediksi (skala normalisasi):", predictions[:5])
print("Dummy sebelum inverse:\n", dummy[:5])
print("Harga Close terakhir (asli):", df['Close'].iloc[-1])

print("Hasil inverse transform (USD):", pred_close_real[:5])

# 7. Output hasil prediksi
for i, price in enumerate(pred_close_real, 1):
    print(f"Prediksi harga ke-{i} jam ke depan: {price:.2f} USD")
