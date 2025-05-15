import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from model import (
    create_the_model,
    split_into_sequences,
    get_train_test_sets,
    dataset,
    df_full1,
)

# Divide the data into shorter-period sequence1s
seq_len = 100
batch_size = 16
dropout = 0.4
window_size = seq_len - 1
n_features = 4    # Sentiment, Close, UpperBolinger, LowerBolinger
# n_steps = 60      # jumlah jam ke depan yang ingin diprediksi


# 1. Load model
# the_model = create_the_model(window_size, dropout, n_features)
# the_model.load_weights('model.weights.h5') 
the_model1 = load_model('keras/model1_4.keras')
the_model4 = load_model('keras/model4_9.keras')
the_model12 = load_model('keras/model12_19.keras')
the_model24 = load_model('keras/model24_34.keras')
the_models = {
    1:the_model1,
    4:the_model4, 
    12:the_model12, 
    24:the_model24
}

dataset_param_end = 0 #hari terakhir yang dipakai adalah hari max dikurangi nilai ini
# 2. Load dataset normalisasi 90 hari terakhir (tanpa kolom Datetime)
# scaler nanti dipakai untuk inverse
df1,  scaler1  = dataset(dataset_param_end, days=180, return_scaler=True,candlehr=1)
df4,  scaler4  = dataset(dataset_param_end, days=180, return_scaler=True,candlehr=4)
df12, scaler12 = dataset(dataset_param_end, days=180, return_scaler=True,candlehr=12)
df24, scaler24 = dataset(dataset_param_end, days=180, return_scaler=True,candlehr=24)

dfs = {
    1:df1,
    4:df4, 
    12:df12, 
    24:df24
}
scalers = {
    1:scaler1,
    4:scaler4, 
    12:scaler12, 
    24:scaler24
}

# print(df1.tail())

# Fit scaler pada data 90 hari terakhir (harus sama dengan saat training)
cols_to_norm = ['Sentiment', 'Close', 'UpperBolinger', 'LowerBolinger']

n_steps = 1 # berapa sequence / step prediksi
prediksi_total = {}
for varian_jam in [1,4,12,24]:
    # 4. Ambil sequence1 terakhir
    sequence = dfs[varian_jam][cols_to_norm].values[-window_size:]  # shape: (window_size, n_features)
    the_model = the_models[varian_jam]

    # 5. Simpan hasil prediksi
    predictions = []

    for step in range(n_steps):
        input_seq = np.expand_dims(sequence, axis=0)  # shape: (1, window_size, n_features)
        y_pred = the_model.predict(input_seq, verbose=0)
        print(y_pred)
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
    # inverse transform
    predictions_to_unscale = np.column_stack([
        predictions,
        np.full_like(predictions, 0),
        np.full_like(predictions, 0),
    ])
    pred_real = scalers[varian_jam].inverse_transform(predictions_to_unscale)[:, 0]
    print(f"{varian_jam}: {pred_real}")
    prediksi_total[varian_jam] = pred_real[0]



# Buat array kosong untuk jam 1-24
pred_per_jam = np.zeros(24)
jam_tersedia = sorted(prediksi_total.keys())
harga_tersedia = [prediksi_total[j] for j in jam_tersedia]

# Interpolasi linear untuk jam 1-24
for jam in range(1, 25):
    if jam in prediksi_total:
        pred_per_jam[jam-1] = prediksi_total[jam]
    else:
        # Cari dua titik terdekat untuk interpolasi
        for i in range(len(jam_tersedia)-1):
            if jam_tersedia[i] < jam < jam_tersedia[i+1]:
                x0, x1 = jam_tersedia[i], jam_tersedia[i+1]
                y0, y1 = prediksi_total[x0], prediksi_total[x1]
                # Linear interpolation
                pred_per_jam[jam-1] = y0 + (y1-y0)*(jam-x0)/(x1-x0)
                break

# Print hasil prediksi jam 1-24
for jam in range(1, 25):
    print(f"Prediksi harga ke-{jam} jam ke depan: {pred_per_jam[jam-1]:.2f} USD")




import matplotlib.pyplot as plt

# Ambil 48 jam terakhir dari data asli
n_history = 48
# df1['Close'] masih dalam skala normalisasi, ambil harga asli
close_norm_hist = df1['Close'].values[-n_history:]
# Inverse transform ke harga asli
dummy_hist = np.column_stack([
    close_norm_hist,
    np.full_like(close_norm_hist, 0),
    np.full_like(close_norm_hist, 0),
])
close_real_hist = scaler1.inverse_transform(dummy_hist)[:, 0]

# Sumbu x untuk data historis: -47, ..., 0
x_hist = np.arange(-n_history+1, 1)  # panjang = n_history
# Sumbu x untuk prediksi: 1, ..., 24
x_pred = np.arange(1, 25)            # panjang = 24

plt.figure(figsize=(12,6))
plt.plot(x_hist, close_real_hist, label='Harga Asli (1 Jam)', color='blue')
plt.plot(x_pred, pred_per_jam, label='Prediksi 1-24 Jam ke Depan', color='red', marker='o')
plt.axvline(0, color='gray', linestyle='--', label='Waktu Prediksi')

#jam saat ini yang dipakai untuk prediksi (pemanggilan model.dataset)
sekarang = df_full1['Datetime'].max() - pd.Timedelta(days=-dataset_param_end)
plt.xlabel(f'Jam ke Depan (0 = {sekarang})')
plt.ylabel('Harga (USD)')
plt.title('Harga Asli dan Prediksi 1-24 Jam ke Depan')
plt.legend()
plt.tight_layout()
plt.show()


"""
# 6. Inverse transform ke harga asli
#harga yang diketahui sebelumnya
close_norm = df1['Close'].iloc[-1]
upper_norm = df1['UpperBolinger'].iloc[-1]
lower_norm = df1['LowerBolinger'].iloc[-1]
dummy = np.array([[close_norm, upper_norm, lower_norm]])
close_real = scaler.inverse_transform(dummy)[0, 0]
print(f"harga yang diketahui terakhir scaled: {close_norm} inverse scale: {close_real}")


# Ambil nilai terakhir UpperBolinger & LowerBolinger
last_upper = sequence1[-1, 2]
last_lower = sequence1[-1, 3]
# Gabungkan prediksi dengan nilai tetap untuk kolom lain
dummy = np.column_stack([
    predictions,                     # prediksi Close (n_steps,)
    np.full_like(predictions, last_upper),  # isi dengan nilai terakhir UpperBolinger
    np.full_like(predictions, last_lower),  # isi dengan nilai terakhir LowerBolinger
])
# Inverse transform
pred_close_real = scaler.inverse_transform(dummy)[:, 0]

# Debug prints
print("Harga Close terakhir (asli):", df1['Close'].iloc[-1])
print("Hasil inverse transform (USD):", pred_close_real[:5])

# 7. Output hasil prediksi
for i, price in enumerate(pred_close_real, 1):
    print(f"Prediksi harga ke-{i} jam ke depan: {price:.2f} USD")
"""