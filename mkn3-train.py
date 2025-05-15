from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime

from model import (
    create_the_model,
    split_into_sequences,
    get_train_test_sets,
    dataset,
)

warnings.filterwarnings('ignore')


# Divide the data into shorter-period sequences
dataset_days = 180
seq_len = 120
batch_size = 32
dropout = 0.4
window_size = seq_len - 1
n_features = 4    # Sentiment, Close, UpperBolinger, LowerBolinger


# Build a 3-layer LSTM RNN
the_model = create_the_model(window_size, dropout, n_features)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


# Define the model heckpoint callback
checkpoint_filepath = 'model.weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

for fit_repeat in range(20):
    end_day = np.random.randint(-364+dataset_days,0)
    df = dataset(end_day, days=dataset_days)
    x_train, y_train, x_test, y_test = get_train_test_sets(df, seq_len, train_frac=0.9)
    # Ambil hanya kolom target
    target_idx = df.columns.get_loc('Close')
    y_train = y_train[:, target_idx].reshape(-1, 1)
    y_test = y_test[:, target_idx].reshape(-1, 1)

    history = the_model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.2,
        callbacks=[model_checkpoint_callback, early_stop]
    )
    the_model.load_weights(checkpoint_filepath)
    the_model.summary()

    test_loss = the_model.evaluate(x_test, y_test)
    print("Test loss:", test_loss)

    y_pred = the_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Test MSE:", mse)

    plt.plot(y_test, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title(f'Refit #{fit_repeat} Day: {end_day-90} ~ {end_day}')
    plt.savefig(f'train_prediction_{fit_repeat}.png')
    plt.close()  # Close the figure to free memory if in a loop
    
    # simpan model dengan history fit_repeat
    the_model.save(f'model_{fit_repeat}.h5')
    print(f"Model saved to model_{fit_repeat}.h5")

#test
