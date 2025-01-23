import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configurations globales
EPOCHS = 20
TEST_SIZE = 0.2
TIME_STEP = 24
BATCH_SIZE = 32
HIDDEN_UNITS = 64

# Charger les données
def load_data(file_path):
    weather = pd.read_csv(file_path)
    weather['datetime'] = pd.to_datetime(weather['Formatted Date'], utc=True)
    weather = weather[['datetime', 'Temperature (C)']]
    weather = weather.rename({'Temperature (C)': 'temp'}, axis=1)
    weather = weather.sort_values('datetime').reset_index(drop=True)
    return weather

# Prétraitement des données
def preprocess_data(weather):
    Y_filt = np.arange(TIME_STEP, weather.shape[0], TIME_STEP)
    y_df = weather.iloc[Y_filt]
    X_temp = weather.iloc[:len(y_df) * TIME_STEP]
    X = np.reshape(X_temp['temp'].values, (y_df.shape[0], TIME_STEP, 1))
    X = X[:, :TIME_STEP-1]
    return X, y_df

# Création du modèle
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(units=HIDDEN_UNITS, input_shape=input_shape, activation='tanh'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Visualisation des résultats
def plot_results(history, y_train, y_train_pred, y_test, y_test_pred, y_df):
    # Courbes de perte
    hist_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    hist_df.plot(y=['loss', 'val_loss'], ax=axes[0])
    axes[0].set_title("Loss")

    hist_df.plot(y=['root_mean_squared_error', 'val_root_mean_squared_error'], ax=axes[1])
    axes[1].set_title("RMSE")

    # Comparaison des prédictions
    y_pred = np.append(y_train_pred, y_test_pred)
    y_df['pred'] = y_pred
    fig, ax = plt.subplots(figsize=(15, 6))
    y_df.plot(x='datetime', y=['temp', 'pred'], ax=ax)
    plt.show()

# Pipeline principal
if __name__ == "__main__":
    weather = load_data("../data/weather_history.csv")
    X, y_df = preprocess_data(weather)
    split = int(len(y_df) * (1 - TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_df['temp'].values[:split], y_df['temp'].values[split:]
    
    model = create_model((TIME_STEP-1, 1))
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    plot_results(history, y_train, y_train_pred, y_test, y_test_pred, y_df)
