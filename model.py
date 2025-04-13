import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def forecast_temperature_lstm(df, future_hours=5, sequence_length=5):
    df = df.copy()
    df = df.sort_values("timestamp")  # just in case

    temps = df['temp'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps)

    X, y = create_sequences(temps_scaled, sequence_length)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)

    # Predict future values
    last_seq = temps_scaled[-sequence_length:]
    future_predictions = []
    current_input = last_seq

    current_input = current_input.reshape(sequence_length, 1)
    for _ in range(future_hours):
        pred = model.predict(current_input.reshape(1, sequence_length, 1), verbose=0)
        future_predictions.append(pred[0][0])

        # Update current_input (drop oldest, add new pred)
        new_input = np.append(current_input[1:], [[pred[0][0]]], axis=0)
        current_input = new_input.reshape(sequence_length, 1)


    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    last_time = df['timestamp'].max()
    future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=future_hours, freq='H')
    
    return pd.DataFrame({
        'timestamp': future_times,
        'predicted_temp': future_predictions
    })
