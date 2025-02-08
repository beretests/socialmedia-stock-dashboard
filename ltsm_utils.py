import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def prepare_ltsm_data(data, window=30):
    data['lagged_return'] = data['daily_return'].shift(1)
    data['lagged_volume'] = data['v'].shift(1)
    data['moving_average'] = data['c'].rolling(window).mean()

    delta = data['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)

    features = ['daily_return', 'lagged_return',
                'lagged_volume', 'moving_average', 'RSI']
    target = 'v'
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target] = scaler.fit_transform(data[[target]])

    X, y = data[features].values, data[target].values
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(5, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=10, batch_size=16,
              validation_data=(X_test, y_test))
    return model, y_test


def plot_predictions(actual, predicted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual Volume'))
    fig.add_trace(go.Scatter(
        y=predicted, mode='lines', name='Predicted Volume'))
    return fig
