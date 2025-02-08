import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def preprocess_data(df, lookback=60):
    """Create features and scale data"""
    df['MA20'] = df['v'].rolling(window=20).mean()
    delta = df['v'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    for lag in [1, 2, 3]:
        df[f'return_lag{lag}'] = df['daily_return'].shift(lag)
        df[f'volume_lag{lag}'] = df['v'].shift(lag)

    df = df.dropna()

    features = ['MA20', 'RSI', 'return_lag1', 'return_lag2',
                'return_lag3', 'volume_lag1', 'volume_lag2', 'volume_lag3']
    target = 'v'

    if df.empty:
        raise ValueError(
            "The input DataFrame is empty. Ensure that the data is correctly loaded and preprocessed.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features + [target]])

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, :-1])
        y.append(scaled_data[i, -1])

    return np.array(X), np.array(y), scaler, features


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def show_metrics(y_test_l, test_predict):
    st.markdown(f"##### LSTM Metrics for Volume")
    cols = st.columns(4, gap='large')
    metrics = {
        "MAE": mean_absolute_error(y_test_l, test_predict),
        "MSE": mean_squared_error(y_test_l, test_predict),
        "RMSE": np.sqrt(mean_squared_error(y_test_l, test_predict)),
        "R² Score": r2_score(y_test_l, test_predict)
    }

    for i, (metric_name, value) in enumerate(metrics.items()):
        color = "#ADD8E6"
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0; color: #000000;">
                    <strong>{metric_name}:</strong> {value:.4f}
                </div>
                """,
                unsafe_allow_html=True)
