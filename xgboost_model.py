import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import streamlit as st
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.base import BaseEstimator, RegressorMixin
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

scaler = MinMaxScaler()


class SklearnXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.01,
                 colsample_bytree=0.3, alpha=0, subsample=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.alpha = alpha
        self.subsample = subsample
        self.model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            colsample_bytree=self.colsample_bytree,
            alpha=self.alpha,
            subsample=self.subsample
        )

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)


def prepare_xgboost_data(data):
    df = data.copy()
    df['SMA_20'] = ta.trend.sma_indicator(
        df['c'], window=20)  # 20-day Simple Moving Average
    # 20-day Exponential Moving Average
    df['EMA_20'] = ta.trend.ema_indicator(df['c'], window=20)
    # Relative Strength Index (14-day)
    df['RSI_14'] = ta.momentum.rsi(df['c'], window=14)
    df['Volatility'] = ta.volatility.bollinger_hband(
        df['c'], window=20) - ta.volatility.bollinger_lband(df['c'], window=20)  # Bollinger Band Width

    # MACD (Moving Average Convergence Divergence)
    df["MACD"] = ta.trend.macd(df["c"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["c"])

    # ADX (Average Directional Index)
    df["ADX"] = ta.trend.adx(df["h"], df["l"], df["c"])

    # OBV (On-Balance Volume)
    df["OBV"] = ta.volume.on_balance_volume(df["c"], df["v"])

    # ATR (Average True Range) - Measures volatility
    df["ATR"] = ta.volatility.average_true_range(df["h"], df["l"], df["c"])

    df.dropna(inplace=True)

    features = ['SMA_20', 'EMA_20', 'RSI_14', 'Volatility',
                'c', 'MACD', 'MACD_Signal', 'ADX', 'OBV', 'ATR', 'o', 'h', 'l', 'v']
    df_data = df[features].astype(float)

    scaled_data = scaler.fit_transform(df_data)

    return scaled_data


def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:(i + lookback)])
        y.append(dataset[i + lookback, 4])
    X, y = np.array(X), np.array(y)
    return X, y


def split_xgboost_data(data, test_size, lookback):
    if len(data) == 0:
        raise ValueError(
            "The input data is empty. Ensure that the data is correctly loaded and preprocessed.")

    X, y = create_dataset(data, lookback)
    test_size = test_size/100
    split = int(len(X) * (1 - test_size))
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    if X_train.size == 0 or X_valid.size == 0:
        raise ValueError(
            "The training or validation data is empty after splitting. Adjust the test_size or lookback parameters.")

    # Reshape X_train and X_valid to 2D
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)

    return X_train, X_valid, y_train, y_valid


def optimize_xgboost_hyperparams(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 300, 500],  # Number of boosting rounds
        "max_depth": [3, 5, 7],  # Tree depth
        "learning_rate": [0.01, 0.1, 0.3],  # Step size
        "colsample_bytree": [0.3, 0.7, 1.0],  # Fraction of features per tree
        "alpha": [0, 10, 100],  # L1 regularization
        "subsample": [0.8, 1.0]  # Fraction of samples per boosting round
    }

    xgb_model = SklearnXGBRegressor()

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # Using MSE as evaluation metric
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1  # Use all available CPU cores
    )

    # Run the grid search
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    # Convert to positive MSE
    print(f"Best MSE Score: {-grid_search.best_score_:.4f}")

    return grid_search.best_params_, -grid_search.best_score_


def train_xgboost_model(X_train, y_train):
    start_time = time.time()

    model = xgb.XGBRegressor(objective='reg:squarederror', alpha=0, colsample_bytree=0.3,
                             learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8)
    model.fit(X_train, y_train)

    end_time = time.time()
    execution_time = end_time - start_time
    return model, execution_time


def xgb_predict_future_prices(xgb_model, last_n_days, future_n_days, lookback):
    if last_n_days.shape != (lookback, 14):
        raise ValueError(
            f"Expected last_n_days to have shape ({lookback}, 14), but got {last_n_days.shape}")

    future_prices = []

    for _ in range(future_n_days):
        last_n_days_reshaped = last_n_days.reshape(1, -1)

        pred_price = xgb_model.predict(last_n_days_reshaped)
        future_prices.append(pred_price[0])
        last_n_days = np.roll(last_n_days, -1, axis=0)
        last_n_days[-1] = pred_price

    future_prices = np.array(future_prices).reshape(-1, 1)
    dummy = np.zeros((future_prices.shape[0], 14))
    dummy[:, 0] = future_prices[:, 0]

    future_prices = scaler.inverse_transform(dummy)[:, 0]
    return future_prices


def plot_actual_forecast(data, future_prices, future_n_days):
    df = data.copy()
    dates = df.index
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=df['c'],
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue')
    ))

    print("Future prices: ", future_prices)
    last_actual_price = df['c'].iloc[-1]
    print("Last Actual Price: ", last_actual_price)
    future_prices_with_last_actual = np.insert(
        future_prices, 0, last_actual_price)
    future_dates = pd.date_range(
        start=dates[-1], periods=future_n_days + 1, freq='D')[1:]
    print("Future prices with Last Actual: ", future_prices_with_last_actual)

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices_with_last_actual,
        mode='lines',
        line=dict(color='red'),
        name='Forecasted Prices'
    ))

    fig.update_layout(
        title=f'Actual and Forecasted Prices for Next {future_n_days} Days using XGBoost',
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        legend_title='Legend'
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, r2


def prepare_lstm_data(df):
    df_lstm = df.copy()
    data = df_lstm.filter(['c'])
    last_date = df_lstm.index[-1]
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    train = data[:training_data_len]
    valid = data[training_data_len:]

    y_test = dataset[training_data_len:, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, training_data_len, y_test, scaler, last_date, train, valid


def create_lstm_dataset(scaled_data, training_data_len, lookback):
    train_data = scaled_data[0:int(training_data_len), :]
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    x_train = []
    y_train = []

    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i-lookback:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i-lookback:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, x_test, y_train


def build_lstm_model(x_train, y_train, epochs):
    lstm_start_time = time.time()

    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=epochs)
    lstm_end_time = time.time()
    lstm_execution_time = lstm_end_time - lstm_start_time
    return model, lstm_execution_time


def lstm_predict_prices(model, x_test, scaler, last_n_days, future_n_days, lookback, last_date):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    next_n_days = []

    for _ in range(future_n_days):
        X_test = np.array([last_n_days])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price_unscaled = scaler.inverse_transform(pred_price)
        next_n_days.append(pred_price_unscaled[0, 0])
        last_n_days = np.append(last_n_days, pred_price, axis=0)[-lookback:]

    future_dates = [last_date + timedelta(days=i)
                    for i in range(1, future_n_days + 1)]
    future_predictions = pd.DataFrame(
        data={'Date': future_dates, 'Close': next_n_days})
    future_predictions.set_index('Date', inplace=True)

    return predictions, future_predictions


def plot_lstm_predictions(train, valid, future_predictions):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=train.index, y=train['c'], name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['c'], name='Val'))
    fig.add_trace(go.Scatter(
        x=valid.index, y=valid['Predictions'], name='Predictions'))
    fig.add_trace(go.Scatter(x=future_predictions.index,
                  y=future_predictions['Close'], name='Future', line=dict(dash='dash', color='red')))

    fig.update_layout(
        title='Model',
        xaxis_title='Date',
        yaxis_title='Close Price USD ($)',
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1),
        width=1000,
        height=600
    )

    fig.update_xaxes(title_font=dict(size=18))
    fig.update_yaxes(title_font=dict(size=18))

    st.plotly_chart(fig, use_container_width=True)


def evaluate_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R² Score': r2
    }


def display_model_metrics(model_name, metrics):
    st.markdown(f"##### {model_name} Metrics")

    cols = st.columns(4, gap='large')

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
