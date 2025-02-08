import streamlit as st
from prophet import Prophet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def prepare_stock_data(symbol_data, target_col='c'):
    data = symbol_data
    data.dropna(inplace=True)
    y = data[target_col].dropna().values

    X = data[['o', 'h', 'l', 'v']].dropna().values

    return X, y, data


def split_data(X, y, data, test_size=0.2):
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size + 1], data.iloc[train_size:]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test, X_train, train, test


def train_arima_model(train_data, order=(1, 1, 1)):
    arima_model = ARIMA(train_data, order=order)
    arima_model = arima_model.fit()
    return arima_model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1):
    model = xgb.XGBRegressor(n_estimators=n_estimators,
                             learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return model


def rf_xgb_forecast(model, X_test):
    forecast = model.predict(X_test)
    conf_int = np.column_stack(
        (forecast - 1.96 * np.std(forecast), forecast + 1.96 * np.std(forecast)))
    return forecast, conf_int


def evaluate_model(y_true, y_pred):
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


def display_metrics(model_name, metrics, min_metrics):
    st.markdown(f"##### {model_name} Metrics")

    for metric_name, value in metrics.items():
        color = "#90EE90" if value == min_metrics[metric_name] else "#f0f0f0"
        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; width: 50%; margin: 10px 0; color: #000000;">
                <strong>{metric_name}:</strong> {value:.4f}
            </div>
            """,
            unsafe_allow_html=True
        )


def prophet_forecast(data, days=30):
    forecasts = {}
    for symbol in data['symbol'].unique():
        symbol_data = data[data['symbol'] == symbol].reset_index()
        symbol_data = symbol_data.rename(
            columns={'timestamp': 'ds', 'close': 'y'})
        model = Prophet()
        model.fit(symbol_data)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        forecasts[symbol] = forecast
    return forecasts
