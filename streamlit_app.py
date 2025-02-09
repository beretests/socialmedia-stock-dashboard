import streamlit as st
from analysis_utils import *
from forecasting_utils import *
from model_parameter_utils import *
import json
import time
import plotly.graph_objects as go
import pandas as pd
from ltsm_utils import *
from lstm2 import *
from datetime import datetime
from xgboost_model import *
from collections import deque


st.set_page_config(
    page_title="Stock Price Trends and Forecast Dashboard", layout="wide")

st.title("Stock Price Trends and Forecast Dashboard")


def read_stocks_from_file(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return ["META", "SNAP", "PINS", "MTCH", "SPOT", "RDDT", "DJT", "MOMO", "BILI", "NTES"]


def write_stocks_to_file(stocks, filename):
    with open(filename, 'w') as f:
        json.dump(stocks, f)


stock_symbols = read_stocks_from_file('stocks.json')
market_indices = read_stocks_from_file('index.json')
api_key = st.secrets["POLYGON_API_KEY"]


def update_stock_data():
    st.session_state.stock_data = fetch_stock_aggregate_bars(
        st.session_state.selected_stocks, api_key, st.session_state.from_date, st.session_state.to_date)


if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = stock_symbols[:3]

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "index_data" not in st.session_state:
    st.session_state.index_data = None

if "stock_details" not in st.session_state:
    st.session_state.stock_details = None

if "from_date" not in st.session_state:
    st.session_state.from_date = datetime.strptime('2023-01-01', "%Y-%m-%d")

today = datetime.today().date().strftime("%Y-%m-%d")
if "to_date" not in st.session_state:
    st.session_state.to_date = datetime.strptime(today, "%Y-%m-%d")

st.write("This is a streamlit-based stock market analysis dashboard that provides real-time data visualization, performance metrics, and price forecasting using advanced machine learning models like ARIMA, XGBoost, and LSTM.")
st.write("You can select multiple stocks, view historical price trends, compare performance metrics, and predict future stock prices.")
st.write("#### Select Stock and Date Ranges to View")


date_col, tick_col = st.columns(2, gap="large", border=True)
with date_col:
    st.write(
        "Select date range to view price trends, forecasts, and performance metrics.")
    from_date = st.date_input("From Date", value=st.session_state.from_date,
                              key="from_date", on_change=update_stock_data)
    to_date = st.date_input("To Date", value=st.session_state.to_date,
                            key="to_date", on_change=update_stock_data)

with tick_col:
    st.write("Select Stocks")
    message_container = st.empty()
    new_ticker = st.text_input("Add a new stock ticker:")

    if new_ticker:
        if new_ticker not in stock_symbols:
            stock_symbols.append(new_ticker)
            write_stocks_to_file(stock_symbols, 'social_media_stocks.json')
            message_container.success(
                f"Added {new_ticker} to the list of stocks.")

            time.sleep(5)
            message_container.empty()
        else:
            message_container.error(
                f"{new_ticker} is already in list of stocks. Select it below to view price trends and forecasts")

            time.sleep(5)
            message_container.empty()

    selected_stocks = st.multiselect(
        "Select stocks for comparison",
        stock_symbols,
        key='selected_stocks',
        on_change=update_stock_data
    )

st.write("#### Add Event Markers")
st.write("Add key events (e.g., earnings reports, product launches) to the charts to assess its possible impact.")
evnt_date, evnt_desc = st.columns([1, 3], gap="large")

with evnt_date:
    event_date = st.date_input("Event Date")

with evnt_desc:
    event_description = st.text_input("Event Description")
add_event = st.button("Add Event")
''
''
if st.button("Clear Cache"):
    st.cache_data.clear()

if 'events' not in st.session_state:
    st.session_state.events = []

if add_event:
    st.session_state.events.append((event_date, event_description))

tab_titles = ['Stock Description', 'Historical Data', 'Performance Metrics', 'Model Parameter Determination',
              'Stock Price Prediction', 'Trade Volume Prediction', 'Comparative Analysis', 'Price Forecasting']


tabs = st.tabs(tab_titles)

######## -------STOCK DATA DESCRIPTION----------########

with tabs[0]:
    st.subheader("Stock data description")

    if st.session_state.stock_details is None:
        st.session_state.stock_details = fetch_stock_details(
            stock_symbols, api_key)
    df_stock_details = []
    for symbol, details in st.session_state.stock_details.items():
        df_stock_details.append(
            {
                "ticker": symbol,
                'Name': details.get('results', {}).get('name', 'N/A'),
                'Description': details.get('results', {}).get('description', 'N/A'),
                'SIC Description': details.get('results', {}).get('sic_description', 'N/A'),
                'Homepage Url': details.get('results', {}).get('homepage_url', 'N/A'),
                'List Date': details.get('results', {}).get('list_date', 'N/A'),
            }
        )
    stock_details_df = pd.DataFrame(df_stock_details)
    stock_details_df.set_index('ticker', inplace=True)
    st.table(stock_details_df)

######## -------HISTORICAL DATA----------########

with tabs[1]:
    if st.session_state.stock_data is None:
        st.session_state.stock_data = fetch_stock_aggregate_bars(
            selected_stocks, api_key, from_date, to_date)

    for symbol, data in st.session_state.stock_data.items():
        if isinstance(data, dict) and 'results' in data:
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            df['daily_return'] = df['c'].pct_change()
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

            df.dropna(inplace=True)

            st.session_state.stock_data[symbol] = df

    st.write("""
    ### Stock Price Gap Analysis

    This section displays the price gap information for the selected stocks, helping you understand how each stock opened compared to its previous closing price.

    - **Bullish Gap**: The stock opened higher than its previous closing price, indicating potential upward momentum.
    - **Bearish Gap**: The stock opened lower than its previous closing price, signaling possible downward movement.
    - **No Gap**: The stock opened at the same price as its previous close, showing no significant price difference.
    """)

    calculate_price_gaps(st.session_state.stock_data)

    st.divider()

    st.write("""
    ### Stock Price Comparison
    This **line chart** compares the closing prices of multiple stocks over time.
    **Upward trends** suggest growth or bullish sentiment, while **downward trends** may indicate declines or bearish sentiment.
    **Gold vertical dashed lines** highlight key events (e.g., earnings reports, news releases) that may have impacted stock prices.
    Use this chart to **spot trends, compare performance, and identify market-moving events**.
    """)

    st.plotly_chart(create_comparative_line_plot(
        st.session_state.stock_data, st.session_state.events), use_container_width=True)
    st.divider()

    st.write("""
    ### Stock Trading Volume Analysis
    This **stacked bar chart** compares the trading volume of different stocks over time.
    **Higher volume** can indicate strong investor interest, either due to earnings reports, news, or market trends.
    **Vertical dashed lines** mark key events (e.g., major market news).
    """)

    st.plotly_chart(create_volume_comparison(st.session_state.stock_data,
                    st.session_state.events), use_container_width=True)
    st.divider()

    st.write("""
    ### Stock Candlestick Charts
    These help visualize stock price movements over time.
    Each candlestick represents **Open, High, Low, and Close (OHLC)** prices for a given period.
    **Red candles** indicate a decrease in price, while **green candles** show an increase.
    **Long wicks** suggest volatility, while small wicks indicate stable movements.
    """)
    st.plotly_chart(create_candlestick_chart(
        st.session_state.stock_data), use_container_width=True)

######## -------PERFORMANCE METRICS----------########

with tabs[2]:
    st.subheader("Stock Performance Metrics")
    st.write("""
    ### Stock Daily Return Analysis
    **Daily return** represents the percentage change in a stock's price from one day to the next.
    Positive values indicate **gains**, while negative values indicate **losses**.
    **High volatility** (frequent ups and downs) suggests uncertainty or market reactions to news.
    **Stable returns** suggest consistent performance with lower risk.
    **Red vertical dashed lines** highlight key events (e.g., earnings reports, major market news) that may have impacted returns.
    Use this chart to **identify trends, assess volatility, and understand event-driven price movements**.
    """)
    st.plotly_chart(create_daily_returns_comparison(
        st.session_state.stock_data, st.session_state.events), use_container_width=True)

    st.write("""
    ### Cumulative Return Analysis
    **Cumulative return** measures the total percentage change in a stock's price over time.
    Unlike daily returns, which show **short-term fluctuations**, cumulative returns reveal **long-term trends**.
    **Upward trends** indicate sustained growth, while **downward trends** suggest prolonged declines.
    **Comparison across stocks** helps identify the best-performing investments over a given period.
    **Red vertical dashed lines** highlight significant events (e.g., earnings reports, policy changes) that might have impacted returns.
    Use this chart to assess **long-term performance, trends, and event-driven market movements**.
    """)
    st.plotly_chart(create_cumulative_returns_comparison(
        st.session_state.stock_data, st.session_state.events), use_container_width=True)

    st.write("""
    ### Daily Returns Distribution Charts

    This set of charts displays the distribution of daily returns for each stock in your portfolio. This visualization helps in understanding the risk and return characteristics of each stock in your portfolio, aiding in investment decisions and risk management.

    ###### Key Insights:

    - **Central Tendency**: The position of the gold line (mean) shows whether the stock tends to have positive or negative daily returns.
    - **Volatility**: The spread of the histogram and the distance between the green lines indicate the stock's volatility. Wider spreads suggest higher risk and potential for both gains and losses.
    - **Skewness**: If the histogram is not symmetrical around the mean, it indicates skewness in returns. Right-skewed distributions have more extreme positive returns, while left-skewed have more extreme negative returns.
    - **Outliers**: Pay attention to any bars far from the mean, especially beyond the standard deviation lines. These represent days with unusually high or low returns.
    """)
    st.plotly_chart(create_daily_returns_distribution(
        st.session_state.stock_data), use_container_width=True)

######## -------ARIMA MODEL PARAMETERS----------########

with tabs[3]:
    st.subheader("Determine ARIMA model parameters")
    st.write("Select a stock symbol to check the stationarity of the selected stock's data using the ARIMA model parameters. This is relevant for identifying the appropriate ARIMA model configuration, which is crucial for accurate stock price forecasting and analysis.")

    symbol = st.selectbox("Select a stock symbol", list(
        st.session_state.stock_data.keys()), key='model_parameter_symbol')
    if symbol in st.session_state.stock_data:
        data_for_modelling = st.session_state.stock_data[symbol]

        if isinstance(data_for_modelling, pd.DataFrame):
            if st.button(f"Check ARIMA Model Parameters for {symbol}"):
                st.write("This section visualizes the differenced time series and calculates the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) values for the selected stock price. It plots the differenced series to check for stationarity and creates bar plots for ACF and PACF values to identify significant lags. This is relevant for determining the appropriate parameters for ARIMA models and diagnosing the residuals to ensure they are white noise, which is crucial for accurate time series forecasting.")
                check_stationary(data_for_modelling)

######## -------STOCK PRICE PREDICTION----------########

with tabs[4]:
    st.subheader("Stock Price Prediction Model Comparison")
    st.write("In this section, you can compare stock price prediction models. select a stock symbol and configure model parameters for ARIMA, Random Forest, and XGBoost models. The models are then trained using the selected parameters and splits the data into training and testing sets before predicting stock prices. The model evaluation metrics are displayed to help you assess the performance of each model.")

    symbol = st.selectbox("Select a stock symbol", list(
        st.session_state.stock_data.keys()), key='prediction_symbol')
    data = st.session_state.stock_data[symbol]

    X, y, prepped_data = prepare_stock_data(data)
    X_train_scaled, X_test_scaled, y_train, y_test, X_test, X_train, train, test = split_data(
        X, y, prepped_data, test_size=0.2)

    st.write('#### Model Parameters')
    st.write("Select the parameters for each model to train and evaluate the performance of the ARIMA, Random Forest, and XGBoost models.")
    col_para1, col_para2 = st.columns(2, border=True)
    with col_para1:
        colp, cold, colq = st.columns(3)
        with colp:
            p = st.slider('ARIMA p', 0, 5, 1, key='arima_p')
        with cold:
            d = st.slider('ARIMA d', 0, 5, 1, key='arima_d')
        with colq:
            q = st.slider('ARIMA q', 0, 5, 1, key='arima_q')
        arima_order = (p, d, q)
    with col_para2:
        col_est, col_learn = st.columns(2)
        with col_est:
            n_estimators = st.slider(
                'Number of estimators for Random Forest/XGBoost:', 10, 200, 100)
        with col_learn:
            learning_rate = st.slider(
                'Learning rate for XGBoost:', 0.01, 0.3, 0.1)

    arima_model = train_arima_model(y_train, order=arima_order)
    rf_model = train_random_forest(X_train_scaled, y_train, n_estimators)
    xgb_model = train_xgboost(X_train_scaled, y_train,
                              n_estimators, learning_rate)

    arima_forecast = arima_model.forecast(steps=len(y_test))
    rf_predictions = rf_model.predict(X_test_scaled)
    xgb_predictions = xgb_model.predict(X_test_scaled)

    arima_metrics = evaluate_model(
        y_test[:len(arima_forecast)], arima_forecast)
    xgb_metrics = evaluate_model(y_test, xgb_predictions)
    rf_metrics = evaluate_model(y_test, rf_predictions)

    min_metrics = {
        'MSE': min(arima_metrics['MSE'], xgb_metrics['MSE'], rf_metrics['MSE']),
        'MAE': min(arima_metrics['MAE'], xgb_metrics['MAE'], rf_metrics['MAE']),
        'RMSE': min(arima_metrics['RMSE'], xgb_metrics['RMSE'], rf_metrics['RMSE']),
        'R² Score': min(arima_metrics['R² Score'], xgb_metrics['R² Score'], rf_metrics['R² Score'])
    }

    st.divider()

    st.subheader("Model Evaluation Metrics")
    st.write("The metrics below are used to evaluate the performance of each model in predicting stock prices. Lower values for MSE, MAE, and RMSE indicate better model performance, while higher values for R² Score suggest a better fit between the predicted and actual values. The preferred metric values are highlighted in green.")
    col1, col2, col3 = st.columns(3)

    with col1:
        display_metrics("ARIMA Model", arima_metrics, min_metrics)

    with col2:
        display_metrics("XGBoost Model", xgb_metrics, min_metrics)

    with col3:
        display_metrics("Random Forest Model", rf_metrics, min_metrics)

    st.divider()

    st.header('Model Forecast vs Actual Values')
    st.write("The following charts compare the actual and forecasted stock prices for each model. Use these interactive charts to visualize the model performance and identify any discrepancies between the predicted and actual values.")

    arima_fig = go.Figure()
    arima_fig.add_trace(go.Scatter(
        x=train.index, y=train["c"], mode='lines', name='Train', line=dict(color='#00008B')))
    arima_fig.add_trace(go.Scatter(
        x=test.index, y=test['c'], mode='lines', name='Test', line=dict(color='#01ef63')))
    arima_fig.add_trace(go.Scatter(x=test.index, y=arima_forecast,
                        mode='lines', name='ARIMA Forecast', line=dict(color='orange')))

    arima_fig.update_layout(
        title='ARIMA - Actual vs Forecasted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Close Price',
        width=1000,
        height=500
    )
    st.plotly_chart(arima_fig, use_container_width=True)

    rf_fig = go.Figure()
    rf_fig.add_trace(go.Scatter(
        x=train.index, y=train["c"], mode='lines', name='Actual'))
    rf_fig.add_trace(go.Scatter(
        x=test.index, y=test['c'], mode='lines', name='Test', line=dict(color='#01ef63')))
    rf_fig.add_trace(go.Scatter(x=test.index, y=rf_predictions,
                     mode='lines', name='Random Forest Forecast'))
    rf_fig.update_layout(title='Random Forest - Actual vs Forecasted Stock Prices',
                         xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(rf_fig)

    xgb_fig = go.Figure()
    xgb_fig.add_trace(go.Scatter(
        x=train.index, y=train["c"], mode='lines', name='Actual'))
    xgb_fig.add_trace(go.Scatter(
        x=test.index, y=test['c'], mode='lines', name='Test', line=dict(color='#01ef63')))
    xgb_fig.add_trace(go.Scatter(x=test.index, y=xgb_predictions,
                      mode='lines', name='XGBoost Forecast'))
    xgb_fig.update_layout(title='XGBoost - Actual vs Forecasted Stock Prices',
                          xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(xgb_fig)

######## -------TRADE VOLUME PREDICTION----------########

with tabs[5]:
    st.subheader("Trade Volume Prediction")
    st.write("In this section, you can predict the trading volume of a selected stock using an LSTM model. Select a stock symbol, configure the model parameters, and train the model to predict the trading volume. The model's performance metrics are displayed to help you assess the accuracy of the volume predictions.")

    symbol = st.selectbox("Select a stock symbol", list(
        st.session_state.stock_data.keys()), key='volume_prediction_symbol')

    col_para3, col_para4 = st.columns(2, border=True)
    with col_para3:
        st.write('#### Model Parameters')
        lookback = st.slider("Lookback Period", 30, 90, 60)
        epochs = st.slider("Training Epochs", 10, 100, 50)
        test_size = st.slider("Test Size (%)", 10, 40, 20)

    with col_para4:
        st.write("#### Features Used for Modeling")

    if symbol in st.session_state.stock_data:
        stock_data = st.session_state.stock_data[symbol]

        if isinstance(stock_data, pd.DataFrame) and 'daily_return' in stock_data:
            if st.button("Predict Trading Volume"):
                X_l, y_l, scaler_l, features = preprocess_data(
                    stock_data, lookback)

                split = int(len(X_l) * (1 - test_size/100))
                X_train_l, X_test_l = X_l[:split], X_l[split:]
                y_train_l, y_test_l = y_l[:split], y_l[split:]
                print("X_train_l.shape:", X_train_l.shape)

                X_train_l = np.reshape(
                    X_train_l, (X_train_l.shape[0], X_train_l.shape[1], len(features)))
                X_test_l = np.reshape(
                    X_test_l, (X_test_l.shape[0], X_test_l.shape[1], len(features)))

                model = build_model((X_train_l.shape[1], X_train_l.shape[2]))
                history = model.fit(X_train_l, y_train_l,
                                    validation_data=(X_test_l, y_test_l),
                                    epochs=epochs, batch_size=32, verbose=0)

                train_predict = model.predict(X_train_l).flatten()
                test_predict = model.predict(X_test_l).flatten()

                st.write("#### Model Evaluation Metrics")
                st.write("The metrics below are used to evaluate the performance of the LSTM model in predicting trading volumes. Lower values for MSE, MAE, and RMSE indicate better model performance, while higher values for R² Score suggest a better fit between the predicted and actual values.")
                show_metrics(y_test_l, test_predict)

                dummy_array = np.zeros((len(y_l), len(features)+1))
                dummy_array[:, -1] = y_l
                y_actual = scaler_l.inverse_transform(dummy_array)[:, -1]

                dummy_array[:, -
                            1] = np.concatenate([train_predict, test_predict])
                y_pred = scaler_l.inverse_transform(dummy_array)[:, -1]

                dates = stock_data.index[lookback:]

                st.write("#### Volume Prediction")
                st.write("The following chart compares the actual and predicted trading volume of the selected stock. Use this chart to visualize the model's performance and identify any discrepancies between the predicted and actual values.")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=y_actual, name='Actual Volume',
                                         line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=dates, y=y_pred,
                              name='Predict Volume', line=dict(color='green')))

                fig.update_layout(title='Volume Prediction Results',
                                  xaxis_title='Date',
                                  yaxis_title='Trading Volume',
                                  height=600,
                                  showlegend=True)

                st.plotly_chart(fig, use_container_width=True)

                st.write("#### Model Training Progress")
                st.write("The following chart shows the training and validation loss of the LSTM model over the training epochs. Decreasing loss values indicate better model performance. If the training loss continues to decrease while the validation loss starts to increase, it suggests that the model is overfitting the training data.")

                loss_fig = go.Figure()
                loss_fig.add_trace(go.Scatter(
                    y=history.history['loss'], name='Training Loss'))
                loss_fig.add_trace(go.Scatter(
                    y=history.history['val_loss'], name='Validation Loss'))
                loss_fig.update_layout(title='Model Training Progress',
                                       xaxis_title='Epochs',
                                       yaxis_title='Loss',
                                       height=400)

                st.plotly_chart(loss_fig, use_container_width=True)

                with col_para4:
                    st.write(features)

######## -------COMPARATIVE ANALYSIS----------########

with tabs[6]:
    st.subheader("Comparative Analysis")
    st.write("In this section, you can compare the performance of a selected stock with various market indices. Select a stock symbol and one or more market indices to compare the stock's price trends and cumulative returns with the selected indices. The correlation heatmap provides insights into the relationships between the stock and other selected stock.")

    index_symbol = st.selectbox("Select a stock symbol", list(
        st.session_state.stock_data.keys()), key='index_comparison_symbol')
    ''
    ''
    st.write("### Correlation Heatmap")
    st.plotly_chart(create_correlation_heatmap(
        st.session_state.stock_data), use_container_width=True)

    st.divider()

    idx_sym1, idx_sym2 = st.columns(2, gap="large")

    with idx_sym1:
        new_index_ticker = st.text_input(
            "Add an index ticker:", key='index_ticker')

        if new_index_ticker:
            if new_index_ticker not in market_indices:
                market_indices.append(new_index_ticker)
                write_stocks_to_file(market_indices, 'index.json')
                message_container.success(
                    f"Added {new_index_ticker} to the list of market indices.")

                time.sleep(5)
                message_container.empty()
            else:
                message_container.error(
                    f"{new_index_ticker} is already in list of stocks. Select it below to view price trends and forecasts")

                time.sleep(5)
                message_container.empty()

    with idx_sym2:
        selected_indices = st.multiselect(
            "Select market index to compare to stock", market_indices, default=market_indices[:1])

    if st.button("Compare stock to index"):
        st.session_state.index_data = fetch_index_aggregate_bars(
            selected_indices, api_key, from_date, to_date)

        if st.session_state.index_data:
            for index, data in st.session_state.index_data.items():
                if isinstance(data, dict) and 'results' in data:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df.set_index('date', inplace=True)
                    df['cumulative_return'] = (
                        1 + df['c'].pct_change()).cumprod() - 1
                    st.session_state.index_data[index] = df

            if index_symbol in st.session_state.stock_data:
                stock_data = st.session_state.stock_data[index_symbol]

                if isinstance(stock_data, pd.DataFrame) and 'cumulative_return' in stock_data:
                    fig = create_index_stock_comparison(
                        st.session_state.index_data, stock_data, index_symbol)
                    st.plotly_chart(fig)
                else:
                    st.error(f"Unexpected data format for {index_symbol}")
            else:
                st.error(f"Stock data for {index_symbol} not found.")


######## -------PRICE FORECASTING----------########

with tabs[7]:
    st.subheader("Stock Price Forecasting")
    st.write("In this section, you can predict future stock prices using XGBoost and LSTM models. Select a stock symbol, configure the model parameters, and predict the stock prices for the next few days. The model evaluation metrics are displayed to help you assess the accuracy of the price forecasts.")

    forecast_symbol = st.selectbox("Select a stock symbol", list(
        st.session_state.stock_data.keys()), key='price_forecasting_symbol')

    st.write('#### Model Parameters')
    col_model1, col_model2, col_model3 = st.columns(3, gap="large")

    with col_model1:
        xgb_lookback = st.slider(
            "Lookback Period", 30, 90, 60, 10, key='xgb_lookback')

    with col_model2:
        xgb_test_size = st.slider(
            "Test Size (%)", 10, 40, 20, 5, key='xgb_test_size')

    with col_model3:
        future_n_days = st.slider(
            "Days in Future", 10, 30, 10, 5, key='xgb_future_days')

    with col_model1:
        lstm_epochs = st.slider("Training Epochs for LSTM", 1,
                                100, 1, 5, key='lstm_epochs')

    if forecast_symbol in st.session_state.stock_data:
        xgb_stock_data = st.session_state.stock_data[forecast_symbol]

        if isinstance(xgb_stock_data, pd.DataFrame):
            st.write('#### Predict Future Prices using XGBoost')
            if st.button(f"Predict {forecast_symbol} Prices for Next {future_n_days} Days using XGBoost"):
                try:
                    xgb_processed_data = prepare_xgboost_data(
                        xgb_stock_data)

                    if len(xgb_processed_data) < xgb_lookback:
                        raise ValueError(
                            f"Insufficient data: Expected at least {xgb_lookback} rows, but got {len(xgb_processed_data)}")

                    X_train, X_valid, y_train, y_valid = split_xgboost_data(
                        xgb_processed_data, test_size=test_size, lookback=xgb_lookback)

                    last_n_days = deque(
                        xgb_processed_data[-xgb_lookback:], maxlen=xgb_lookback)

                    xgb_model, xgb_execution_time = train_xgboost_model(
                        X_train, y_train)
                    st.write(
                        f"Execution time: {xgb_execution_time:.2f} seconds")

                    future_prices = xgb_predict_future_prices(
                        xgb_model, np.array(last_n_days), future_n_days, xgb_lookback)

                    y_valid_pred = xgb_model.predict(X_valid)

                    xgb_metrics = evaluate_model_metrics(y_valid, y_valid_pred)
                    display_model_metrics("XGBoost Model", xgb_metrics)

                    plot_actual_forecast(xgb_stock_data,
                                         future_prices, future_n_days)
                except ValueError as e:
                    st.error(f"Error in preprocessing data: {e}")
                except IndexError as e:
                    st.error(f"Error in reshaping data: {e}")
                except TypeError as e:
                    st.error(f"Error in predicting future prices: {e}")

            st.write('#### Predict Future Prices using LSTM')
            if st.button(f"Predict {forecast_symbol} Prices for Next {future_n_days} Days using LSTM"):
                try:
                    lstm_processed_data, training_data_len, y_test, lstm_scaler, lstm_last_date, lstm_train, lstm_valid = prepare_lstm_data(
                        xgb_stock_data)

                    if len(lstm_processed_data) < xgb_lookback:
                        raise ValueError(
                            f"Insufficient data: Expected at least {xgb_lookback} rows, but got {len(lstm_processed_data)}")

                    x_train, x_test, y_train = create_lstm_dataset(
                        lstm_processed_data, training_data_len, xgb_lookback)

                    lstm_last_n_days = lstm_processed_data[-xgb_lookback:]

                    lstm_model, lstm_exec_time = build_lstm_model(
                        x_train, y_train, lstm_epochs)

                    st.write(f"Execution time: {lstm_exec_time:.2f} seconds")

                    lstm_predictions, lstm_future_predictions = lstm_predict_prices(
                        lstm_model, x_test, lstm_scaler, lstm_last_n_days, future_n_days, xgb_lookback, lstm_last_date)

                    lstm_valid['Predictions'] = lstm_predictions

                    lstm_metrics = evaluate_model_metrics(
                        y_test, lstm_valid['Predictions'])
                    display_model_metrics("LSTM Model", lstm_metrics)

                    plot_lstm_predictions(
                        lstm_train, lstm_valid, lstm_future_predictions)

                except ValueError as e:
                    st.error(f"Error in preprocessing data: {e}")
                except IndexError as e:
                    st.error(f"Error in reshaping data: {e}")
                except TypeError as e:
                    st.error(f"Error in predicting future prices: {e}")
