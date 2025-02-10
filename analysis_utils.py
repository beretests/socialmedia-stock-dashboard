import math
import pandas as pd
import plotly.graph_objects as go
from polygon import (StocksClient, IndexClient, ReferenceClient)
import streamlit as st
from plotly.subplots import make_subplots
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor

# Set up rate limiting
CALLS_PER_MINUTE = 5
PERIOD = 60


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
def rate_limited_agg_call(stocks_client, symbol, from_date, to_date):
    return stocks_client.get_aggregate_bars(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        timespan='day'
    )


@st.cache_data(ttl=3600)
def fetch_stock_aggregate_bars(symbols, api_key, from_date, to_date):
    stocks_client = StocksClient(api_key)
    results = {}

    def fetch_symbol(symbol):
        try:
            aggs = rate_limited_agg_call(
                stocks_client, symbol, from_date, to_date)
            return symbol, aggs

        except Exception as e:
            st.error(f"API Error for {symbol}: {e}")
            return symbol, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=5) as executor:
        for symbol, data in executor.map(fetch_symbol, symbols):
            results[symbol] = data

    return results


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
def rate_limited_index_call(index_client, symbol, from_date, to_date):
    return index_client.get_aggregate_bars(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        timespan='day'
    )


@st.cache_data(ttl=3600)
def fetch_index_aggregate_bars(indices, api_key, from_date, to_date):
    index_client = IndexClient(api_key)
    results = {}

    def fetch_index(index):
        try:
            index_aggs = rate_limited_index_call(
                index_client, index, from_date, to_date)
            return index, index_aggs

        except Exception as e:
            st.error(f"API Error for {index}: {e}")
            return index, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, data in executor.map(fetch_index, indices):
            results[index] = data

    return results


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
def rate_limited_api_call(reference_client, symbol):
    return reference_client.get_ticker_details(symbol=symbol)


@st.cache_data(ttl=3600)
def fetch_stock_details(symbols, api_key):
    reference_client = ReferenceClient(api_key)
    results = {}

    def fetch_symbol(symbol):
        try:
            details = rate_limited_api_call(reference_client, symbol)
            return symbol, details
        except Exception as e:
            st.error(f"API Error for {symbol}: {e}")
            return symbol, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=5) as executor:
        for symbol, details in executor.map(fetch_symbol, symbols):
            results[symbol] = details

    return results


def create_comparative_line_plot(data_dict, events):
    fig = go.Figure()

    for symbol, data in data_dict.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['c'], name=symbol))

    fig.update_layout(
        title="Comparative Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Symbols"
    )

    if events:
        for date, description in events:
            fig.add_vline(x=date, line_dash="dash", line_color="gold")
            fig.add_annotation(x=date, y=fig.layout.yaxis.range[1], text=description,
                               showarrow=True, arrowhead=1)

    return fig


def create_candlestick_chart(data_dict):
    num_stocks = len(data_dict)
    cols = math.ceil(math.sqrt(num_stocks))
    rows = math.ceil(num_stocks / cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(data_dict.keys()),
                        vertical_spacing=0.1, horizontal_spacing=0.05)

    for idx, (symbol, data) in enumerate(data_dict.items(), start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1

        candlestick = go.Candlestick(
            x=data.index,
            open=data['o'],
            high=data['h'],
            low=data['l'],
            close=data['c'],
            name=symbol
        )
        fig.add_trace(candlestick, row=row, col=col)

        fig.update_xaxes(rangeslider_visible=False, row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    fig.update_layout(height=450*rows, width=1000, showlegend=False)
    return fig


def create_volume_comparison(data_dict, events):
    fig = go.Figure()

    for symbol, data in data_dict.items():
        fig.add_trace(go.Bar(x=data.index, y=data['v'], name=symbol))

    fig.update_layout(
        title="Trading Volume Comparison",
        xaxis_title="Date",
        yaxis_title="Volume",
        barmode='group'
    )

    for date, description in events:
        fig.add_vline(x=date, line_dash="dash", line_color="red")
        fig.add_annotation(x=date, y=1, text=description,
                           showarrow=True, arrowhead=1)

    return fig


def create_daily_returns_comparison(data_dict, events):
    fig = go.Figure()

    for symbol, data in data_dict.items():
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['daily_return'],
            mode='lines',
            name=symbol
        ))

    fig.update_layout(
        title="Daily Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        legend_title="Symbols"
    )

    # Add event lines and annotations
    for date, description in events:
        fig.add_vline(x=date, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=date,
            y=fig.layout.yaxis.range[1],
            text=description,
            showarrow=True,
            arrowhead=1,
            yshift=10
        )

    return fig


def create_cumulative_returns_comparison(data_dict, events):
    fig = go.Figure()

    for symbol, data in data_dict.items():
        if "cumulative_return" not in data:
            raise KeyError(
                f"Expected key 'cumulative_return' not found in stock_data for {symbol}")

        fig.add_trace(go.Scatter(
            x=data.index, y=data['cumulative_return'], name=symbol))

    fig.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend_title="Symbols"
    )

    for date, description in events:
        fig.add_vline(x=date, line_dash="dash", line_color="red")
        fig.add_annotation(x=date, y=1, text=description,
                           showarrow=True, arrowhead=1)

    return fig


def create_index_stock_comparison(index_data_dict, stock_data, stock_name):
    fig = go.Figure()

    # Ensure stock_data contains the expected keys
    if "cumulative_return" not in stock_data:
        raise KeyError(
            f"Expected key'cumulative_return' not found in stock_data for {stock_name}")

    # Ensure index_data contains the expected keys
    for symbol, data in index_data_dict.items():
        if isinstance(data, pd.DataFrame) and 'cumulative_return' in data:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['cumulative_return'],
                mode='lines',
                name=symbol,
                line=dict(color='#ff7f0e')
            ))

    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['cumulative_return'],
        mode='lines',
        name=stock_name,
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title=f'Performance Comparison of {stock_name} vs Different Market Indices',
        xaxis_title="Date",
        yaxis_title='Cumulative Returns',
        hovermode='x unified',
        legend_title="Symbols"
    )

    return fig


def create_correlation_heatmap(data_dict):
    dfs = {k: v for k, v in data_dict.items()}

    combined_df = pd.concat({k: df['c'] for k, df in dfs.items()}, axis=1)
    combined_df.index = pd.to_datetime(combined_df.index)
    cumulative_returns = (1 + combined_df.pct_change()).cumprod() - 1

    corr_matrix = cumulative_returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.index,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Correlation Heatmap of Cumulative Returns',
        xaxis_title='Stocks',
        yaxis_title='Stocks'
    )

    return fig


def create_daily_returns_distribution(data_dict):
    n = len(data_dict)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Create subplots
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=list(data_dict.keys()))

    for i, (symbol, data) in enumerate(data_dict.items(), start=1):

        daily_returns = data['daily_return'].dropna()
        mean = daily_returns.mean()
        std = daily_returns.std()

        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1

        fig.add_trace(go.Histogram(x=daily_returns, name=symbol,
                      opacity=0.7), row=row, col=col)

        # Add mean line
        fig.add_vline(x=mean, line_dash="dash", line_color="gold",
                      annotation_text=f"Mean: {mean:.2%}",
                      annotation_position="top right", row=row, col=col)

        fig.add_vline(x=mean+std, line_dash="dash", line_color="green",
                      annotation_text=f"+1 Std Dev: {(mean+std):.2%}",
                      annotation_position="top right", row=row, col=col)
        fig.add_vline(x=mean-std, line_dash="dash", line_color="green",
                      annotation_text=f"-1 Std Dev: {(mean-std):.2%}",
                      annotation_position="bottom right", row=row, col=col)

        fig.update_xaxes(title_text="Daily Return", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Update overall layout
    fig.update_layout(
        title="Daily Returns Distribution",
        showlegend=False,
        height=300 * rows,
        width=400 * cols
    )

    return fig


def calculate_price_gaps(data_dict):
    processed_data = {}
    for symbol, data in data_dict.items():
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        if 'o' not in data.columns or 'c' not in data.columns:
            raise KeyError(
                f"Expected columns 'o' and 'c' not found in data for {symbol}")

        data['gap'] = data['o'] - data['c'].shift(1)
        data['gap_type'] = data['gap'].apply(
            lambda x: 'Bullish' if x > 0 else 'Bearish' if x < 0 else 'No Gap')
        processed_data[symbol] = data

    num_stocks = len(processed_data)
    num_cols = 5
    num_rows = math.ceil(num_stocks / num_cols)

    grid_container = st.container()

    for row in range(num_rows):
        with grid_container:
            cols = st.columns(num_cols)
            for col in range(num_cols):
                index = row * num_cols + col
                if index < num_stocks:
                    symbol, data = list(processed_data.items())[index]
                    latest_data = data.iloc[-1]
                    gap = latest_data['gap']
                    gap_type = latest_data['gap_type']

                    if gap_type == 'Bullish' or gap_type == 'Bearish':
                        color = "normal"
                    else:
                        color = "off"

                    cols[col].metric(
                        label=f"{symbol} Gap",
                        value=f"{gap_type}",
                        delta=f"{gap:.2f}",
                        delta_color=color,
                        border=True
                    )
