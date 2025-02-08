from statsmodels.tsa.stattools import adfuller, acf, pacf
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def check_stationary(data, target_col='c'):
    """
    Check if the time series is stationary and perform differencing if necessary.

    Parameters
    ----------
    data : dict
        The input data dictionary containing stock information.
    ticker : str
        The stock ticker symbol.
    target_col : str
        The target column to check for stationarity.

    Returns
    -------
    None
    """
    st.subheader(f"ARIMA Model Parameters")
    st.write("**Check for stationarity and perform differencing**")

    df = data.copy()
    result_original = adfuller(df[target_col])

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write(f"**ADF Statistic (Original)**: {result_original[0]:.4f}")
        st.write(f"**p-value (Original)**: {result_original[1]:.4f}")
        st.write("**Critical Values (Original)**:")
        for key, value in result_original[4].items():
            st.write(f"\t{key}: {value:.4f}")

    if result_original[1] > 0.05:
        df[f'{target_col}_Diff'] = df[target_col].diff().dropna()
        result_diff = adfuller(df[f'{target_col}_Diff'].dropna())

        with col2:
            st.write(f"**ADF Statistic (Differenced)**: {result_diff[0]:.4f}")
            st.write(f"**p-value (Differenced)**: {result_diff[1]:.4f}")
            st.write("\t**Critical Values (Differenced)**:")
            for key, value in result_diff[4].items():
                st.write(f"\t{key}: {value:.4f}")
            if result_diff[1] < 0.05:
                st.success(
                    "Interpretation: The differenced series is Stationary.")
            else:
                st.warning(
                    "Interpretation: The differenced series is Non-Stationary.")

    diff_fig = go.Figure()
    diff_fig.add_trace(go.Scatter(
        x=df.index, y=df[f'{target_col}_Diff'], mode='lines', name=f'{target_col} Differenced', line=dict(color='orange')))

    diff_fig.update_layout(
        title=f'{target_col} Differenced Time Series',
        xaxis_title='Date',
        yaxis_title=f'{target_col} Differenced',
        height=500
    )

    st.plotly_chart(diff_fig, use_container_width=True)

    lags = min(40, len(df[f'{target_col}_Diff'].dropna()) // 2 - 1)
    acf_values = acf(df[f'{target_col}_Diff'].dropna(), nlags=lags)
    pacf_values = pacf(df[f'{target_col}_Diff'].dropna(), nlags=lags)

    acf_pacf_fig = go.Figure()

    acf_pacf_fig.add_trace(go.Bar(
        x=np.arange(len(acf_values)),
        y=acf_values,
        name='ACF',
        yaxis='y',
        marker_color='blue'
    ))

    acf_pacf_fig.add_trace(go.Bar(
        x=np.arange(len(pacf_values)),
        y=pacf_values,
        name='PACF',
        yaxis='y2',
        marker_color='red'
    ))

    acf_pacf_fig.add_hline(y=1.96/np.sqrt(len(df)),
                           line_dash="dot", line_color="gray", opacity=0.7)
    acf_pacf_fig.add_hline(y=-1.96/np.sqrt(len(df)),
                           line_dash="dot", line_color="gray", opacity=0.7)

    acf_pacf_fig.update_layout(
        title='ACF and PACF Plots',
        xaxis_title='Lag',
        yaxis_title='Correlation',
        yaxis2=dict(title='Correlation', overlaying='y', side='right'),
        width=1000,
        height=500
    )

    st.plotly_chart(acf_pacf_fig, use_container_width=True)
