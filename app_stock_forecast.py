# app_stock_forecast.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Stock Analyzer with ML Forecast")
st.markdown("Enter a stock symbol and date range to view trends and forecast the next 7 days.")

# --- Stock Symbol Input ---
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA)", "AAPL").upper()

# --- API Key Input with Default ---
default_key = "GKI9ZPPWV5BUJ5VH"
api_key = st.text_input("🔐 Enter your Alpha Vantage API Key", value=default_key, type="password")

# --- Date Inputs ---
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", pd.to_datetime("2024-01-01"))
with col2:
    end_date = st.date_input("📅 End Date", pd.to_datetime("2025-06-30"))

# --- Fetch Button ---
if st.button("📥 Fetch & Forecast"):
    try:
        # Step 1: Load Data
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data.index = pd.to_datetime(data.index)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        filtered = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
        filtered = filtered.sort_index(ascending=True)

        if filtered.empty:
            st.error("❌ No data available for selected range.")
            st.stop()

        st.success("✅ Data loaded successfully!")
        st.write("### 🧾 Raw Data", filtered.tail())

        # Step 2: Technical Indicators
        filtered['MA_20'] = filtered['Close'].rolling(window=20).mean()

        # ✅ Correct RSI Calculation
        delta = filtered['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        filtered['RSI'] = 100 - (100 / (1 + rs))

        # Step 3: ML Forecasting
        forecast_days = 7
        df_ml = filtered[['Close']].copy()
        df_ml['Target'] = df_ml['Close'].shift(-forecast_days)
        df_ml = df_ml[:-forecast_days]

        X = df_ml[['Close']].to_numpy()
        y = df_ml['Target'].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.info(f"📉 RMSE on test set: {rmse:.2f}")

        # Step 4: Forecast Next 7 Days
        last_7_input = filtered['Close'][-forecast_days:].to_numpy().reshape(-1, 1)
        future_predictions = model.predict(last_7_input)

        last_date = filtered.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
        forecast_df.set_index('Date', inplace=True)

        st.write("### 📆 Forecast for Next 7 Days")
        st.dataframe(forecast_df.style.format({"Predicted_Close": "{:.2f}"}))

        # Step 5: Matplotlib Forecast Plot
        st.write("### 📉 ML Forecast Plot")
        fig, ax = plt.subplots(figsize=(12, 6))

        # ✅ Convert to NumPy for indexing
        actual_dates = filtered.index.to_numpy()
        actual_close = filtered['Close'].to_numpy()
        future_x = forecast_df.index.to_numpy()
        future_y = forecast_df['Predicted_Close'].to_numpy()

        ax.plot(actual_dates, actual_close, label='Actual Close', alpha=0.6)
        ax.plot(future_x, future_y, label='Predicted Close', color='red')
        ax.set_title(f"{symbol} - Forecast for Next 7 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Step 6: Interactive Candlestick + Volume + RSI Plot
        st.write("### 📊 Interactive Visualization")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"type": "candlestick"}],
                   [{"type": "bar"}],
                   [{"type": "scatter"}]]
        )

        fig.add_trace(go.Candlestick(
            x=filtered.index,
            open=filtered['Open'],
            high=filtered['High'],
            low=filtered['Low'],
            close=filtered['Close'],
            name='Candlestick'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=filtered.index,
            y=filtered['MA_20'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='MA 20'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=filtered.index,
            y=filtered['Volume'],
            name='Volume',
            marker=dict(color='rgba(153, 204, 255, 0.6)')
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=filtered.index,
            y=filtered['RSI'],
            mode='lines',
            name='RSI (14)',
            line=dict(color='orange')
        ), row=3, col=1)

        fig.update_layout(
            height=900,
            title=f"{symbol} - Candlestick, Volume & RSI",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            legend=dict(orientation='h')
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
