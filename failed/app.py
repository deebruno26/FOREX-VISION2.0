
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="AI Forex Market Intelligence System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Forex Market Intelligence System")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_selected_model(pair):

    models = {
        "USDCHF": "usdchf_model.h5",
        "EURUSD": "eurusd_model.h5",
        "GBPUSD": "gbpusd_model.h5",
        "USDJPY": "usdjpy_model.h5"
    }

    return load_model(models[pair], compile=False)


# ----------------------------------
# LOAD SCALER
# ----------------------------------
@st.cache_resource
def load_scaler(pair):

    scalers = {
        "USDCHF": "usdchf_scaler.pkl",
        "EURUSD": "eurusd_scaler.pkl",
        "GBPUSD": "gbpusd_scaler.pkl",
        "USDJPY": "usdjpy_scaler.pkl"
    }

    return joblib.load(scalers[pair])


# ----------------------------------
# MARKET DATA
# ----------------------------------
def get_market_data(pair):

    tickers = {
        "USDCHF": "USDCHF=X",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X"
    }

    ticker = tickers[pair]

    data = yf.download(
        ticker,
        period="3mo",
        interval="1d",
        progress=False
    )

    return data


# ----------------------------------
# SIDEBAR
# ----------------------------------
pair = st.sidebar.selectbox(
    "Select Trading Pair",
    ["USDCHF", "EURUSD", "GBPUSD", "USDJPY"]
)

# ----------------------------------
# LOAD DATA
# ----------------------------------
data = get_market_data(pair)

if len(data) < 30:
    st.error("Not enough data")
    st.stop()

close_prices = data["Close"].to_numpy().reshape(-1)

latest_price = float(close_prices[-1])

# ----------------------------------
# HEADER INFO
# ----------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Pair", pair)

with col2:
    st.metric("Current Price", round(latest_price, 4))


# ----------------------------------
# SENTIMENT ANALYSIS
# ----------------------------------
def get_sentiment_score(text):

    positive_words = [
        "bullish",
        "rise",
        "surge",
        "gain",
        "strong",
        "up"
    ]

    negative_words = [
        "crash",
        "fall",
        "drop",
        "weak",
        "bearish",
        "decline"
    ]

    score = 0

    for word in positive_words:
        if word in text.lower():
            score += 1

    for word in negative_words:
        if word in text.lower():
            score -= 1

    return score


sample_news = (
    "usd is showing strong bullish momentum "
    "after recent surge"
)

sentiment = get_sentiment_score(sample_news)


# ----------------------------------
# DECISION ENGINE
# ----------------------------------
def final_decision(signal, sentiment):

    if signal == "BUY" and sentiment > 0:
        return "STRONG BUY"

    if signal == "SELL" and sentiment < 0:
        return "STRONG SELL"

    if sentiment > 0:
        return "WEAK BUY"

    if sentiment < 0:
        return "WEAK SELL"

    return "NO CLEAR SIGNAL"


# ----------------------------------
# VOLATILITY FILTER
# ----------------------------------
def volatility_filter(prices, threshold=0.03):

    returns = np.diff(prices) / prices[:-1]

    vol = np.std(returns)

    return vol > threshold


# ----------------------------------
# SIGNAL GENERATOR
# ----------------------------------
def generate_signal(prices):

    if len(prices) < 2:
        return "NO CLEAR SIGNAL"

    return "BUY" if prices[-1] > prices[-2] else "SELL"


# ----------------------------------
# MARKET INTELLIGENCE
# ----------------------------------
st.subheader("Market Intelligence")

st.write("News Sentiment Score:", sentiment)

if volatility_filter(close_prices):
    st.warning("⚠ High Market Volatility Detected")

signal = generate_signal(close_prices)

st.write("AI Trend Signal:", signal)

final_signal = final_decision(signal, sentiment)

st.subheader("Final Decision")
st.success(final_signal)


# ----------------------------------
# CHART
# ----------------------------------
st.subheader("Market Chart")

st.line_chart(
    pd.DataFrame({
        "Close": close_prices
    })
)


# ----------------------------------
# LOAD MODEL & SCALER
# ----------------------------------
model = load_selected_model(pair)
scaler = load_scaler(pair)


# ----------------------------------
# AI PREDICTION BUTTON
# ----------------------------------
if st.button("Generate AI Signal"):

    try:

        last_30 = close_prices[-30:].reshape(-1, 1)

        last_30_scaled = scaler.transform(last_30)

        input_data = last_30_scaled.reshape(1, 30, 1)

        pred_scaled = model.predict(
            input_data,
            verbose=0
        )

        pred_price = scaler.inverse_transform(
            pred_scaled
        )

        prediction = float(pred_price[0][0])

        st.subheader("Prediction Results")

        st.success(
            f"Predicted Price: {prediction:.4f}"
        )

        diff = prediction - latest_price

        st.write(
            f"Current Price: {latest_price:.4f}"
        )

        st.write(
            f"Difference: {diff:.4f}"
        )

        if prediction > latest_price:

            st.markdown(
                "## 📈 BULLISH SIGNAL"
            )

            st.success(
                "Market expected to move UP"
            )

        else:

            st.markdown(
                "## 📉 BEARISH SIGNAL"
            )

            st.error(
                "Market expected to move DOWN"
            )

        st.warning(
            "⚠ This is an AI prediction using LSTM. "
            "Markets are highly volatile."
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )


# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption(
    "Built with LSTM + Streamlit + Yahoo Finance"
)

