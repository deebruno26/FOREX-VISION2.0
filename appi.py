import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import requests
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="AI Forex Market Intelligence System",
    page_icon="", 
    layout="wide"
)

st.title("AI Forex Market Intelligence System")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
# ----------------------------------
# LOAD MODEL
# ----------------------------------

#@st.cache_resource
#def load_selected_model(pair):
  ##  models = {
    #    "USDCHF": "usdchf_model.keras",
   #     "EURUSD": "eurusd_model.keras",
  #      "GBPUSD": "gbpusd_model.keras",
  #      "USDJPY": "usdjpy_model.keras"
 #   }

    # The .keras format loads cleanly without extra flags
   # return load_model(models[pair], compile=False)
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
    data = yf.download(
        tickers[pair],
        period="3mo",
        interval="1d",
        progress=False
    )
    return data

# ----------------------------------
# NEWS FETCHER
# ----------------------------------
@st.cache_data(ttl=300)
def fetch_news(pair):
    keywords = {
        "EURUSD": "EUR USD forex",
        "GBPUSD": "GBP USD forex",
        "USDJPY": "USD JPY forex",
        "USDCHF": "USD CHF forex"
    }
    query = keywords.get(pair, "forex")
    query_encoded = query.replace(" ", "+")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    articles = []

    # GNews API
    try:
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={query_encoded}"
            f"&lang=en"
            f"&country=us"
            f"&max=10"
            f"&apikey=82b89f8d0b498886b827f52ba30d0edd" 
        )
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "articles" in data:
                for article in data["articles"]:
                    title = article.get("title", "")
                    desc  = article.get("description", "")
                    if title:
                        articles.append(f"{title}. {desc}")
    except Exception:
        pass

    # Yahoo RSS Fallback
    if not articles:
        try:
            rss_keywords = {
                "EURUSD": "EURUSD",
                "GBPUSD": "GBPUSD",
                "USDJPY": "USDJPY",
                "USDCHF": "USDCHF"
            }
            ticker = rss_keywords.get(pair, "EURUSD")
            rss_url = (
                f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={ticker}=X&region=US&lang=en-US"
            )
            rss_response = requests.get(rss_url, headers=headers, timeout=5)
            
            if rss_response.status_code != 200:
                # Generic Fallback
                rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDXY&region=US&lang=en-US"
                rss_response = requests.get(rss_url, headers=headers, timeout=5)

            if rss_response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(rss_response.content)
                for item in root.iter("item"):
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        articles.append(title_elem.text)
        except Exception as e:
            print(f"RSS Error: {e}")

    if not articles:
        return ["Market data currently unavailable. Neutral sentiment applied."]

    return articles[:10]

# ----------------------------------
# SENTIMENT ANALYSIS
# ----------------------------------
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

def analyze_sentiment(articles):
    analyzer = load_vader()
    scores = []
    details = []

    for article in articles:
        vs = analyzer.polarity_scores(article)
        compound = vs["compound"]
        scores.append(compound)

        # UPDATED THRESHOLD: 0.01 for higher sensitivity
        # This captures "Falls", "Drops", "Growth" better
        if compound >= 0.01:
            label = "BULLISH"
        elif compound <= -0.01:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        details.append((article[:120], round(compound, 4), label))

    if scores:
        aggregate = round(sum(scores) / len(scores), 4)
    else:
        aggregate = 0.0

    # UPDATED THRESHOLD: 0.01
    if aggregate >= 0.01:
        agg_label = "BULLISH"
    elif aggregate <= -0.01:
        agg_label = "BEARISH"
    else:
        agg_label = "NEUTRAL"

    return scores, aggregate, agg_label, details

# ----------------------------------
# DECISION ENGINE
# ----------------------------------
def final_decision(model_signal, agg_label):
    """
    Professional Decision Logic.
    1. Neutral News -> Trust AI Model.
    2. Agreement -> Strong Signal.
    3. Conflict -> Warning.
    """
    
    if agg_label == "NEUTRAL":
        if model_signal == "BUY":
            return "AI SIGNAL: BUY"
        elif model_signal == "SELL":
            return "AI SIGNAL: SELL"
        else:
            return "NO SIGNAL"

    if model_signal == "BUY" and agg_label == "BULLISH":
        return "STRONG BUY"
    
    if model_signal == "SELL" and agg_label == "BEARISH":
        return "STRONG SELL"

    if (model_signal == "BUY" and agg_label == "BEARISH") or \
       (model_signal == "SELL" and agg_label == "BULLISH"):
        return "CONFLICTING SIGNAL"

    if agg_label == "BULLISH":
        return "WEAK BUY"
    if agg_label == "BEARISH":
        return "WEAK SELL"

    return "HOLD"

# ----------------------------------
# VOLATILITY FILTER
# ----------------------------------
def volatility_filter(prices, threshold=0.01):
    returns = np.diff(prices) / prices[:-1]
    vol = np.std(returns)
    return vol > threshold

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
    st.error("Insufficient data to run analysis.")
    st.stop()

close_prices = data["Close"].to_numpy().reshape(-1)
latest_price = float(close_prices[-1])

# ----------------------------------
# HEADER INFO
# ----------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("Currency Pair", pair)
with col2:
    st.metric("Current Price", round(latest_price, 5))

# ----------------------------------
# SENTIMENT SECTION
# ----------------------------------
st.subheader("Market Sentiment Analysis")

with st.spinner("Analyzing market news..."):
    articles = fetch_news(pair)
    scores, aggregate, agg_label, details = analyze_sentiment(articles)

# Sentiment Metrics
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Sources Analyzed", len(articles))
with col_s2:
    st.metric("Sentiment Score", aggregate)
with col_s3:
    st.metric("Market Sentiment", agg_label)

# Article Breakdown
if details:
    with st.expander("View News Analysis"):
        df_news = pd.DataFrame(details, columns=["Headline", "Score", "Sentiment"])
        st.dataframe(df_news, use_container_width=True)

# ----------------------------------
# MARKET STATUS (No Decision here)
# ----------------------------------
st.subheader("Market Status")

if volatility_filter(close_prices):
    st.warning("High Market Volatility Detected")

st.write(f"**Aggregate Sentiment:** {agg_label} (Score: {aggregate})")

# ----------------------------------
# CHART
# ----------------------------------
st.subheader("Price History")
st.line_chart(pd.DataFrame({"Close": close_prices}))

# ----------------------------------
# LOAD MODEL & SCALER
# ----------------------------------
model   = load_selected_model(pair)
scaler  = load_scaler(pair)

# ----------------------------------
# PREDICTION BUTTON
# ----------------------------------
st.divider()
if st.button("GENERATE AI FORECAST", use_container_width=True):

    try:
        # Prepare Data
        last_30 = close_prices[-30:].reshape(-1, 1)
        last_30_scaled = scaler.transform(last_30)
        input_data = last_30_scaled.reshape(1, 30, 1)

        # Run Model
        pred_scaled = model.predict(input_data, verbose=0)
        pred_price  = scaler.inverse_transform(pred_scaled)
        prediction  = float(pred_price[0][0])

        diff = prediction - latest_price
        ml_signal = "BUY" if prediction > latest_price else "SELL"

        st.subheader("Forecast Results")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Current Price", f"{latest_price:.5f}")
        with col_p2:
            st.metric("AI Forecast", f"{prediction:.5f}")
        with col_p3:
            st.metric("Expected Change", f"{diff:+.5f}")

        # Direction
        if prediction > latest_price:
            st.markdown("### MODEL PREDICTION: UPWARD TREND")
        else:
            st.markdown("### MODEL PREDICTION: DOWNWARD TREND")

        # Final Decision
        st.subheader("Trading Recommendation")
        final_sig = final_decision(ml_signal, agg_label)

        # Professional Coloring
        if "STRONG BUY" in final_sig:
            st.success(final_sig)
        elif "STRONG SELL" in final_sig:
            st.error(final_sig)
        elif "CONFLICTING" in final_sig:
            st.warning(final_sig)
        elif "BUY" in final_sig:
            st.info(final_sig)
        elif "SELL" in final_sig:
            st.warning(final_sig)
        else:
            st.info(final_sig)

        st.write(f"**Technical Signal:** {ml_signal}")
        st.write(f"**Fundamental Signal:** {agg_label}")

        st.caption("Disclaimer: This system provides analytical data only. Not financial advice.")

    except Exception as e:
        st.error(f"System Error: {str(e)}")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption("Forex Vision 2.0 — System Operational")
