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
    data = yf.download(
        tickers[pair],
        period="3mo",
        interval="1d",
        progress=False
    )
    return data


# ----------------------------------
# REAL NEWS FETCHER
# Uses GNews free API — no key needed
# ----------------------------------
@st.cache_data(ttl=300)
def fetch_news(pair):
    # 1. Keywords for GNews (Keep these)
    keywords = {
        "EURUSD": "EUR USD forex",
        "GBPUSD": "GBP USD forex",
        "USDJPY": "USD JPY forex",
        "USDCHF": "USD CHF forex" # Try shorter keywords
    }
    query = keywords.get(pair, "forex")
    query_encoded = query.replace(" ", "+")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    articles = []

    # --- TRY GNEWS ---
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

    # --- FALLBACK: YAHOO RSS ---
    # If no articles found yet, try Yahoo.
    if not articles:
        try:
            # Attempt specific pair RSS first
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
            
            # IF SPECIFIC PAIR RSS FAILS (Status 404 or empty), USE GENERAL FOREX RSS
            if rss_response.status_code != 200:
                 # Fallback to general Forex news (DXY, EURUSD, etc usually covered here)
                rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDXY&region=US&lang=en-US"
                rss_response = requests.get(rss_url, headers=headers, timeout=5)

            if rss_response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(rss_response.content)
                # Yahoo RSS namespace handling
                # Sometimes titles are buried in namespaces, but standard .find usually works for 'title'
                for item in root.iter("item"):
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        articles.append(title_elem.text)
        except Exception as e:
            print(f"RSS Error: {e}")

    # --- FINAL SAFETY NET ---
    # If STILL no articles, return a generic placeholder so the app doesn't crash
    if not articles:
        return ["Forex markets are currently closed. General sentiment analysis unavailable."]

    return articles[:10]
# ----------------------------------
# REAL VADER SENTIMENT ANALYSIS
# ----------------------------------
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


def analyze_sentiment(articles):
    """
    Runs VADER sentiment on each article headline.
    Returns:
        - scores: list of per-article compound scores
        - aggregate: average compound score
        - label: Bullish / Bearish / Neutral
        - details: list of (headline, score, label) tuples
    """
    analyzer = load_vader()
    scores = []
    details = []

    for article in articles:
        vs = analyzer.polarity_scores(article)
        compound = vs["compound"]
        scores.append(compound)

        if compound >= 0.05:
            label = "🟢 Bullish"
        elif compound <= -0.05:
            label = "🔴 Bearish"
        else:
            label = "⚪ Neutral"

        details.append((article[:120], round(compound, 4), label))

    if scores:
        aggregate = round(sum(scores) / len(scores), 4)
    else:
        aggregate = 0.0

    if aggregate >= 0.05:
        agg_label = "BULLISH"
    elif aggregate <= -0.05:
        agg_label = "BEARISH"
    else:
        agg_label = "NEUTRAL"

    return scores, aggregate, agg_label, details


# ----------------------------------
# DECISION ENGINE
# ----------------------------------
# ----------------------------------
# DECISION ENGINE (UPDATED)
# ----------------------------------
def final_decision(model_signal, agg_label):
    """
    Combines AI Model Signal with News Sentiment.
    
    Rules:
    1. If News is Neutral/Mixed, trust the AI Model.
    2. If News and Model agree, Strong Signal.
    3. If News fights Model, Warning.
    """
    
    # 1. If Sentiment is NEUTRAL (or No Clear Signal), trust the MODEL
    if agg_label == "NEUTRAL":
        if model_signal == "BUY":
            return "🤖 AI SIGNAL: BUY (Neutral News)"
        elif model_signal == "SELL":
            return "🤖 AI SIGNAL: SELL (Neutral News)"
        else:
            return "➖ NO CLEAR SIGNAL"

    # 2. Perfect Alignment (Model + News agree)
    if model_signal == "BUY" and agg_label == "BULLISH":
        return "💥 STRONG BUY 🚀 (Model + News)"
    
    if model_signal == "SELL" and agg_label == "BEARISH":
        return "💥 STRONG SELL 🔻 (Model + News)"

    # 3. Conflict (Model says Up, News says Down, or vice versa)
    if (model_signal == "BUY" and agg_label == "BEARISH") or \
       (model_signal == "SELL" and agg_label == "BULLISH"):
        return "⚠️ CONFLICTING: News vs AI Model"

    # 4. Weak News Alignment (News is slightly one way, Model is Neutral)
    # (Assuming if we reach here, Model is neutral but News is strong)
    if agg_label == "BULLISH":
        return "📉 WEAK BUY (News Driven)"
    if agg_label == "BEARISH":
        return "📉 WEAK SELL (News Driven)"

    return "➖ HOLD / WAIT"


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
    st.error("Not enough data to run analysis.")
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
# SENTIMENT SECTION
# ----------------------------------
st.subheader("📰 Real-Time News Sentiment Analysis")

with st.spinner("Fetching latest news and analysing sentiment..."):
    articles = fetch_news(pair)
    scores, aggregate, agg_label, details = analyze_sentiment(articles)

# Sentiment summary
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Articles Analysed", len(articles))
with col_s2:
    st.metric("Aggregate Sentiment Score", aggregate)
with col_s3:
    color = "🟢" if agg_label == "BULLISH" else "🔴" if agg_label == "BEARISH" else "⚪"
    st.metric("Market Sentiment", f"{color} {agg_label}")

# Individual article breakdown
if details:
    with st.expander("📄 View Individual Article Sentiment"):
        df_news = pd.DataFrame(details, columns=["Headline", "Score", "Sentiment"])
        st.dataframe(df_news, use_container_width=True)
else:
    st.info("No news articles found for this pair. Using neutral sentiment.")

# ----------------------------------
# MARKET INTELLIGENCE
# ----------------------------------
st.subheader("🧠 Market Intelligence")

# ----------------------------------
# MARKET INTELLIGENCE
# ----------------------------------
st.subheader("🧠 Market Status")

# Keep the volatility warning, it looks smart
if volatility_filter(close_prices):
    st.warning("⚠️ High Market Volatility Detected — Trade with caution")

# Show the Sentiment clearly
st.write(f"**Market Sentiment:** {agg_label} (Score: {aggregate})")

# REMOVED THE OLD SIGNAL HERE TO AVOID CONFUSION WITH THE AI BUTTON

# ----------------------------------
# CHART
# ----------------------------------
st.subheader("📊 Market Chart")
st.line_chart(pd.DataFrame({"Close": close_prices}))

# ----------------------------------
# LOAD MODEL & SCALER
# ----------------------------------
model   = load_selected_model(pair)
scaler  = load_scaler(pair)

# ----------------------------------
# AI PREDICTION BUTTON
# ----------------------------------
# ----------------------------------
# AI PREDICTION BUTTON
# ----------------------------------
if st.button("🤖 Generate AI Prediction"):

    try:
        # 1. Prepare Data
        last_30 = close_prices[-30:].reshape(-1, 1)
        last_30_scaled = scaler.transform(last_30)
        input_data = last_30_scaled.reshape(1, 30, 1)

        # 2. Run Model
        pred_scaled = model.predict(input_data, verbose=0)
        pred_price  = scaler.inverse_transform(pred_scaled)
        prediction  = float(pred_price[0][0])

        # 3. Calculate Metrics
        diff = prediction - latest_price
        ml_signal = "BUY" if prediction > latest_price else "SELL"

        st.subheader("🔮 Prediction Results")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Current Price", f"{latest_price:.4f}")
        with col_p2:
            st.metric("Predicted Price", f"{prediction:.4f}")
        with col_p3:
            st.metric("Difference", f"{diff:+.4f}")

        # 4. Simple Direction Indicator
        if prediction > latest_price:
            st.markdown("## 📈 MODEL PREDICTS UP")
        else:
            st.markdown("## 📉 MODEL PREDICTS DOWN")

        # 5. COMBINED DECISION (THE FIX)
        st.subheader("🎯 Final Strategy Decision")
        
        # Call the new function
        final_sig = final_decision(ml_signal, agg_label)

        # Display the result with appropriate color
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

        # 6. Breakdown details
        st.write(f"**🤖 AI Model:** {ml_signal}")
        st.write(f"**📰 News Sentiment:** {agg_label}")

        st.warning(
            "⚠️ This is an AI prediction using LSTM + VADER Sentiment Analysis. "
            "Forex markets are highly volatile. This is not financial advice."
        )

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption(
    "Forex Vision 2.0 — Built with LSTM + VADER Sentiment + Streamlit + Yahoo Finance"
)