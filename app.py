import os
import streamlit as st
import ccxt
import pandas as pd
import plotly.express as px

# ==============================
# 1. LOAD API KEYS FROM RENDER
# ==============================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")

# ======================
# 2. INIT EXCHANGES
# ======================
@st.cache_resource
def init_exchanges():
    try:
        binance = ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_SECRET_KEY
        })
    except:
        binance = None

    try:
        bybit = ccxt.bybit({
            "apiKey": BYBIT_API_KEY,
            "secret": BYBIT_SECRET_KEY
        })
    except:
        bybit = None

    return binance, bybit

binance, bybit = init_exchanges()

# ======================
# 3. STREAMLIT UI
# ======================
st.title("🚀 Exahash AI — Multi Exchange Dashboard")
st.write("Monitoring & Analysis")

exchange_choice = st.selectbox(
    "Pilih Exchange:",
    ["Binance", "Bybit"]
)

symbol = st.text_input("Masukkan simbol (contoh: BTC/USDT)", "BTC/USDT")
limit = st.slider("Jumlah data OHLCV", 50, 500, 200)

# ======================
# 4. FETCH MARKET DATA
# ======================
@st.cache_data(ttl=60)
def fetch_ohlcv(exchange, symbol, limit):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

if exchange_choice == "Binance" and binance:
    data = fetch_ohlcv(binance, symbol, limit)
elif exchange_choice == "Bybit" and bybit:
    data = fetch_ohlcv(bybit, symbol, limit)
else:
    data = None

# ======================
# 5. DISPLAY CHART
# ======================
if data is not None:
    st.write("### 📈 Price Chart")
    fig = px.line(data, x="time", y="close", title=f"{symbol} Price")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### 💹 Volume")
    fig2 = px.bar(data, x="time", y="volume")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Tidak bisa memuat data. Cek API key atau simbol.")
