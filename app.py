import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="ExaHash AI | Institutional Grade",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Nansen/Coinbase" Professional Look
st.markdown("""
<style>
    /* Global Styles */
    .main { background-color: #f0f2f6; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    
    /* Metrics Card */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* Custom Cards */
    .pro-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6e8eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Whale Alert Text */
    .whale-alert { color: #d946ef; font-weight: bold; }
    .bull-text { color: #16a34a; }
    .bear-text { color: #dc2626; }
</style>
""", unsafe_allow_html=True)

# Environment Variables Loading
# Pastikan API Key diset di Render Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

# Init AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ Google API Key missing. AI features will be disabled.")

# ==========================================
# 2. DATA ENGINE (CCXT)
# ==========================================
class DataEngine:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET,
            'options': {'defaultType': 'future'} # Futures data is often better for flow
        })

    @st.cache_data(ttl=10)
    def fetch_ticker_data(_self, symbol):
        try:
            ticker = _self.exchange.fetch_ticker(symbol)
            # Fetch recent trades for Whale Analysis
            trades = _self.exchange.fetch_trades(symbol, limit=100)
            # Fetch OHLCV for Risk Calculation
            ohlcv = _self.exchange.fetch_ohlcv(symbol, '1h', limit=24) 
            
            return {
                "ticker": ticker,
                "trades": trades,
                "ohlcv": ohlcv,
                "valid": True
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def calculate_risk_metrics(self, ohlcv_data):
        if not ohlcv_data: return 0, 0
        df = pd.DataFrame(ohlcv_data, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        
        # Volatility (Standard Deviation of returns)
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].std() * 100 # Percentage
        
        # RSI Simple Calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return volatility, rsi.iloc[-1]

    def get_whale_activity(self, trades, min_usd=50000):
        whales = []
        net_flow = 0
        for t in trades:
            cost = t['cost'] if 'cost' in t else (t['price'] * t['amount'])
            if cost >= min_usd:
                whales.append(t)
                if t['side'] == 'buy': net_flow += cost
                else: net_flow -= cost
        return whales, net_flow

engine = DataEngine()

# ==========================================
# 3. UI COMPONENTS
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🧭 ExaHash Controller")
    selected_symbol = st.selectbox("Select Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"])
    
    st.markdown("---")
    st.markdown("### AI Settings")
    ai_model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-pro"])
    
    st.markdown("---")
    st.info("💡 Data Source: Binance Futures (Real-time)")

# --- MAIN LOGIC ---
st.title(f"ExaHash AI Dashboard: {selected_symbol}")

# Fetch Data
data = engine.fetch_ticker_data(selected_symbol)

if data['valid']:
    ticker = data['ticker']
    trades = data['trades']
    ohlcv = data['ohlcv']
    
    current_price = ticker['last']
    change_24h = ticker['percentage']
    volume_24h = ticker['quoteVolume']
    
    volatility, rsi = engine.calculate_risk_metrics(ohlcv)
    whales, whale_net_flow = engine.get_whale_activity(trades)
    
    # 1. TOP METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${current_price:,.2f}", f"{change_24h:.2f}%")
    c2.metric("24h Volume", f"${volume_24h/1000000:.2f}M")
    c3.metric("RSI (1H)", f"{rsi:.1f}", "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"))
    c4.metric("Volatility", f"{volatility:.2f}%")

    # 2. TRADINGVIEW WIDGET (PROFESSIONAL CHART)
    st.markdown("### 📉 Live Market Chart")
    # TradingView Widget Embed
    tv_symbol = f"BINANCE:{selected_symbol.replace('/','')}"
    
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{tv_symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=500)

    # 3. AI RISK SCORING & ANALYSIS
    col_ai, col_nansen = st.columns([1, 1])

    with col_ai:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Risk Scorer")
        
        # Calculate Logic for Prompt
        sentiment_label = "BULLISH" if whale_net_flow > 0 else "BEARISH"
        risk_level = "HIGH" if volatility > 2.0 else "MODERATE"
        
        if st.button("Generate AI Risk Report"):
            if GOOGLE_API_KEY:
                with st.spinner("Analyzing market structure..."):
                    prompt = f"""
                    Act as a Senior Crypto Risk Analyst. Analyze {selected_symbol} based on this real-time data:
                    - Price: ${current_price}
                    - 24h Change: {change_24h}%
                    - RSI (1h): {rsi:.1f}
                    - Volatility: {volatility:.2f}%
                    - Whale Net Flow (Last 100 trades): ${whale_net_flow:,.2f}
                    
                    Provide a professional JSON-style output with:
                    1. Risk Score (0-100, where 100 is extreme risk).
                    2. Market Sentiment (Bullish/Bearish/Neutral).
                    3. Key Resistance/Support levels (estimate based on price).
                    4. Short Actionable Insight for Institutional Traders.
                    """
                    try:
                        model = genai.GenerativeModel(ai_model)
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
            else:
                st.error("Configure GOOGLE_API_KEY to use AI.")
        else:
            st.write("Click button to analyze market data with Gemini.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. NANSEN-STYLE WHALE FEED
    with col_nansen:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.subheader("🐋 Whale Signal Feed (>$50k)")
        
        if whales:
            df_whales = pd.DataFrame(whales)
            df_whales['time'] = pd.to_datetime(df_whales['timestamp'], unit='ms').dt.strftime('%H:%M:%S')
            df_whales['value'] = df_whales['cost'].apply(lambda x: f"${x:,.0f}")
            df_whales['side'] = df_whales['side'].str.upper()
            
            # Styling grid
            for idx, row in df_whales.iterrows():
                color = "#dcfce7" if row['side'] == 'BUY' else "#fee2e2" # Light Green / Red
                text_color = "green" if row['side'] == 'BUY' else "red"
                icon = "🟢" if row['side'] == 'BUY' else "🔴"
                
                st.markdown(
                    f"""
                    <div style="background-color:{color}; padding:8px; border-radius:6px; margin-bottom:8px; font-size:14px; display:flex; justify-content:space-between;">
                        <span>{icon} <b>{row['side']}</b></span>
                        <span>{row['amount']:.4f} {selected_symbol.split('/')[0]}</span>
                        <span style="font-weight:bold;">{row['value']}</span>
                        <span style="color:#666;">{row['time']}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("No whale movements detected in the last 100 trades.")
            
        st.caption(f"Net Whale Flow: ${whale_net_flow:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error(f"Failed to load data: {data.get('error')}")

# Footer
st.markdown("---")
st.markdown("<center style='color:#888;'>ExaHash AI • Powered by Gemini 2.0 & CCXT • Institutional Data</center>", unsafe_allow_html=True)
