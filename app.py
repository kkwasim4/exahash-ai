import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import google.generativeai as genai
import os
import time

# ==========================================
# 1. SETUP & UI THEME (TITAN EDITION)
# ==========================================
st.set_page_config(
    page_title="ExaHash TITAN | Pro Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PRO TRADER CSS
st.markdown("""
<style>
    /* Background & Global Font */
    .stApp { background-color: #0b0e11; color: #c9d1d9; font-family: 'Roboto Mono', monospace; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
    }
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700; }
    
    /* Tables */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; }
    
    /* Custom Badges */
    .signal-buy { background: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-weight:bold; }
    .signal-sell { background: #da3633; color: white; padding: 2px 8px; border-radius: 4px; font-weight:bold; }
    .signal-neu { background: #6e7681; color: white; padding: 2px 8px; border-radius: 4px; }
    
    /* AI Panel */
    .titan-card {
        background: #161b22;
        border-left: 4px solid #a371f7; /* Purple AI */
        padding: 20px;
        border-radius: 6px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. TECHNICAL ANALYSIS ENGINE (THE MATH)
# ==========================================
class TitanEngine:
    BASE_URL = "https://api.coingecko.com/api/v3"
    FG_URL = "https://api.alternative.me/fng/"
    
    def calculate_rsi(self, prices, period=14):
        """Menghitung RSI dari list harga (Sparkline)"""
        if len(prices) < period: return 50 # Default Neutral
        
        series = pd.Series(prices)
        delta = series.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] # Ambil RSI terakhir

    def get_fear_greed(self):
        try:
            res = requests.get(self.FG_URL).json()
            data = res['data'][0]
            return int(data['value']), data['value_classification']
        except:
            return 50, "Neutral"

    @st.cache_data(ttl=120) 
    def scan_market(_self):
        all_coins = []
        # Fetch 2 Pages (Top 300 Liquid Assets)
        for page in [1, 2]:
            try:
                params = {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 150, 
                    "page": page,
                    "sparkline": "true",
                    "price_change_percentage": "1h,24h,7d"
                }
                res = requests.get(f"{_self.BASE_URL}/coins/markets", params=params)
                if res.status_code == 200: all_coins.extend(res.json())
            except: pass
            
        if not all_coins: return pd.DataFrame()
        
        df = pd.DataFrame(all_coins)
        
        # --- ADVANCED CALCULATIONS ---
        
        # 1. Technicals (RSI) from Sparkline
        # Sparkline CoinGecko itu array harga per jam selama 7 hari (168 jam)
        # Kita pakai ini untuk hitung RSI 14-Jam terakhir
        df['prices_array'] = df['sparkline_in_7d'].apply(lambda x: x.get('price', []) if isinstance(x, dict) else [])
        df['RSI'] = df['prices_array'].apply(lambda x: _self.calculate_rsi(x, 14))
        
        # 2. Volume/Cap Ratio (Liquidity Stress)
        df['Vol/Cap'] = df['total_volume'] / df['market_cap']
        
        # 3. Deviation from ATH (Discount Factor)
        df['ATH Dist'] = df['ath_change_percentage']
        
        # 4. Signal Flagging
        def get_signal(row):
            score = 0
            # RSI Logic
            if row['RSI'] < 30: score += 2 # Oversold (Bullish Divergence Potential)
            elif row['RSI'] > 70: score -= 2 # Overbought
            
            # Momentum Logic
            if row['price_change_percentage_24h'] > 5: score += 1
            if row['Vol/Cap'] > 0.1: score += 1
            
            if score >= 3: return "STRONG BUY"
            elif score >= 1: return "BUY"
            elif score <= -2: return "SELL"
            else: return "HOLD"
            
        df['Signal'] = df.apply(get_signal, axis=1)
        
        # Clean Data
        cols = [
            'image', 'symbol', 'name', 'current_price', 
            'price_change_percentage_24h', 'total_volume', 'market_cap', 
            'RSI', 'Vol/Cap', 'ATH Dist', 'Signal', 'prices_array'
        ]
        final = df[cols].copy()
        final.columns = [
            'Icon', 'Ticker', 'Name', 'Price', 
            '24h %', 'Volume', 'Mkt Cap', 
            'RSI (14)', 'Vol/Cap', 'ATH Dist %', 'Signal', 'ChartData'
        ]
        final['Ticker'] = final['Ticker'].str.upper()
        return final

engine = TitanEngine()

# ==========================================
# 3. AI STRATEGIST
# ==========================================
def get_titan_strategy(row, fg_index):
    if not GOOGLE_API_KEY: return "⚠️ AI Key Missing"
    
    rsi_state = "Oversold" if row['RSI (14)'] < 30 else ("Overbought" if row['RSI (14)'] > 70 else "Neutral")
    
    prompt = f"""
    Act as an Institutional Algo-Trader. Analyze {row['Ticker']}:
    
    MARKET CONTEXT:
    - Global Fear & Greed: {fg_index} (0=Extreme Fear, 100=Greed)
    
    ASSET DATA:
    - Price: ${row['Price']}
    - Trend (24h): {row['24h %']:.2f}%
    - RSI (14hr): {row['RSI (14)']:.1f} ({rsi_state})
    - Volume Intensity: {row['Vol/Cap']:.2f} (Active if > 0.1)
    - Discount from ATH: {row['ATH Dist %']:.1f}%
    
    TASK:
    1. Technical Diagnosis: (Is it extended? Is it bottoming?)
    2. Smart Money Action: (Are whales accumulating or distributing based on volume?)
    3. Final Call: [AGGRESSIVE BUY / ACCUMULATE / WAIT / TAKE PROFIT]
    
    Keep it short, sharp, and professional.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model.generate_content(prompt).text
    except: return "AI Busy."

# ==========================================
# 4. TITAN DASHBOARD
# ==========================================

# --- GLOBAL METRICS ---
fg_val, fg_class = engine.get_fear_greed()
c_head1, c_head2 = st.columns([3, 1])

with c_head1:
    st.title("⚡ ExaHash TITAN")
    st.caption("Advanced Screener: RSI • Momentum • Volume • AI Logic")

with c_head2:
    # Fear & Greed Widget
    color = "#238636" if fg_val > 50 else "#da3633"
    st.markdown(f"""
    <div style="text-align:center; padding:10px; border:1px solid #30363d; border-radius:8px; background:#161b22;">
        <div style="font-size:12px; color:#8b949e;">FEAR & GREED</div>
        <div style="font-size:24px; font-weight:bold; color:{color};">{fg_val}</div>
        <div style="font-size:12px; color:{color};">{fg_class}</div>
    </div>
    """, unsafe_allow_html=True)

# LOAD DATA
with st.spinner("Calculating RSI & processing market data..."):
    df = engine.scan_market()

if not df.empty:
    # --- METRICS ROW ---
    # Filter "Oversold Gems" (RSI < 35 & Vol > 0.05) - Setup Reversal
    oversold_gems = df[(df['RSI (14)'] < 35) & (df['Vol/Cap'] > 0.05)]
    # Filter "Momentum Rockets" (RSI > 50 & 24h > 5%) - Setup Follow Trend
    rockets = df[(df['24h %'] > 5) & (df['Vol/Cap'] > 0.1)]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Breadth", f"{len(df)} Pairs")
    m2.metric("Oversold Opps", f"{len(oversold_gems)}", "Potential Reversal")
    m3.metric("Momentum Movers", f"{len(rockets)}", "Trend Following")
    m4.metric("BTC Price", f"${df.iloc[0]['Price']:,.0f}", f"{df.iloc[0]['24h %']:.2f}%")

    st.write("")
    
    # --- TABS FOR DIFFERENT TRADING STYLES ---
    tab_main, tab_rsi, tab_vol = st.tabs(["📊 MAIN BOARD", "📉 RSI SCANNER", "🐋 VOLUME RADAR"])
    
    with tab_main:
        st.subheader("Market Overview (Integrated Signals)")
        st.dataframe(
            df,
            column_config={
                "Icon": st.column_config.ImageColumn(width="small"),
                "Price": st.column_config.NumberColumn(format="$%.4f"),
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "RSI (14)": st.column_config.ProgressColumn(
                    "RSI Strength", min_value=0, max_value=100, format="%d"
                ),
                "Signal": st.column_config.TextColumn("Titan Algo"),
                "Vol/Cap": st.column_config.NumberColumn(format="%.2f"),
                "ChartData": None, "ATH Dist %": None
            },
            use_container_width=True,
            height=600,
            selection_mode="single-row",
            on_select="rerun",
            key="main_select"
        )

    with tab_rsi:
        st.subheader("Snipe the Bottom / Sell the Top")
        st.caption("Focus on RSI Extremes: <30 (Oversold) or >70 (Overbought)")
        # Sort by RSI ascending (Show lowest first)
        df_rsi = df.sort_values('RSI (14)', ascending=True)
        st.dataframe(
            df_rsi,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "RSI (14)": st.column_config.NumberColumn("RSI Value", format="%.1f"),
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "ATH Dist %": st.column_config.NumberColumn("Disc. from ATH", format="%.1f%%"),
                "Price": st.column_config.NumberColumn(format="$%.4f"),
                "Signal": None, "ChartData": None, "Vol/Cap": None
            },
            use_container_width=True
        )

    with tab_vol:
        st.subheader("Follow the Money (Whale Tracking)")
        # Sort by Volume/Cap
        df_vol = df.sort_values('Vol/Cap', ascending=False)
        st.dataframe(
            df_vol,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "Vol/Cap": st.column_config.ProgressColumn("Money Flow", min_value=0, max_value=1.5),
                "Volume": st.column_config.NumberColumn("24h Vol", format="$%d"),
                "Mkt Cap": st.column_config.NumberColumn(format="$%d"),
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "Signal": None, "ChartData": None, "RSI (14)": None
            },
            use_container_width=True
        )

    # --- AI ANALYSIS PANEL ---
    sel = st.session_state.main_select
    if sel and sel["selection"]["rows"]:
        idx = sel["selection"]["rows"][0]
        # Note: Index selection logic needs care if sorted, but streamlit default dataframe keeps index
        # Better: use the actual filtered df from main tab
        row = df.iloc[idx]
        
        st.divider()
        st.markdown(f"## 🔭 Titan Analysis: {row['Ticker']}")
        
        c_chart, c_ai = st.columns([2, 1])
        
        with c_chart:
            # CHART VISUALIZATION
            prices = row['ChartData']
            if len(prices) > 0:
                # Warna Chart berdasarkan RSI
                line_color = '#00ff7f' if row['RSI (14)'] > 50 else '#ff4b4b'
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=prices, mode='lines', line=dict(color=line_color, width=2), fill='tozeroy'))
                fig.update_layout(
                    title=f"7-Day Price Action ({len(prices)} hours)",
                    template="plotly_dark",
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=350,
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor='#30363d')
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with c_ai:
            st.markdown('<div class="titan-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Strategist")
            
            if st.button("GENERATE STRATEGY", key="gen_ai"):
                with st.spinner(f"Analyzing {row['Ticker']} Technicals..."):
                    strategy = get_titan_strategy(row, fg_val)
                    st.markdown(strategy)
            else:
                st.info("Ask Gemini to correlate RSI, Volume, and Trend.")
            
            st.divider()
            # QUICK STATS
            c1, c2 = st.columns(2)
            c1.metric("RSI State", f"{row['RSI (14)']:.0f}", "Neutral" if 30<row['RSI (14)']<70 else "EXTREME")
            c2.metric("Algo Signal", row['Signal'])
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Data Feed Initializing... Please refresh.")
