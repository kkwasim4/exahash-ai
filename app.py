import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os

# ==========================================
# 1. SETUP: DARK MODE & TRADER THEME
# ==========================================
st.set_page_config(
    page_title="ExaHash Alpha Hunter",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# TRADING TERMINAL STYLE (Dark & Neon)
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
    }
    div[data-testid="stMetricValue"] { color: #00ff7f !important; font-family: 'Monospace'; }
    
    /* Tables */
    div[data-testid="stDataFrame"] { background-color: #0e1117; }
    
    /* Custom Badge */
    .pump-badge {
        background-color: #238636;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        border: 1px solid #2ea043;
    }
    .vol-badge {
        background-color: #1f6feb;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }
    
    /* AI Card */
    .alpha-card {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #00ff7f;
    }
</style>
""", unsafe_allow_html=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. ALPHA ENGINE (MOMENTUM LOGIC)
# ==========================================
class AlphaEngine:
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    @st.cache_data(ttl=60) # Refresh tiap 60 detik untuk data momentum
    def scan_market(_self):
        all_coins = []
        # Tarik 400 koin teratas (Cukup untuk cari hidden gem, jangan terlalu bawah nanti scam)
        pages = [1, 2] 
        
        for page in pages:
            try:
                params = {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 200, 
                    "page": page,
                    "sparkline": "true",
                    "price_change_percentage": "1h,24h,7d"
                }
                res = requests.get(f"{_self.BASE_URL}/coins/markets", params=params)
                if res.status_code == 200:
                    all_coins.extend(res.json())
            except: pass
            
        if not all_coins: return pd.DataFrame()
        
        df = pd.DataFrame(all_coins)
        
        # --- RUMUS PUMP DETECTOR ---
        
        # 1. Volume Intensity (Turnover)
        # Jika Volume > 20% Market Cap, artinya ada akumulasi masif
        df['vol_intensity'] = df['total_volume'] / df['market_cap']
        
        # 2. Momentum Score (0-100)
        # Fokus pada kenaikan 1 jam terakhir (Immediate Pump)
        # Rumus: (1h% * 3) + (24h% * 1). Kita beri bobot tinggi pada 1H.
        df['1h_change'] = df['price_change_percentage_1h_in_currency']
        df['24h_change'] = df['price_change_percentage_24h_in_currency']
        
        # Normalisasi NaNs
        df['1h_change'] = df['1h_change'].fillna(0)
        df['24h_change'] = df['24h_change'].fillna(0)
        
        df['momentum_score'] = (df['1h_change'] * 3) + df['24h_change']
        
        # 3. Near ATH (Breakout Potential)
        df['dist_from_ath'] = df['ath_change_percentage'] # Usually negative
        
        # 4. Filter Volume Minimum (Hilangkan koin mati/scam dengan volume < $1M)
        df = df[df['total_volume'] > 1_000_000].copy()
        
        # Formatting Columns
        cols = [
            'image', 'symbol', 'name', 'current_price', 
            '1h_change', '24h_change', 'total_volume', 
            'vol_intensity', 'momentum_score', 'sparkline_in_7d', 'dist_from_ath'
        ]
        
        final = df[cols].copy()
        final.columns = [
            'Icon', 'Ticker', 'Name', 'Price', 
            '1h %', '24h %', 'Volume', 
            'Vol/Cap', 'Score', 'Chart', 'ATH Dist %'
        ]
        final['Ticker'] = final['Ticker'].str.upper()
        
        # Sort by Momentum Score by default
        return final.sort_values(by='Score', ascending=False)

engine = AlphaEngine()

# ==========================================
# 3. AI ALPHA CALLER
# ==========================================
def get_alpha_call(row):
    if not GOOGLE_API_KEY: return "AI Key Missing."
    
    # Prompt khusus mencari PUMP
    prompt = f"""
    You are a "Degen" Crypto Analyst looking for pumps and breakouts.
    Analyze {row['Ticker']}:
    - 1h Change: {row['1h %']:.2f}% (Immediate Action)
    - 24h Change: {row['24h %']:.2f}%
    - Volume Intensity: {row['Vol/Cap']:.2f} (Normal is 0.05, Pump is > 0.2)
    - Distance from ATH: {row['ATH Dist %']:.2f}%
    
    Is this a "PUMP SIGNAL"?
    Output strict format:
    **SIGNAL:** [BUY BREAKOUT / WAIT / FAKE PUMP]
    **CONFIDENCE:** [0-100]%
    **REASON:** (One short sentence about volume/momentum).
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content(prompt).text
    except: return "AI Busy."

# ==========================================
# 4. DASHBOARD UI
# ==========================================

# HEADER
c1, c2 = st.columns([8, 2])
with c1:
    st.title("🚀 ExaHash Alpha Hunter")
    st.caption("Momentum Scanner • Volume Shock Detector • Breakout Radar")
with c2:
    if st.button("🔥 SCAN MARKET"):
        st.cache_data.clear()
        st.rerun()

# LOAD DATA
with st.spinner("Scanning for high momentum assets..."):
    df = engine.scan_market()

if not df.empty:
    # --- HOT METRICS ---
    # Cari koin yang lagi "Gila" (1h change > 3% dan Volume tinggi)
    hot_coins = df[ (df['1h %'] > 2) & (df['Vol/Cap'] > 0.1) ]
    top_vol = df.sort_values('Vol/Cap', ascending=False).iloc[0]
    top_gain = df.sort_values('24h %', ascending=False).iloc[0]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pump Candidates", f"{len(hot_coins)} Assets", "Active Now")
    m2.metric("Highest Volume Stress", f"{top_vol['Ticker']}", f"{top_vol['Vol/Cap']:.2f} Ratio")
    m3.metric("Top Gainer (24h)", f"{top_gain['Ticker']}", f"+{top_gain['24h %']:.1f}%")
    m4.metric("Market Sentiment", "Greed" if len(hot_coins) > 5 else "Neutral")

    # --- TABS STRATEGI ---
    tab1, tab2, tab3 = st.tabs(["🚀 MOMENTUM (Pumps)", "🐋 VOLUME SHOCK", "💎 NEAR ATH"])
    
    with tab1:
        st.markdown("### ⚡ Fast Movers (High 1H %)")
        st.write("Koin yang sedang bergerak **sekarang**. Fokus pada kolom **1h %**.")
        
        # Styling Tabel Momentum
        st.dataframe(
            df,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "Price": st.column_config.NumberColumn(format="$%.4f"),
                "1h %": st.column_config.NumberColumn(format="%.2f%%"), # Highlight logic auto by value?
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "Volume": st.column_config.NumberColumn(format="$%d"),
                "Vol/Cap": st.column_config.ProgressColumn("Intensity", min_value=0, max_value=1, format="%.2f"),
                "Chart": st.column_config.LineChartColumn("Trend"),
                "Score": st.column_config.NumberColumn(help="High Score = Strong Pump Algo"),
                "ATH Dist %": None
            },
            height=500,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="mom_select"
        )
        
    with tab2:
        st.markdown("### 🐋 Whale Accumulation (High Vol/Cap)")
        st.write("Koin dengan volume beli yang tidak wajar dibanding Market Cap-nya. Seringkali mendahului kenaikan harga.")
        
        # Filter: Urutkan berdasarkan Vol/Cap
        df_vol = df.sort_values(by='Vol/Cap', ascending=False)
        st.dataframe(
            df_vol,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "Vol/Cap": st.column_config.ProgressColumn("Buying Pressure", min_value=0, max_value=2, format="%.2f"),
                "1h %": st.column_config.NumberColumn(format="%.2f%%"),
                "Chart": st.column_config.LineChartColumn("Trend"),
                "ATH Dist %": None, "Score": None
            },
            use_container_width=True,
            height=500
        )

    with tab3:
        st.markdown("### 💎 Breakout Ready (Near ATH)")
        st.write("Koin yang harganya sudah dekat dengan All Time High (< 15% drop). Jika tembus, biasanya pump keras.")
        
        # Filter: ATH Dist > -15%
        df_ath = df[df['ATH Dist %'] > -15].sort_values('ATH Dist %', ascending=False)
        st.dataframe(
            df_ath,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "ATH Dist %": st.column_config.NumberColumn("Dist to ATH", format="%.2f%%"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Chart": st.column_config.LineChartColumn("Trend"),
                "Vol/Cap": None, "Score": None
            },
            use_container_width=True
        )

    # --- ALPHA SIGNAL PANEL ---
    # Logic: Ambil seleksi dari Tab 1 (Momentum)
    sel = st.session_state.mom_select
    if sel and sel["selection"]["rows"]:
        idx = sel["selection"]["rows"][0]
        # Perhatikan: row index harus diambil dari dataframe asli yang sudah di sort di Tab 1
        # Karena st.dataframe menampilkan df yang sudah di sort engine, indexnya lurus
        row = df.iloc[idx]
        
        st.markdown("---")
        c_chart, c_signal = st.columns([2, 1])
        
        with c_chart:
            # Render Chart Lebih Detail
            st.subheader(f"{row['Ticker']} Price Action")
            
            # Simulated Candle Data from Sparkline (Approximation)
            spk = row['Chart']
            if isinstance(spk, dict): spk = spk.get('price', [])
            
            if len(spk) > 0:
                fig = px.line(y=spk, title=f"7 Day Trend: {row['Name']}")
                fig.update_traces(line_color='#00ff7f', line_width=3)
                fig.update_layout(
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font_color='white',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor='#30363d')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with c_signal:
            st.markdown('<div class="alpha-card">', unsafe_allow_html=True)
            st.markdown(f"### 🤖 AI Alpha Call: {row['Ticker']}")
            
            if st.button("GENERATE SIGNAL", key="btn_alpha"):
                with st.spinner("Analyzing Momentum & Volume..."):
                    signal = get_alpha_call(row)
                    st.markdown(signal)
            else:
                st.info("Click to ask Gemini about Pump Probability.")
            
            st.divider()
            st.metric("1H Momentum", f"{row['1h %']:.2f}%")
            st.metric("Volume Strength", f"{row['Vol/Cap']:.2f}", delta="High" if row['Vol/Cap'] > 0.1 else "Normal")
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("API Error. CoinGecko sedang sibuk. Coba lagi dalam 1 menit.")
