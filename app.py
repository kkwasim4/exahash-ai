# app_patched.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import google.generativeai as genai
import os
import time
from streamlit_autorefresh import st_autorefresh

# ==========================================
# CONFIG
# ==========================================
AUTO_REFRESH_INTERVAL = 15000  # milliseconds (15000ms = 15s)
autorefresh_count = st_autorefresh(interval=AUTO_REFRESH_INTERVAL, key="auto_refresh_counter")

LOGO_PATH = "/mnt/data/logo.png"  # <-- replace if you uploaded logo elsewhere
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALCHEMY_DEFAULT_TTL = 600  # seconds for contract detection cache

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 1. SETUP & UI THEME (KKWASIM4 EDITION)
# ==========================================
st.set_page_config(
    page_title="Kkwasim4 Spot Trading Scanner",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0e11; color: #c9d1d9; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700; }
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; }
    .signal-buy { background: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-weight:bold; }
    .signal-sell { background: #da3633; color: white; padding: 2px 8px; border-radius: 4px; font-weight:bold; }
    .signal-neu { background: #6e7681; color: white; padding: 2px 8px; border-radius: 4px; }
    .titan-card {
        background: #161b22;
        border-left: 4px solid #a371f7;
        padding: 20px;
        border-radius: 6px;
        margin-top: 10px;
    }
    .chat-bubble { padding:8px; margin:6px 0; border-radius:8px; }
    .chat-me { background:#0b1220; color:#bfe; text-align:right; }
    .chat-others { background:#1b1f24; color:#ddd; text-align:left; }
</style>
""", unsafe_allow_html=True)

# Sidebar info + logo + autorefresh debug
with st.sidebar:
    try:
        st.image(LOGO_PATH, width=120)
    except:
        st.write("")  # logo missing ok
    st.title("KKWASIM4 Spot Trading Scanner")
    st.caption("Auto-refresh every {}s (cached fetches).".format(AUTO_REFRESH_INTERVAL // 1000))
    st.write(f"Auto-refresh count: {autorefresh_count}")
    st.markdown("---")
    st.markdown("**Contract Scanner (Alchemy)**")
    alchemy_key_input = st.text_input("ALCHEMY_API_KEY", key="alc_key", placeholder="paste your Alchemy API key here")
    contract_input = st.text_input("Paste contract address to detect", key="contract_addr")
    if st.button("Detect Contract", key="detect_contract_btn"):
        if not alchemy_key_input:
            st.error("Provide Alchemy API key in sidebar input above.")
        elif not contract_input:
            st.error("Provide contract address first.")
        else:
            with st.spinner("Detecting contract..."):
                res = None
                try:
                    @st.cache_data(ttl=ALCHEMY_DEFAULT_TTL)
                    def _detect_contract_cached(contract_addr, alchemy_key):
                        url = f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}"
                        payload = {
                            "jsonrpc": "2.0",
                            "id": 0,
                            "method": "alchemy_getTokenMetadata",
                            "params": [contract_addr]
                        }
                        r = requests.post(url, json=payload, timeout=15).json()
                        return r
                    r = _detect_contract_cached(contract_input, alchemy_key_input)
                    # r may include 'result' with metadata
                    if r.get("result"):
                        meta = r["result"]
                        st.success(f"Detected: {meta.get('symbol','?')} — {meta.get('name','?')}")
                        st.json(meta)
                    else:
                        st.warning("No metadata found. The token may be non-standard or RPC blocked.")
                        st.write(r)
                except Exception as e:
                    st.error(f"Detect error: {e}")

# ==========================================
# 2. TECHNICAL ANALYSIS ENGINE (THE MATH)
# ==========================================
class TitanEngine:
    BASE_URL = "https://api.coingecko.com/api/v3"
    FG_URL = "https://api.alternative.me/fng/"

    def calculate_rsi(self, prices, period=14):
        if len(prices) < period: return 50
        s = pd.Series(prices)
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def calculate_obv(self, prices, total_volume):
        # approximate per-hour volume
        if not prices or len(prices) < 2:
            return 0
        per_hour_vol = (total_volume / 24.0) if total_volume and total_volume > 0 else 0
        obv = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += per_hour_vol
            elif prices[i] < prices[i-1]:
                obv -= per_hour_vol
        return float(obv)

    def calculate_mfi(self, prices, total_volume, period=14):
        if not prices or len(prices) < period + 1:
            return 50.0
        # approximate volumes per hour equally
        vols = np.full(len(prices), (total_volume / 24.0) if total_volume and total_volume > 0 else 1.0)
        typical = pd.Series(prices)
        money_flow = typical * vols
        pos = 0.0
        neg = 0.0
        for i in range(1, len(prices)):
            if typical.iloc[i] > typical.iloc[i-1]:
                pos += money_flow[i]
            else:
                neg += money_flow[i]
        if neg == 0:
            return 100.0
        mfr = pos / neg
        mfi = 100 - (100 / (1 + mfr))
        return float(mfi)

    def calculate_chaikin(self, prices, total_volume):
        if not prices or len(prices) < 2:
            return 0.0
        vols = np.full(len(prices), (total_volume / 24.0) if total_volume and total_volume > 0 else 1.0)
        mfvs = []
        for i in range(1, len(prices)):
            close = prices[i]
            open_p = prices[i-1]
            rng = abs(close - open_p) if abs(close - open_p) > 1e-9 else 1e-9
            mfv = ((close - open_p) / rng) * vols[i]
            mfvs.append(mfv)
        denom = np.sum(vols[1:]) if np.sum(vols[1:]) != 0 else 1.0
        return float(np.sum(mfvs) / denom)

    def get_fear_greed(self):
        try:
            res = requests.get(self.FG_URL, timeout=6).json()
            data = res['data'][0]
            return int(data['value']), data['value_classification']
        except:
            return 50, "Neutral"

    @st.cache_data(ttl=120)
    def scan_market(_self):
        all_coins = []
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
                res = requests.get(f"{_self.BASE_URL}/coins/markets", params=params, timeout=20)
                if res.status_code == 200:
                    all_coins.extend(res.json())
            except:
                pass

        if not all_coins:
            return pd.DataFrame()

        df = pd.DataFrame(all_coins)
        df['prices_array'] = df['sparkline_in_7d'].apply(lambda x: x.get('price', []) if isinstance(x, dict) else [])
        df['RSI'] = df['prices_array'].apply(lambda x: _self.calculate_rsi(x, 14))
        df['Vol/Cap'] = df['total_volume'] / (df['market_cap'].replace({0: np.nan}) )
        # Money Flow (estimated) based on last change * volume
        def calc_money_flow(row):
            prices = row['prices_array']
            if not isinstance(prices, list) or len(prices) < 2:
                return 0.0, 0.0
            open_p = prices[-2]
            close_p = prices[-1]
            vol = row.get('total_volume', 0) or 0
            raw_flow = (close_p - open_p) * vol
            money_in = raw_flow if raw_flow > 0 else 0.0
            money_out = abs(raw_flow) if raw_flow < 0 else 0.0
            return money_in, money_out

        df['Money In'], df['Money Out'] = zip(*df.apply(calc_money_flow, axis=1))

        # Extra indicators
        df['OBV'] = df.apply(lambda r: _self.calculate_obv(r['prices_array'], r.get('total_volume', 0) or 0), axis=1)
        df['MFI'] = df.apply(lambda r: _self.calculate_mfi(r['prices_array'], r.get('total_volume', 0) or 0), axis=1)
        df['Chaikin'] = df.apply(lambda r: _self.calculate_chaikin(r['prices_array'], r.get('total_volume', 0) or 0), axis=1)

        # Signal
        def get_signal(row):
            score = 0
            if row['RSI'] < 30: score += 2
            elif row['RSI'] > 70: score -= 2
            if row['price_change_percentage_24h'] and row['price_change_percentage_24h'] > 5: score += 1
            if row['Vol/Cap'] is not None and row['Vol/Cap'] > 0.1: score += 1
            if score >= 3: return "STRONG BUY"
            elif score >= 1: return "BUY"
            elif score <= -2: return "SELL"
            else: return "HOLD"

        df['Signal'] = df.apply(get_signal, axis=1)

        cols = [
            'image', 'symbol', 'name', 'current_price',
            'price_change_percentage_24h', 'total_volume', 'market_cap',
            'RSI', 'Vol/Cap', 'Money In', 'Money Out', 'OBV', 'MFI', 'Chaikin',
            'Signal', 'prices_array'
        ]
        final = df[cols].copy()
        final.columns = [
            'Icon', 'Ticker', 'Name', 'Price',
            '24h %', 'Volume', 'Mkt Cap',
            'RSI (14)', 'Vol/Cap', 'Money In', 'Money Out', 'OBV', 'MFI', 'Chaikin',
            'Signal', 'ChartData'
        ]
        final['Ticker'] = final['Ticker'].str.upper()
        final = final.reset_index(drop=True)
        return final

engine = TitanEngine()

# ==========================================
# 3. AI STRATEGIST (only on demand)
# ==========================================
def get_titan_strategy(row, fg_index):
    if not GOOGLE_API_KEY:
        return "⚠️ AI Key Missing"
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
    - Money In/Out (est): {row['Money In']:.0f} / {row['Money Out']:.0f}
    - OBV: {row['OBV']:.0f}, MFI: {row['MFI']:.1f}, Chaikin: {row['Chaikin']:.3f}

    TASK:
    1. Technical Diagnosis.
    2. Smart Money Action.
    3. Final Call: [AGGRESSIVE BUY / ACCUMULATE / WAIT / TAKE PROFIT]

    Keep short and professional.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Busy / Error: {e}"

# ==========================================
# 4. DASHBOARD
# ==========================================
fg_val, fg_class = engine.get_fear_greed()
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("<h1 style='color:#c9d1d9; margin:0;'>⚡ Kkwasim4 Spot Trading Scanner</h1>", unsafe_allow_html=True)
    st.caption("RSI • OBV • MFI • Chaikin • Money Flow • Contract Scanner • Auto-refresh")
with h2:
    color = "#238636" if fg_val > 50 else "#da3633"
    st.markdown(f"""
    <div style="text-align:center; padding:10px; border:1px solid #30363d; border-radius:8px; background:#161b22;">
        <div style="font-size:12px; color:#8b949e;">FEAR & GREED</div>
        <div style="font-size:24px; font-weight:bold; color:{color};">{fg_val}</div>
        <div style="font-size:12px; color:{color};">{fg_class}</div>
    </div>
    """, unsafe_allow_html=True)

# LOAD DATA (cached)
with st.spinner("Fetching market data (cached)..."):
    df = engine.scan_market()

if df.empty:
    st.warning("Data Feed Initializing or API blocked. Try again shortly.")
else:
    # Metrics row
    oversold_gems = df[(df['RSI (14)'] < 35) & (df['Vol/Cap'].fillna(0) > 0.05)]
    rockets = df[(df['24h %'] > 5) & (df['Vol/Cap'].fillna(0) > 0.1)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Breadth", f"{len(df)} Pairs")
    m2.metric("Oversold Opps", f"{len(oversold_gems)}", "Potential Reversal")
    m3.metric("Momentum Movers", f"{len(rockets)}", "Trend Following")
    m4.metric("Top Price (sample)", f"${df.iloc[0]['Price']:,.0f}", f"{df.iloc[0]['24h %']:.2f}%")

    st.write("")

    tab_main, tab_rsi, tab_vol = st.tabs(["📊 MAIN BOARD", "📉 RSI SCANNER", "🐋 VOLUME RADAR"])

    with tab_main:
        st.subheader("Market Overview (Integrated Signals)")
        # show dataframe with selection
        st.dataframe(
            df,
            column_config={
                "Icon": st.column_config.ImageColumn(width="small"),
                "Price": st.column_config.NumberColumn(format="$%.4f"),
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "RSI (14)": st.column_config.ProgressColumn("RSI Strength", min_value=0, max_value=100, format="%d"),
                "Signal": st.column_config.TextColumn("Titan Algo"),
                "Vol/Cap": st.column_config.NumberColumn(format="%.4f"),
                "Money In": st.column_config.NumberColumn(format="$%d"),
                "Money Out": st.column_config.NumberColumn(format="$%d"),
                "OBV": st.column_config.NumberColumn(format="%.0f"),
                "MFI": st.column_config.ProgressColumn("MFI", min_value=0, max_value=100, format="%d"),
                "Chaikin": st.column_config.NumberColumn(format="%.4f"),
                "ChartData": None
            },
            use_container_width=True,
            height=600,
            selection_mode="single-row",
            on_select="rerun",
            key="main_select"
        )

    with tab_rsi:
        st.subheader("Snipe the Bottom / Sell the Top")
        df_rsi = df.sort_values('RSI (14)', ascending=True)
        st.dataframe(
            df_rsi,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "RSI (14)": st.column_config.NumberColumn("RSI Value", format="%.1f"),
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "ATH Dist %": st.column_config.NumberColumn("Disc. from ATH", format="%.1f%%"),
                "Price": st.column_config.NumberColumn(format="$%.4f"),
            },
            use_container_width=True
        )

    with tab_vol:
        st.subheader("Follow the Money (Whale Tracking)")
        df_vol = df.sort_values('Money In', ascending=False)
        st.dataframe(
            df_vol,
            column_config={
                "Icon": st.column_config.ImageColumn(),
                "Money In": st.column_config.NumberColumn(format="$%d"),
                "Money Out": st.column_config.NumberColumn(format="$%d"),
                "Vol/Cap": st.column_config.ProgressColumn("Money Flow", min_value=0, max_value=1.5),
                "Volume": st.column_config.NumberColumn("24h Vol", format="$%d"),
                "Mkt Cap": st.column_config.NumberColumn(format="$%d"),
            },
            use_container_width=True
        )

    # Selection handling - robust (use session state)
    sel = st.session_state.get("main_select")
    selected_row = None
    if sel and sel.get("selection") and sel["selection"].get("rows"):
        idx = sel["selection"]["rows"][0]
        if idx is not None and 0 <= idx < len(df):
            selected_row = df.iloc[idx]

    if selected_row is not None:
        row = selected_row
        st.divider()
        st.markdown(f"## 🔭 Pair Detail — {row['Ticker']}")
        c_chart, c_ai = st.columns([2, 1])

        with c_chart:
            prices = row['ChartData'] if isinstance(row['ChartData'], list) else []
            if prices and len(prices) > 0:
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
            else:
                st.info("No chart data available for this pair.")

            # TradingView mini widget embed (detail panel) - uses TradingView's mini symbol overview if allowed
            st.markdown("**Mini Market Overview (TradingView)**")
            try:
                symbol_for_tv = f"BINANCE:{row['Ticker'].replace('-USDT','').replace('/USDT','')}"  # best-effort mapping
                tv_iframe = f"""
                <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{row['Ticker']}&symbol={symbol_for_tv}&interval=60&cols=1&rows=1&theme=dark" style="width:100%;height:220px;"></iframe>
                """
                st.components.v1.html(tv_iframe, height=240, scrolling=False)
            except:
                st.write("TradingView widget unavailable.")

        with c_ai:
            st.markdown('<div class="titan-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Strategist")
            if st.button("GENERATE STRATEGY", key=f"gen_ai_{row['Ticker']}"):
                with st.spinner("Generating strategy (AI)..."):
                    strategy = get_titan_strategy(row, fg_val)
                    st.markdown(strategy)
            else:
                st.info("Press GENERATE STRATEGY to ask Gemini for a short analysis (one-time).")
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("RSI", f"{row['RSI (14)']:.0f}", "EXTREME" if row['RSI (14)'] < 30 or row['RSI (14)'] > 70 else "Neutral")
            c2.metric("Signal", row['Signal'])
            st.markdown('</div>', unsafe_allow_html=True)

        # Mini chat (local)
        st.markdown("### 💬 Pair Chat (local)")
        chat_key = f"chat_{row['Ticker']}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
        for msg in st.session_state[chat_key]:
            sender = msg.get("sender", "user")
            text = msg.get("text", "")
            cls = "chat-me" if sender == "me" else "chat-others"
            st.markdown(f"<div class='chat-bubble {cls}'>{text}</div>", unsafe_allow_html=True)
        chat_in = st.text_input("Message to local chat (press Enter to send)", key=f"chat_input_{row['Ticker']}")
        if chat_in:
            st.session_state[chat_key].append({"sender": "me", "text": chat_in})
            # auto-bot acknowledgement
            st.session_state[chat_key].append({"sender": "bot", "text": "Noted (local)."})

    else:
        st.info("Select a row in the MAIN BOARD to view Pair Detail, AI-strat, mini chart, and pair-chat.")

# End of file
