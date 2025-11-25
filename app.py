# app.py — ExaHash Premium Streamlit App (Bybit Edition) with Gemini-2.5 Summarizer
import os
import time
from datetime import datetime
from math import isnan

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Try import Google generative ai SDK (Gemini)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Use environment variables or Streamlit secrets (recommended)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")

    # Google / Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Put your Google API key here in Render env

    # Thresholds
    RETAIL_LIMIT_USD = 1000.0
    WHALE_LIMIT_USD = 50000.0

    DEFAULT_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

    MAX_TRADES = 1000

# Configure Gemini if available
if GENAI_AVAILABLE and Config.GOOGLE_API_KEY:
    genai.configure(api_key=Config.GOOGLE_API_KEY)

# ==========================================
# 2. INIT EXCHANGE (ccxt)
# ==========================================
@st.cache_resource
def init_bybit():
    """Inisialisasi koneksi ke Bybit (Unified/V5)"""
    try:
        params = {'enableRateLimit': True, 'options': {'defaultType': 'swap'}}
        if Config.BYBIT_API_KEY and Config.BYBIT_SECRET_KEY:
            params.update({'apiKey': Config.BYBIT_API_KEY, 'secret': Config.BYBIT_SECRET_KEY})
        exchange = ccxt.bybit(params)
        return exchange
    except Exception as e:
        print(f"Bybit Init Error: {e}")
        return None

@st.cache_resource
def init_binance():
    """Inisialisasi koneksi ke Binance Futures"""
    try:
        params = {'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
            params.update({'apiKey': Config.BINANCE_API_KEY, 'secret': Config.BINANCE_SECRET_KEY})
        # try USDM first
        try:
            return ccxt.binanceusdm(params)
        except Exception:
            return ccxt.binance(params)
    except Exception as e:
        print(f"Binance Init Error: {e}")
        return None

# Initialize both exchanges
binance_exchange = init_binance()
bybit_exchange = init_bybit()

# Aggregator dict
exchanges = {}
if binance_exchange:
    exchanges['binance'] = binance_exchange
if bybit_exchange:
    exchanges['bybit'] = bybit_exchange

# Helper: fetch symbol data from all exchanges (best-effort)
def fetch_symbol_across_exchanges(symbol):
    result = {}
    for name, ex in exchanges.items():
        ob, trades, ohlcv_df = None, [], pd.DataFrame()
        try:
            ob = ex.fetch_order_book(symbol, limit=50)
        except Exception:
            ob = None

        try:
            trades = ex.fetch_trades(symbol, limit=Config.MAX_TRADES)
        except Exception:
            trades = []

        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe='1m', limit=200)
            if ohlcv:
                ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                cols = ['open','high','low','close','volume']
                ohlcv_df[cols] = ohlcv_df[cols].apply(pd.to_numeric, errors='coerce')
        except Exception:
            ohlcv_df = pd.DataFrame()

        result[name] = {
            'orderbook': ob,
            'trades': trades,
            'ohlcv': ohlcv_df
        }
    return result

# Cross-exchange spread helper
def compute_spreads(symbol_data):
    prices = {}
    for ex, d in symbol_data.items():
        price = None
        if d['trades']:
            try:
                last_trade = d['trades'][-1]
                price = float(last_trade.get('price', last_trade.get('price', None)))
            except Exception:
                price = None
        if price is None and not d['ohlcv'].empty:
            try:
                price = float(d['ohlcv']['close'].iloc[-1])
            except Exception:
                price = None
        prices[ex] = price

    spreads = {}
    exs = list(prices.keys())
    for i in range(len(exs)):
        for j in range(i+1, len(exs)):
            a, b = exs[i], exs[j]
            pa, pb = prices.get(a), prices.get(b)
            if pa and pb and (pa + pb) > 0:
                spread = abs(pa - pb) / ((pa + pb) / 2) * 100
            else:
                spread = None
            spreads[f"{a}_vs_{b}"] = spread
    return prices, spreads

# ==========================================
# 3. UTIL FUNCTIONS & INDICATORS
# ==========================================
def trades_to_df(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        except Exception:
            pass
    if 'cost' in df.columns and not df['cost'].isnull().all():
        df['value'] = df['cost']
    elif 'price' in df.columns and 'amount' in df.columns:
        df['value'] = df['price'] * df['amount']
    else:
        df['value'] = 0.0
    if 'side' not in df.columns:
        df['side'] = 'unknown'
    return df

def vwap_from_trades(df):
    if df.empty or 'value' not in df.columns or 'amount' not in df.columns:
        return np.nan
    total_vol = df['amount'].sum()
    total_val = df['value'].sum()
    if total_vol == 0:
        return np.nan
    return total_val / total_vol

def atr(df, period=14):
    if df.empty or len(df) < period + 1:
        return np.nan
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean().iloc[-1]
    return atr_val

def rsi(df, period=14):
    if df.empty or len(df) < period + 1:
        return np.nan
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.iloc[-1]

# ==========================================
# 4. MARKET INTELLIGENCE (per symbol)
# ==========================================
@st.cache_data(ttl=10) # Cache 10 detik
def get_market_intelligence(symbol):
    symbol_data = fetch_symbol_across_exchanges(symbol)
    prices, spreads = compute_spreads(symbol_data)

    all_trades = []
    for ex_name, d in symbol_data.items():
        trades = d['trades'] or []
        for t in trades:
            t_copy = dict(t)
            t_copy['source'] = ex_name
            if 'info' in t_copy: del t_copy['info']
            all_trades.append(t_copy)
    trades_df = trades_to_df(all_trades)

    ob_imbalances = {}
    combined_bid_vol = 0
    combined_ask_vol = 0

    for ex_name, d in symbol_data.items():
        ob = d['orderbook']
        imbalance = np.nan
        if ob and 'bids' in ob and 'asks' in ob and len(ob['bids'])>0 and len(ob['asks'])>0:
            bid_vol = sum([p*a for p,a in ob['bids'][:20]])
            ask_vol = sum([p*a for p,a in ob['asks'][:20]])
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = ((bid_vol - ask_vol) / total) * 100
            combined_bid_vol += bid_vol
            combined_ask_vol += ask_vol
        ob_imbalances[ex_name] = imbalance

    combined_total = combined_bid_vol + combined_ask_vol
    combined_ob_imbalance = ((combined_bid_vol - combined_ask_vol) / combined_total) * 100 if combined_total > 0 else 0

    retail_fomo = 50.0
    whale_net_flow = 0.0
    if not trades_df.empty:
        retail_df = trades_df[trades_df['value'] <= Config.RETAIL_LIMIT_USD]
        if not retail_df.empty:
            retail_buy = retail_df[retail_df['side']=='buy']['value'].sum()
            retail_total = retail_df['value'].sum()
            if retail_total > 0:
                retail_fomo = (retail_buy / retail_total * 100)
        whale_df = trades_df[trades_df['value'] >= Config.WHALE_LIMIT_USD]
        if not whale_df.empty:
            whale_buy = whale_df[whale_df['side']=='buy']['value'].sum()
            whale_sell = whale_df[whale_df['side']=='sell']['value'].sum()
            whale_net_flow = whale_buy - whale_sell

    vwap = vwap_from_trades(trades_df)
    available_prices = [p for p in prices.values() if p is not None]
    agg_price = sum(available_prices)/len(available_prices) if available_prices else np.nan

    per_ex_stats = {}
    for ex_name, d in symbol_data.items():
        ohlcv_df = d['ohlcv']
        vol = np.nan; atr_val = np.nan; rsi_val = np.nan
        if not ohlcv_df.empty:
            try:
                ret = ohlcv_df['close'].pct_change().dropna()
                if len(ret) > 1:
                    vol = ret.std() * np.sqrt(365 * 24 * 60)
                atr_val = atr(ohlcv_df)
                rsi_val = rsi(ohlcv_df)
            except Exception:
                pass
        per_ex_stats[ex_name] = {
            'price': prices.get(ex_name),
            'volatility': vol,
            'atr': atr_val,
            'rsi': rsi_val,
            'ob_imbalance': ob_imbalances.get(ex_name),
        }

    components = []
    binance_vol = per_ex_stats.get('binance', {}).get('volatility', np.nan)
    if not isnan(binance_vol):
        components.append(min(binance_vol * 100, 100))
    binance_rsi = per_ex_stats.get('binance', {}).get('rsi', np.nan)
    if not isnan(binance_rsi):
        components.append(abs(binance_rsi - 50) * 2)
    if combined_ob_imbalance is not None and not isnan(combined_ob_imbalance):
        components.append(min(abs(combined_ob_imbalance), 100))

    risk_score = np.mean(components) if components else 0

    primary_ex = 'binance' if prices.get('binance') is not None else ('bybit' if prices.get('bybit') is not None else None)
    primary_orderbook = symbol_data.get(primary_ex, {}).get('orderbook') if primary_ex else None
    primary_ohlcv = symbol_data.get(primary_ex, {}).get('ohlcv') if primary_ex else pd.DataFrame()

    return {
        'symbol': symbol,
        'price': agg_price,
        'per_exchange': per_ex_stats,
        'spreads': spreads,
        'ob_imbalance': combined_ob_imbalance,
        'retail_fomo': retail_fomo,
        'whale_flow': whale_net_flow,
        'vwap': vwap,
        'trades_df': trades_df,
        'risk_score': risk_score,
        'timestamp': datetime.utcnow(),
        'orderbook': primary_orderbook,
        'ohlcv': primary_ohlcv,
        'primary_ex': primary_ex
    }

# ==========================================
# 5. STREAMLIT UI / LAYOUT + Styling (White background + Coinbase-like blue buttons)
# ==========================================
st.set_page_config(page_title='ExaHash AI Analytic', layout='wide', page_icon='🧠')

st.markdown("""
<style>
  /* White background */
  .stApp { background: #ffffff; color: #0b1726; }
  /* Coinbase-like blue button */
  .stButton>button { background-color: #1652f0 !important; color: white !important; border-radius: 8px; border: none; font-weight: 600; padding: 8px 12px; }
  /* Navbar card */
  .exahash-navbar { display:flex; align-items:center; justify-content:space-between; padding:15px 20px; border-radius:12px; background: white; border: 1px solid #e6eefc; margin-bottom: 20px; box-shadow: 0 6px 18px -8px rgba(22,82,240,0.12); } 
  .brand { display:flex; align-items:center; gap:12px; }
  .brand .logo { width:44px; height:44px; border-radius:10px; background: linear-gradient(135deg,#1652f0,#6f9bff); display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size: 18px; }
  .small-muted { font-size:12px; color:#64748b; }
  /* Card style for panels */
  .card { background: #ffffff; border: 1px solid #eef3ff; padding: 12px; border-radius: 10px; box-shadow: 0 4px 12px -8px rgba(16,24,40,0.06); }
  /* Make metrics bold */
  .metric-label { font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="exahash-navbar">
  <div class="brand">
    <div class="logo">EH</div>
    <div>
      <div style="font-weight:800; font-size: 18px;">ExaHash AI</div>
      <div class="small-muted">Crypto Intelligence • Binance & Bybit Data</div>
    </div>
  </div>
  <div>
    <span class="small-muted">Live Market Data</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title('🧭 ExaHash AI')
    watchlist_input = st.text_area('Watchlist', value=','.join(Config.DEFAULT_WATCHLIST), height=120)
    watchlist = [s.strip().upper() for s in watchlist_input.split(',') if s.strip()]
    refresh_seconds = st.number_input('Auto Refresh (s)', min_value=5, max_value=600, value=30, step=5)
    st.markdown('---')
    st.subheader('🔎 Search Symbol')
    query = st.text_input('Example: BTC, ETH/USDT')
    if st.button('Search') and query:
        q = query.strip().upper()
        if '/' not in q:
            q = q + '/USDT'
        st.experimental_set_query_params(page='detail', symbol=q)
        st.experimental_rerun()
    st.markdown('---')
    st.subheader('⚙️ Controls')
    if st.button('Go to Dashboard'):
        st.experimental_set_query_params(page='dashboard')
        st.experimental_rerun()

# --- ROUTING LOGIC ---
qp = st.experimental_get_query_params()
page = qp.get('page', ['dashboard'])[0]
current_symbol = qp.get('symbol', [None])[0]

# Auto-refresh handling
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()
now = time.time()
if now - st.session_state['last_refresh'] > refresh_seconds:
    st.session_state['last_refresh'] = now
    st.experimental_rerun()

if st.button('Refresh Data Now', key='top_refresh'):
    st.session_state['last_refresh'] = now
    st.experimental_rerun()

# ==========================================
# Helper: Build dashboard text for AI summarization
# ==========================================
def build_dashboard_text(all_data, watchlist):
    lines = []
    lines.append(f"Snapshot time: {datetime.utcnow().isoformat()} UTC")
    lines.append("Watchlist: " + ", ".join(watchlist))
    for s, d in all_data.items():
        if not d:
            lines.append(f"{s}: no data")
            continue
        lines.append(f"Symbol: {s}")
        lines.append(f"  Agg Price: {d.get('price')}")
        lines.append(f"  Risk Score: {d.get('risk_score'):.2f}")
        lines.append(f"  Whale Net Flow: {d.get('whale_flow')}")
        lines.append(f"  Retail FOMO: {d.get('retail_fomo'):.2f}%")
        lines.append(f"  OB Imbalance: {d.get('ob_imbalance')}")
        lines.append(f"  Spreads: {d.get('spreads')}")
        # small sample of recent trades
        trades = d.get('trades_df')
        if trades is not None and not trades.empty:
            top_trades = trades.sort_values('timestamp', ascending=False).head(3)
            for _, row in top_trades.iterrows():
                ttime = row['timestamp'] if 'timestamp' in row else ''
                lines.append(f"    trade: {ttime} side={row.get('side')} price={row.get('price')} amount={row.get('amount')} value={row.get('value')}")
    return "\n".join(lines)

# ==========================================
# AI Summarizer (Gemini 2.5) — cached
# ==========================================
@st.cache_data(ttl=30)
def ai_summarize(prompt_text, max_tokens=512):
    if not GENAI_AVAILABLE or not Config.GOOGLE_API_KEY:
        return "Gemini SDK not available or GOOGLE_API_KEY not set in environment."
    try:
        # Use text generation API for Gemini 2.5
        # NOTE: API may return different structure depending on google.generativeai version.
        response = genai.generate_text(model="gemini-2.5", prompt=prompt_text, max_output_tokens=max_tokens)
        # Try common response shapes
        if hasattr(response, 'text') and response.text:
            return response.text
        if isinstance(response, dict):
            # try "candidates"
            cands = response.get('candidates')
            if cands:
                return cands[0].get('content', cands[0].get('text', ''))
            return str(response)
        return str(response)
    except Exception as e:
        return f"AI Error: {e}"

# ==========================================
# --- PAGE: DETAIL ---
# ==========================================
if page == 'detail' and current_symbol:
    st.markdown(f"## 🔍 Detail Analysis: {current_symbol}")
    with st.spinner(f'Mengambil data dari exchanges untuk {current_symbol}...'):
        detail_data = get_market_intelligence(current_symbol)
    if not detail_data or isnan(detail_data['price']):
        st.error(f'Data tidak ditemukan untuk {current_symbol}. Coba simbol lain atau periksa koneksi.')
        if st.button('Kembali ke Dashboard'):
            st.experimental_set_query_params(page='dashboard'); st.experimental_rerun()
    else:
        st.subheader('Overview')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Agg. Price', f"${detail_data['price']:.2f}" if not isnan(detail_data['price']) else "N/A")
        col2.metric('Risk Score (0-100)', f"{detail_data['risk_score']:.1f}")
        col3.metric('Whale Net Flow', f"${detail_data['whale_flow']:,.0f}", delta_color="normal")
        col4.metric('Retail FOMO', f"{detail_data['retail_fomo']:.1f}%")

        st.subheader('Exchange Comparison')
        per_ex_rows = []
        for ex_name, ex_stats in detail_data['per_exchange'].items():
            per_ex_rows.append({
                'Exchange': ex_name.capitalize(),
                'Price': f"${ex_stats.get('price', 0):.2f}" if ex_stats.get('price') else "N/A",
                'RSI': f"{ex_stats.get('rsi', 0):.1f}" if ex_stats.get('rsi') else "N/A",
                'Volatility': f"{ex_stats.get('volatility', 0):.4f}" if ex_stats.get('volatility') else "N/A",
                'OB Imbalance': f"{ex_stats.get('ob_imbalance', 0):.2f}%" if ex_stats.get('ob_imbalance') else "N/A"
            })
        st.table(pd.DataFrame(per_ex_rows))

        st.subheader(f"Price Chart ({detail_data.get('primary_ex', 'Unknown').capitalize()})")
        ohlcv = detail_data.get('ohlcv', pd.DataFrame())
        if not ohlcv.empty:
            fig = go.Figure(data=[go.Candlestick(x=ohlcv['timestamp'], open=ohlcv['open'], high=ohlcv['high'], low=ohlcv['low'], close=ohlcv['close'])])
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        trades_df = detail_data.get('trades_df', pd.DataFrame())
        if not trades_df.empty:
            st.subheader('Recent Large Trades (> $10k)')
            large_trades = trades_df[trades_df['value'] > 10000].copy()
            if not large_trades.empty:
                large_trades['time'] = large_trades['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(large_trades[['time', 'source', 'side', 'price', 'amount', 'value']].sort_values('timestamp', ascending=False).head(20), use_container_width=True)
            else:
                st.info("No recent large trades found.")
    if st.button('⬅ Back to Dashboard'):
        st.experimental_set_query_params(page='dashboard'); st.experimental_rerun()

# --- PAGE: DASHBOARD ---
elif page == 'dashboard':
    cols = st.columns([2, 3])
    left_col = cols[0]; right_col = cols[1]

    # Fetch Data Loop
    all_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, s in enumerate(watchlist):
        status_text.text(f"Scanning {s} on exchanges...")
        try:
            all_data[s] = get_market_intelligence(s)
        except Exception:
            all_data[s] = None
        progress_bar.progress((i + 1) / max(1, len(watchlist)))
    status_text.empty(); progress_bar.empty()

    # Left: Whale Leaderboard
    with left_col:
        st.header('📈 Whale Leaderboard')
        leaderboard_data = []
        for s, d in all_data.items():
            if d:
                leaderboard_data.append({
                    'Symbol': s,
                    'Whale Flow ($)': d['whale_flow'],
                    'Retail FOMO (%)': d['retail_fomo'],
                    'Risk': d['risk_score']
                })
        ld_df = pd.DataFrame(leaderboard_data)
        if not ld_df.empty:
            st.dataframe(
                ld_df.style.format({'Whale Flow ($)': "{:,.0f}", 'Retail FOMO (%)': "{:.1f}", 'Risk': "{:.1f}"})
                .background_gradient(subset=['Whale Flow ($)'], cmap='RdYlGn'),
                use_container_width=True
            )
            st.caption("Tip: Positive Whale Flow indicates accumulation.")
        else:
            st.warning("No data available.")

        # AI Summary block (whole-screen)
        st.markdown("---")
        st.subheader("🤖 AI Market Summary (Gemini 2.5)")
        if not GENAI_AVAILABLE:
            st.info("Gemini SDK not available or GOOGLE_API_KEY missing. Add GOOGLE_API_KEY to environment to enable AI.")
        else:
            if st.button("Generate AI Summary of Entire Screen"):
                with st.spinner("Generating AI summary from Gemini 2.5..."):
                    dashboard_text = build_dashboard_text(all_data, watchlist)
                    prompt = (
                        "You are an expert crypto market analyst. Summarize the following dashboard snapshot into:\n"
                        "1) One-line headline (market direction & highest-risk symbol)\n"
                        "2) 3-sentence summary with key observations (whale flow, retail sentiment, OB imbalance)\n"
                        "3) Bullet list of 3 actionable points (what to watch / possible entry/exit signals)\n\n"
                        "Dashboard snapshot:\n\n" + dashboard_text
                    )
                    ai_output = ai_summarize(prompt, max_tokens=512)
                st.success("AI Summary (Gemini 2.5)")
                st.markdown(ai_output)

    # Right: Market Correlation + Charts
    with right_col:
        st.header('🔬 Market Correlation')
        price_dict = {}
        for s, d in all_data.items():
            if d and d.get('ohlcv') is not None and not d['ohlcv'].empty:
                price_dict[s] = d['ohlcv'].set_index('timestamp')['close']
        if price_dict:
            prices_df = pd.concat(price_dict, axis=1).dropna()
            if not prices_df.empty:
                corr = prices_df.pct_change().corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix (returns)")
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough OHLCV data to compute correlation.")

        st.markdown('---')
        st.header('🔎 Quick Symbol Actions')
        selected = st.selectbox("Open detail for symbol:", options=watchlist)
        if st.button("Open Detail"):
            st.experimental_set_query_params(page='detail', symbol=selected)
            st.experimental_rerun()

# End of app
