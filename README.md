# Exahash AI — Streamlit + Multi Exchange

Deployment guide:

1. Fork / clone repo
2. Upload ke GitHub
3. Buka https://render.com
4. New → Web Service
5. Build Command:
    pip install -r requirements.txt
6. Start Command:
    streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
7. Tambahkan Environment Variables:
    BINANCE_API_KEY
    BINANCE_SECRET_KEY
    BYBIT_API_KEY
    BYBIT_SECRET_KEY

Done — app langsung live di Render.
