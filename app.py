import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model import Kronos, KronosTokenizer, KronosPredictor

st.set_page_config(page_title="Kronos AI Market Predictor", layout="wide")

# ==================== PASSWORD PROTECTION ====================
# Replace "YourSecretPassword123" below with your desired password
PASSWORD = "Tradingawais123"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Private Access Required")
    user_input = st.text_input("Enter Dashboard Password", type="password")
    if st.button("Login"):
        if user_input == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password!")
    st.stop()
# ==============================================================

st.title("📈 Kronos AI Financial Forecasting Dashboard")
st.write("Select your market asset, timeframe, and visualization options below.")

# Sidebar Settings
st.sidebar.header("Configuration")

CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", 
    "TRX-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "BCH-USD", "NEAR-USD", 
    "UNI-USD", "LEO-USD", "ICP-USD", "DAI-USD", "ETC-USD", "APT-USD", "STX-USD", "FIL-USD", 
    "XLM-USD", "CRO-USD", "ATOM-USD", "RNDR-USD", "HBAR-USD", "ARB-USD", "MKR-USD", "GRT-USD", 
    "VET-USD", "INJ-USD", "OP-USD", "THETA-USD", "FTM-USD", "ALGO-USD", "SUI-USD", "AAVE-USD", 
    "FLOW-USD", "TIA-USD", "PEPE-USD", "FET-USD", "RUNE-USD", "GALA-USD", "EGLD-USD", "STRK-USD", 
    "KAS-USD", "ORDI-USD", "SEI-USD", "FLOKI-USD"
]

FOREX_LIST = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "EURCAD=X", "GBPAUD=X",
    "GBPCAD=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "EURNZD=X", "GBPNZD=X"
]

asset_mode = st.sidebar.radio("Asset Selection Mode", ["Preset Lists", "Custom Symbol"])

if asset_mode == "Preset Lists":
    asset_type = st.sidebar.radio("Asset Type", ["Top Crypto (50+)", "Top Forex (20)"])
    if asset_type == "Top Crypto (50+)":
        ticker = st.sidebar.selectbox("Select Crypto Pair", CRYPTO_LIST)
    else:
        ticker = st.sidebar.selectbox("Select Forex Pair", FOREX_LIST)
else:
    ticker = st.sidebar.text_input("Enter Any Yahoo Ticker (e.g., SOL-USD, GBPJPY=X)", value="BTC-USD")

tf_options = {
    "1m": {"period": "7d", "fetch_interval": "1m", "resample": None, "freq": "1min"},
    "5m": {"period": "7d", "fetch_interval": "5m", "resample": None, "freq": "5min"},
    "10m": {"period": "7d", "fetch_interval": "5m", "resample": "10min", "freq": "10min"},
    "15m": {"period": "30d", "fetch_interval": "15m", "resample": None, "freq": "15min"},
    "30m": {"period": "30d", "fetch_interval": "30m", "resample": None, "freq": "30min"},
    "1h": {"period": "60d", "fetch_interval": "1h", "resample": None, "freq": "1h"},
    "2h": {"period": "60d", "fetch_interval": "1h", "resample": "2h", "freq": "2h"},
    "3h": {"period": "60d", "fetch_interval": "1h", "resample": "3h", "freq": "3h"},
    "4h": {"period": "60d", "fetch_interval": "1h", "resample": "4h", "freq": "4h"},
    "1D": {"period": "2y", "fetch_interval": "1d", "resample": None, "freq": "1D"},
    "1W": {"period": "5y", "fetch_interval": "1wk", "resample": None, "freq": "1W"}
}

timeframe = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=4)
cfg = tf_options[timeframe]
pred_len = st.sidebar.slider("Prediction Horizon (Candles)", min_value=12, max_value=48, value=24)

@st.cache_resource
def load_kronos():
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model=model, tokenizer=tokenizer, max_context=512)

predictor = load_kronos()

def process_and_resample(df, resample_rule):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    if resample_rule:
        df = df.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
    else:
        df = df.reset_index()

    time_col_new = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={
        time_col_new: 'timestamps',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df['amount'] = df['volume'] * df['close']
    return df

if st.button("🚀 Run Forecast"):
    with st.spinner(f"Fetching latest data and calculating predictions for {ticker}..."):
        raw_df = yf.download(ticker, period=cfg["period"], interval=cfg["fetch_interval"]).reset_index()

        if raw_df.empty:
            st.error(f"Could not download data for ticker '{ticker}'. Please verify the symbol and try again.")
        else:
            df = process_and_resample(raw_df, cfg["resample"])

            lookback = 400
            if len(df) > lookback:
                df_recent = df.tail(lookback).reset_index(drop=True)
            else:
                df_recent = df.reset_index(drop=True)

            x_df = df_recent[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = pd.Series(pd.to_datetime(df_recent['timestamps']))

            last_ts = x_timestamp.iloc[-1]
            y_timestamp = pd.Series(pd.date_range(start=last_ts, periods=pred_len + 1, freq=cfg["freq"])[1:])

            forecast_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=5
            )

            forecast_df['timestamps'] = y_timestamp.values

            st.success("Forecast Generated!")

            col1, col2 = st.columns([2.5, 1])

            with col1:
                st.subheader(f"TradingView Style Candlestick Chart ({ticker} - {timeframe})")
                hist_tail = df_recent.tail(40)[['timestamps', 'open', 'high', 'low', 'close']]
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist_tail['timestamps'], open=hist_tail['open'], high=hist_tail['high'], low=hist_tail['low'], close=hist_tail['close'],
                    name="Historical", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ))
                fig.add_trace(go.Candlestick(
                    x=forecast_df['timestamps'], open=forecast_df['open'], high=forecast_df['high'], low=forecast_df['low'], close=forecast_df['close'],
                    name="Kronos AI Forecast", increasing_line_color='#00e676', decreasing_line_color='#ff1744'
                ))
                fig.update_layout(
                    template="plotly_dark", xaxis_rangeslider_visible=False, height=500,
                    margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Timestamp", yaxis_title="Price"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Predicted Candlesticks")
                st.dataframe(forecast_df[['timestamps', 'open', 'high', 'low', 'close']])