# ==========================================================
# Cross-Asset Market Regime Monitor
# MSc in Financial Markets and Investments - Skema Business School
# Python Programming for Finance
# Academic Year 2024/25
#
# This section imports all the necessary libraries for:
# - Data collection and processing
# - Visualization
# - Dashboard deployment using Streamlit
# ==========================================================

# --- Data handling ---
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web


# --- Visualization libraries ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --- Dashboard framework ---
import streamlit as st

# --- Utilities ---
from datetime import datetime, timedelta
import requests


# ==========================================================
# Utility Functions
#
# This section defines reusable functions that will be called
# throughout the dashboard project. These functions handle:
# - Data download from Yahoo Finance
# - Data preprocessing
# - Performance metrics calculation
# - Visualization helpers
# ==========================================================


# --- Function to download price data from Yahoo Finance ---
def get_price_data(ticker, start="1800-01-01", end=datetime.today().strftime("%Y-%m-%d"), interval="1d"):
    """
    Download historical price data for a given ticker from Yahoo Finance.
    Returns a clean DataFrame with one column named after the ticker.
    Handles both 'Adj Close' and 'Close'.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        print(f"[WARNING] No data for {ticker}")
        return pd.DataFrame()

    if "Adj Close" in df.columns:
        series = df["Adj Close"]
    elif "Close" in df.columns:
        series = df["Close"]
    else:
        print(f"[WARNING] No 'Adj Close' or 'Close' for {ticker}")
        return pd.DataFrame()

    # Always return as DataFrame with ticker as column name
    if isinstance(series, pd.Series):
        return series.to_frame(name=ticker)
    else:
        return series.rename(columns={series.columns[0]: ticker})

    
def get_multi_price_data(tickers, start="1800-01-01", end=datetime.today().strftime("%Y-%m-%d"), interval="1d"):
    """
    Download historical price data for multiple tickers from Yahoo Finance.
    Returns a clean DataFrame with tickers as columns.
    Automatically handles both 'Adj Close' and 'Close'.
    """
    

    
    data = pd.DataFrame()

    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, interval=interval)

        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        else:
            print(f"[WARNING] No usable column for {ticker}. Skipped.")
            continue

        # Ensure output is always a DataFrame with correct name
        if isinstance(series, pd.Series):
            series = series.to_frame(name=ticker)
        elif isinstance(series, pd.DataFrame):
            series = series.rename(columns={series.columns[0]: ticker})

        data = pd.concat([data, series], axis=1)

    return data

# --- Function to check and fill missing values ---
def check_and_fill_missing(data):
    """
    Check if the DataFrame has missing values.
    If missing values exist, they are replaced with the next available value (forward in time).
    Returns a cleaned DataFrame.
    """
    if data.isnull().sum().sum() > 0:
        print(f"[INFO] Missing values detected: {data.isnull().sum().sum()} filled with next available values.")
        data = data.bfill()  # replace missing values with the next available (backward fill)
    else:
        print("[INFO] No missing values detected.")
    return data


# --- Data Collection for All Assets ---
def load_all_assets(asset_dict, start="1900-01-01", end=datetime.today().strftime("%Y-%m-%d"), interval="1d"):
    """
    Download and clean price data for all assets listed in the asset_dict.
    Returns a dictionary of DataFrames, one per asset class.
    """
    all_data = {}

    for category, tickers in asset_dict.items():
        print(f"[INFO] Downloading {category} data...")
        df = yf.download(list(tickers.values()), start=start, end=end, interval=interval)["Adj Close"]

        # Ensure DataFrame format even for single tickers
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # Clean missing values
        df = check_and_fill_missing(df)

        # Store cleaned DataFrame in dictionary
        all_data[category] = df

    print("[INFO] All data downloaded and cleaned successfully.")
    return all_data




# ==========================================================
# Data Preprocessing & Metrics
#
# This section defines functions to transform raw market data
# into meaningful indicators and metrics for visualization.
# Includes:
# - Returns (daily, cumulative)
# - Normalization (base 100)
# - Volatility (rolling)
# - Drawdown
# ==========================================================


# --- Function to calculate daily returns ---
def calculate_returns(price_df):
    """
    Calculate daily percentage returns from a price DataFrame.
    """
    returns = price_df.pct_change().dropna()
    return returns


# --- Function to calculate cumulative performance ---
def calculate_cumulative_returns(price_df):
    """
    Calculate cumulative returns over time from a price DataFrame.
    """
    returns = price_df.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return cumulative


# --- Function to normalize data (base 100) ---
def normalize_data(price_df):
    """
    Normalize prices to start at 100 for comparison across assets.
    """
    normalized = price_df / price_df.iloc[0] * 100
    return normalized


# --- Function to calculate rolling volatility ---
def calculate_volatility(price_df, window=30):
    """
    Calculate rolling volatility over a specified window (default: 30 days).
    Annualized using sqrt(252).
    """
    returns = price_df.pct_change().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility


# --- Function to calculate drawdown ---
def calculate_drawdown(price_df):
    """
    Calculate drawdown and maximum drawdown from a price DataFrame.
    Returns two DataFrames:
    - drawdown series
    - max drawdown value
    """
    cum_returns = calculate_cumulative_returns(price_df)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

# --- Function to plot correlation between two assets over a fixed period ---
def plot_correlation_period(price_df, asset_x, asset_y, start_date, end_date, title=None):
    """
    Plot correlation between two assets on a selected period using scatter + regression line.
    """
    returns = calculate_returns(price_df)

    # Filtrer la période
    period_data = returns.loc[start_date:end_date, [asset_x, asset_y]].dropna()

    # Corrélation
    corr_value = period_data[asset_x].corr(period_data[asset_y])

    if title is None:
        title = f"Correlation between {asset_x} and {asset_y} ({start_date} to {end_date})"

    fig = px.scatter(
        period_data,
        x=asset_x,
        y=asset_y,
        trendline="ols",  # ajoute une droite de régression
        title=title,
        labels={asset_x: f"{asset_x} Returns", asset_y: f"{asset_y} Returns"}
    )

    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"Correlation = {corr_value:.2f}",
        showarrow=False,
        font=dict(size=14, color="red"),
        bgcolor="white"
    )

    return fig

# --- Function to calculate portfolio returns given weights ---
def calculate_portfolio_returns(price_df, weights):
    """
    Calculate portfolio cumulative returns given asset prices and weights.
    - weights: dict mapping tickers to portfolio weights (must sum to 1)
    """
    returns = calculate_returns(price_df)
    weighted_returns = (returns * pd.Series(weights)).sum(axis=1)
    portfolio_cum = (1 + weighted_returns).cumprod()
    return portfolio_cum, weighted_returns

# --- Breadth Indicator ---
def plot_breadth(window=200):
    """
    Breadth Indicator: % of S&P500 above 200-day MA (proxy: ^GSPC).
    """
    sp500 = get_price_data("^GSPC")
    if sp500 is None:
        st.warning("No data available for ^GSPC.")
        return px.line()

    ma = sp500["^GSPC"].rolling(window).mean()
    breadth = (sp500["^GSPC"] / ma - 1) * 100
    fig = px.line(breadth, title="Breadth Indicator (^GSPC vs 200d MA)")
    fig.update_layout(yaxis_title="% Above MA (proxy)", xaxis_title="Date")
    return fig


# --- Yield Curve ---
def plot_yield_curve():
    """
    Yield curve: 10Y (^TNX) - 3M (^IRX) spread.
    """
    t10y = get_price_data("^TNX")
    t3m = get_price_data("^IRX")  # proxy short term
    
    if t10y is None or t3m is None:
        st.warning("No data available for ^TNX or ^IRX.")
        return px.line()

    spread = (t10y["^TNX"]/100) - (t3m["^IRX"]/100)
    fig = px.line(spread, title="Yield Curve (10Y - 3M Spread)")
    fig.update_layout(yaxis_title="Spread (%)", xaxis_title="Date")
    return fig


# --- Credit Spread ---
def plot_credit_spread():
    """
    Credit spread proxy: High Yield (HYG) vs Investment Grade (LQD).
    """
    hyg = get_price_data("HYG")
    lqd = get_price_data("LQD")
    
    if hyg is None or lqd is None:
        st.warning("No data available for HYG or LQD.")
        return px.line()

    spread = hyg["HYG"].pct_change() - lqd["LQD"].pct_change()
    spread = spread.cumsum()

    fig = px.line(spread, title="Credit Spread (HYG - LQD, cumulative)")
    fig.update_layout(yaxis_title="Relative Perf.", xaxis_title="Date")
    return fig


# --- Volatility Regime ---
def plot_volatility_regime():
    """
    Volatility Regime: CBOE VIX Index.
    """
    vix = get_price_data("^VIX")
    if vix is None:
        st.warning("No data available for ^VIX.")
        return px.line()

    fig = px.line(vix, title="Volatility Regime (VIX Index)")
    fig.update_layout(yaxis_title="VIX Level", xaxis_title="Date")
    return fig


# --- USD Strength ---
def plot_usd_strength():
    """
    USD Strength: Dollar Index (DXY).
    """
    dxy = get_price_data("DX-Y.NYB")
    if dxy is None:
        st.warning("No data available for DXY (DX-Y.NYB).")
        return px.line()

    fig = px.line(dxy, title="USD Strength (DXY Index)")
    fig.update_layout(yaxis_title="Index Level", xaxis_title="Date")
    return fig


# --- Global Market Regime Score ---
def plot_market_regime_score():
    """
    Composite Market Regime Score = Breadth + Credit + Yield + Vol + USD.
    """
    sp500 = get_price_data("^GSPC")
    hyg = get_price_data("HYG")
    lqd = get_price_data("LQD")
    t10y = get_price_data("^TNX")
    t3m = get_price_data("^IRX")
    vix = get_price_data("^VIX")
    dxy = get_price_data("DX-Y.NYB")

    if any(x is None for x in [sp500, hyg, lqd, t10y, t3m, vix, dxy]):
        st.warning("Missing data for one or more regime indicators.")
        return px.line()

    spread_credit = hyg["HYG"].pct_change() - lqd["LQD"].pct_change()
    spread_yield = (t10y["^TNX"]/100) - (t3m["^IRX"]/100)

    norm_credit = (spread_credit - spread_credit.mean()) / spread_credit.std()
    norm_yield = (spread_yield - spread_yield.mean()) / spread_yield.std()
    norm_vix = (vix["^VIX"] - vix["^VIX"].mean()) / vix["^VIX"].std()
    norm_dxy = (dxy["DX-Y.NYB"] - dxy["DX-Y.NYB"].mean()) / dxy["DX-Y.NYB"].std()

    # Score global (Risk-On positif, Risk-Off négatif)
    score = -norm_credit + norm_yield - norm_vix - norm_dxy

    fig = px.line(score, title="Global Market Regime Score")
    fig.update_layout(yaxis_title="Composite Score", xaxis_title="Date")
    return fig


# ==========================================================
# Data Visualization
#
# This section defines reusable plotting functions for:
# - Time series (prices, cumulative returns, volatility)
# - Heatmaps (correlations, performance comparison)
# ==========================================================

# --- Function to plot normalized time series (base 100) ---
def plot_normalized_series(price_df, title="Normalized Performance (Base 100)"):
    """
    Plot normalized price series (starting at 100) using Plotly.
    """
    normalized = normalize_data(price_df)
    fig = px.line(normalized, title=title)
    fig.update_layout(yaxis_title="Index (Base 100)", xaxis_title="Date")
    return fig


# --- Function to plot cumulative returns ---
def plot_cumulative_returns(price_df, title="Cumulative Returns"):
    """
    Plot cumulative returns over time.
    """
    cum_returns = calculate_cumulative_returns(price_df)
    fig = px.line(cum_returns, title=title)
    fig.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date")
    return fig


# --- Function to plot rolling volatility ---
def plot_volatility(price_df, window=30, title="Rolling Volatility (30d)"):
    """
    Plot rolling volatility (annualized) over time.
    """
    vol = calculate_volatility(price_df, window=window)
    fig = px.line(vol, title=title)
    fig.update_layout(yaxis_title="Volatility (annualized)", xaxis_title="Date")
    return fig


# --- Function to plot correlation heatmap ---
def plot_correlation_heatmap(price_df, title="Correlation Heatmap"):
    """
    Plot correlation matrix of returns using Seaborn + Matplotlib.
    """
    returns = calculate_returns(price_df)
    corr_matrix = returns.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(title)
    return fig


# --- Breadth Indicator ---
def plot_breadth(window=200):
    """
    Breadth Indicator: % of S&P500 above 200-day MA (proxy: ^GSPC).
    """
    sp500 = get_price_data("^GSPC")
    if sp500 is None or sp500.empty:
        st.warning("No data available for ^GSPC.")
        return px.line()

    ma = sp500["^GSPC"].rolling(window).mean()
    breadth = (sp500["^GSPC"] / ma - 1) * 100

    fig = px.line(breadth, title="Breadth Indicator (^GSPC vs 200d MA)")
    fig.update_layout(yaxis_title="% Above MA (proxy)", xaxis_title="Date")
    return fig


# --- Yield Curve ---
# --- Yield Curves: US & German ---
# --- Yield Curve ---
def plot_us_yield_curve():
    """
    US yield curve: 10Y (^TNX) - 3M (^IRX).
    """
    t10y = get_price_data("^TNX")
    t3m = get_price_data("^IRX")  # 3M T-Bill proxy

    if t10y is None or t10y.empty or t3m is None or t3m.empty:
        st.warning("No data available for US yield curve (^TNX, ^IRX).")
        return px.line()

    spread = (t10y["^TNX"]/100) - (t3m["^IRX"]/100)
    fig = px.line(spread, title="US Yield Curve (10Y - 3M Spread)")
    fig.update_layout(yaxis_title="Spread (%)", xaxis_title="Date")
    return fig


def plot_german_proxy():
    """
    German Bund proxy using ETF IS0L.DE.
    """
    bund_proxy = get_price_data("IS0L.DE")

    if bund_proxy is None or bund_proxy.empty:
        st.warning("No data available for German Bund proxy (IS0L.DE).")
        return px.line()

    fig = px.line(bund_proxy, title="German Bund Proxy (IS0L.DE)")
    fig.update_layout(yaxis_title="Price Level", xaxis_title="Date")
    return fig

# --- Credit Spread ---
def plot_credit_spread():
    """
    Credit spread proxy: High Yield (HYG) vs Investment Grade (LQD).
    """
    hyg = get_price_data("HYG")
    lqd = get_price_data("LQD")

    if hyg is None or lqd is None or hyg.empty or lqd.empty:
        st.warning("No data available for HYG or LQD.")
        return px.line()

    df = pd.concat([hyg, lqd], axis=1).dropna()
    df.columns = ["HYG", "LQD"]

    hyg_ret = df["HYG"].pct_change().fillna(0)
    lqd_ret = df["LQD"].pct_change().fillna(0)
    spread = (hyg_ret - lqd_ret).cumsum()

    fig = px.line(spread, title="Credit Spread (HYG - LQD, cumulative)")
    fig.update_layout(yaxis_title="Relative Performance", xaxis_title="Date")
    return fig


# --- Volatility Regime ---
def plot_volatility_regime():
    """
    Volatility Regime: CBOE VIX Index.
    """
    vix = get_price_data("^VIX")
    if vix is None or vix.empty:
        st.warning("No data available for ^VIX.")
        return px.line()

    fig = px.line(vix, title="Volatility Regime (VIX Index)")
    fig.update_layout(yaxis_title="VIX Level", xaxis_title="Date")
    return fig


# --- USD Strength ---
def plot_usd_strength():
    """
    USD Strength: Dollar Index (DXY).
    """
    dxy = get_price_data("DX-Y.NYB")
    if dxy is None or dxy.empty:
        st.warning("No data available for DXY (DX-Y.NYB).")
        return px.line()

    fig = px.line(dxy, title="USD Strength (DXY Index)")
    fig.update_layout(yaxis_title="Index Level", xaxis_title="Date")
    return fig

def display_market_regime_score(show_chart=True):
    """
    Display the Global Market Regime Score in Streamlit.
    - Shows the latest score as a gauge + interpretation
    - Optionally plots a smoothed time series with Min/Max markers
    """

    # --- Charger les données nécessaires ---
    sp500 = get_price_data("^GSPC")
    hyg = get_price_data("HYG")
    lqd = get_price_data("LQD")
    t10y = get_price_data("^TNX")
    t3m = get_price_data("^IRX")
    vix = get_price_data("^VIX")
    dxy = get_price_data("DX-Y.NYB")

    if any(x is None or x.empty for x in [sp500, hyg, lqd, t10y, t3m, vix, dxy]):
        st.warning("Missing data for one or more regime indicators.")
        return

    # --- Calcul des composantes ---
    spread_credit = hyg["HYG"].pct_change() - lqd["LQD"].pct_change()
    spread_yield = (t10y["^TNX"]/100) - (t3m["^IRX"]/100)

    norm_credit = (spread_credit - spread_credit.mean()) / spread_credit.std()
    norm_yield = (spread_yield - spread_yield.mean()) / spread_yield.std()
    norm_vix = (vix["^VIX"] - vix["^VIX"].mean()) / vix["^VIX"].std()
    norm_dxy = (dxy["DX-Y.NYB"] - dxy["DX-Y.NYB"].mean()) / dxy["DX-Y.NYB"].std()

    # --- Score global ---
    score = -norm_credit + norm_yield - norm_vix - norm_dxy
    latest_score = score.iloc[-1]

    # --- Jauge numérique ---
    min_score = score.min()
    max_score = score.max()

    st.metric(
        label="📊 Global Market Regime Score",
        value=f"{latest_score:.2f}",
        delta=f"Min: {min_score:.2f} | Max: {max_score:.2f}"
    )

    # --- Interprétation ---
    if latest_score > 1:
        interpretation = "🟢 Risk-On (markets strong)"
    elif latest_score < -1:
        interpretation = "🔴 Risk-Off (markets stressed)"
    else:
        interpretation = "🟡 Neutral (mixed signals)"

    st.markdown(f"**Interpretation:** {interpretation}")

    # --- Graphique historique optionnel ---
    if show_chart:
        # Zoom sur période récente
        recent_score = score[score.index >= "2019-01-01"]
        smooth_score = recent_score.rolling(30).mean()

        min_score = smooth_score.min()
        max_score = smooth_score.max()

        fig = px.line(
            smooth_score,
            title="Global Market Regime Score (30d Smoothed, Since 2019)"
        )
        fig.update_layout(
            yaxis_title="Score (30d MA)",
            xaxis_title="Date",
            yaxis=dict(range=[min_score - 1, max_score + 1])
        )

        # Zones colorées
        fig.add_hrect(y0=-999, y1=-1, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=-1, y1=1, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=1, y1=999, fillcolor="green", opacity=0.1, line_width=0)

        # Annoter min et max
        min_date = smooth_score.idxmin()
        max_date = smooth_score.idxmax()
        fig.add_annotation(
            x=min_date, y=min_score - 0.2,
            text=f"Min: {min_score:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="red", ay=30,
            font=dict(color="red")
        )
        fig.add_annotation(
            x=max_date, y=max_score + 0.2,
            text=f"Max: {max_score:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="green", ay=-30,
            font=dict(color="green")
        )

        st.plotly_chart(fig, use_container_width=True, key="global_score_chart")


# ==========================================================
# Streamlit Dashboard Layout
# ==========================================================

# --- Page setup ---
st.set_page_config(
    page_title="All-in-One Asset Dashboard",  
    layout="wide"
)

# --- Main title ---
st.title("All-in-One Asset Dashboard")  
st.markdown("Welcome to your dashboard. Its goal is to help you deal with daily market information.")  


# --- Sidebar (filters) ---
st.sidebar.header("Filters")  

# Multiselect instead of selectbox (choix multiples)
asset_classes = st.sidebar.multiselect(
    "Select Asset Classes", 
    ["Equities", "Commodities", "Currencies", "Bonds", "Gold", "Energy"],  
    default=["Equities"]  
)

time_range = st.sidebar.selectbox(
    "Select Time Range",  
    ["1D", "1W", "1M", "1Y", "5Y", "Max"] 
)

# --- Date range selector ---
start_date = st.sidebar.date_input(
    "Start Date", 
    pd.to_datetime("2025-01-01")  
)
end_date = st.sidebar.date_input(
    "End Date", 
    datetime.today()
)



# ==========================================================
# Main Chart Section (Clean version using utility functions)
# ==========================================================

# --- Mapping asset classes to tickers (primary ticker for each class) ---
ASSET_CLASS_MAP = {
    "Equities": ["^GSPC"],       # S&P500
    "Commodities": ["ZW=F"],     # Wheat
    "Currencies": ["DX-Y.NYB"],  # Dollar Index
    "Bonds": ["^TNX"],           # US 10Y Treasury Yield
    "Gold": ["GC=F"],            # Gold futures
    "Energy": ["CL=F"]           # Crude Oil
}

# --- Collect tickers from selected asset classes ---
tickers = []
for cls in asset_classes:
    tickers.extend(ASSET_CLASS_MAP[cls])

# --- Download price data using your utility function ---
raw_data = get_multi_price_data(tickers, start=start_date, end=end_date, interval="1d")

# --- Clean missing values ---
data = check_and_fill_missing(raw_data)

# --- Calculate performance metrics ---
returns = calculate_returns(data)
cumulative = calculate_cumulative_returns(data)

# --- Choose theme for plotting ---
template_choice = "plotly_dark" if theme == "Dark" else "plotly_white"



# ==========================================================
# Dashboard Tabs
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cumulative Returns", 
    "Volatility", 
    "Correlation Heatmap", 
    "Correlation Chart",
    "Portfolio Simulation"
])

# --- Mapping tickers to human-readable labels ---
TICKER_LABEL_MAP = {
    "^GSPC": "Equities – S&P 500",
    "ZW=F": "Commodities – Wheat",
    "DX-Y.NYB": "Currencies – Dollar Index",
    "^TNX": "Bonds – US 10Y Treasury Yield",
    "GC=F": "Gold – Gold Futures",
    "CL=F": "Energy – Crude Oil"
}


# --- Tab 1: Cumulative Returns ---
with tab1:
    st.subheader("Cumulative Returns")
    fig_cum = plot_cumulative_returns(data)
    fig_cum.update_layout(template=template_choice)
    st.plotly_chart(fig_cum, use_container_width=True, key="cum_chart")

# --- Tab 2: Volatility ---
with tab2:
    st.subheader("Rolling Volatility (30 days)")
    fig_vol = plot_volatility(data, window=30)
    fig_vol.update_layout(template=template_choice)
    st.plotly_chart(fig_vol, use_container_width=True, key="vol_chart")

# --- Tab 3: Correlation ---
with tab3:
    st.subheader("Correlation Heatmap of Returns")

    if data.empty or len(data.columns) < 2:
        st.warning("Not enough assets selected to compute a correlation heatmap.")
    else:
        fig_corr = plot_correlation_heatmap(data)
        st.pyplot(fig_corr)
    

# --- Tab 4: Correlation Chart ---
with tab4:
    st.subheader("Correlation Between Two Assets (Fixed Period)")

    if data.empty or len(data.columns) < 2:
        st.warning("Not enough assets selected to compute correlation between two assets.")
    else:
        asset_x = st.selectbox("Select first asset", data.columns, index=0, key="corr_fixed_x")
        asset_y = st.selectbox("Select second asset", data.columns, index=1, key="corr_fixed_y")

        start_corr = st.date_input("Start date for correlation", data.index.min().date())
        end_corr = st.date_input("End date for correlation", data.index.max().date())

        if start_corr < end_corr:
            fig_corr_fixed = plot_correlation_period(data, asset_x, asset_y, start_corr, end_corr)
            fig_corr_fixed.update_layout(template=template_choice)
            st.plotly_chart(fig_corr_fixed, use_container_width=True, key="fixed_corr_chart")
        else:
            st.error("Please select a valid date range (start < end).")
        
# --- Tab 5: Portfolio Simulation ---
with tab5:
    st.subheader("Portfolio Simulation")

    if len(data.columns) >= 2:
        st.write("Select assets and adjust weights to simulate a portfolio.")

        # Construire labels lisibles
        ticker_to_label = {ticker: TICKER_LABEL_MAP.get(ticker, ticker) for ticker in data.columns}
        label_to_ticker = {v: k for k, v in ticker_to_label.items()}
        available_labels = list(ticker_to_label.values())

        # Choisir les actifs
        selected_labels = st.multiselect("Select assets for your portfolio", available_labels, default=available_labels[:2])
        selected_tickers = [label_to_ticker[label] for label in selected_labels]

        # Définir les pondérations
        weights = {}
        total_weight = 0
        for label in selected_labels:
            w = st.slider(f"Weight for {label}", min_value=0.0, max_value=1.0, value=1.0/len(selected_labels), step=0.05)
            weights[label_to_ticker[label]] = w
            total_weight += w

        # Normalisation des poids (somme = 1)
        weights = {k: v/total_weight for k, v in weights.items()}

        # Calcul du portefeuille
        portfolio_cum, weighted_returns = calculate_portfolio_returns(data[selected_tickers], weights)

        # Affichage des métriques
        st.markdown("### Portfolio Metrics")
        st.write(f"Volatility (annualized): {weighted_returns.std() * (252**0.5):.2%}")
        st.write(f"Sharpe Ratio (r=0): {weighted_returns.mean()/weighted_returns.std() * (252**0.5):.2f}")
        st.write(f"Max Drawdown: {((portfolio_cum / portfolio_cum.cummax()) - 1).min():.2%}")

        # Graphique comparatif
        fig_port = px.line(portfolio_cum, title="Portfolio Cumulative Performance", template=template_choice)
        st.plotly_chart(fig_port, use_container_width=True, key="portfolio_chart")

    else:
        st.warning("You need at least two assets to build a portfolio.")
        
VALID_TERM_CLASSES = ["Gold", "Energy", "Commodities", "Bonds"]


        
 # ==========================================================
# Section: Health of the Market – Market Regime Monitor
# ==========================================================

st.header("Health of the Market – Market Regime Monitor")

# Onglets
tab_breadth, tab_credit, tab_yield, tab_vol, tab_usd, tab_score = st.tabs([
    "Breadth Indicator",
    "Credit Spread",
    "Yield Curve",
    "Volatility Regime",
    "USD Strength",
    "Global Score"
])


# --- Breadth Indicator ---
with tab_breadth:
    st.subheader("Breadth Indicator (S&P500)")
    fig_breadth = plot_breadth()  
    st.plotly_chart(fig_breadth, use_container_width=True, key="breadth_chart")
    st.markdown(
        "**Interpretation:** Shows % of S&P500 stocks trading above their 200-day moving average. "
        "High breadth (>50%) = market strength (Risk-On). Low breadth (<50%) = weakness (Risk-Off).  \n"
        "📈 **Ticker used:** ^GSPC (S&P500 components)."
    )

# --- Credit Spread ---
with tab_credit:
    st.subheader("Credit Spread (HYG vs LQD)")
    fig_credit = plot_credit_spread()
    st.plotly_chart(fig_credit, use_container_width=True, key="credit_chart")
    st.markdown(
        "**Interpretation:** Spread between High Yield (HYG) and Investment Grade (LQD) bonds. "
        "Wider spread = more credit stress (Risk-Off). Narrow spread = improving credit conditions (Risk-On).  \n"
        "📈 **Tickers used:** HYG (High Yield ETF), LQD (Investment Grade ETF)."
    )

# --- Yield Curve ---
with tab_yield:
    st.subheader("US Yield Curve (10Y - 3M Spread)")
    fig_us_yield = plot_us_yield_curve()
    st.plotly_chart(fig_us_yield, use_container_width=True, key="us_yield_chart")

    st.markdown(
        "**Interpretation (US):** Spread between 10Y and 3M US Treasuries. "
        "Normal (10Y > 3M) = healthy growth. Inverted (10Y < 3M) = recession risk."
    )

    st.subheader("German Yield Proxy (Bund ETF IS0L.DE)")
    fig_germany = plot_german_proxy()
    st.plotly_chart(fig_germany, use_container_width=True, key="german_yield_chart")

    st.markdown(
        "**Interpretation (Germany):** Proxy for German Bunds using IS0L.DE ETF. "
        "Falling prices = rising yields. Rising prices = falling yields."
    )

# --- Volatility Regime ---
with tab_vol:
    st.subheader("Volatility Regime (VIX Index)")
    fig_vol_regime = plot_volatility_regime()
    st.plotly_chart(fig_vol_regime, use_container_width=True, key="vol_regime_chart")
    st.markdown(
        "**Interpretation:** Market volatility proxy. Low VIX = stable, Risk-On. High VIX = stress, Risk-Off.  \n"
        "📈 **Ticker used:** ^VIX (CBOE Volatility Index)."
    )

# --- USD Strength ---
with tab_usd:
    st.subheader("USD Strength (Dollar Index)")
    fig_usd = plot_usd_strength()
    st.plotly_chart(fig_usd, use_container_width=True, key="usd_strength_chart")
    st.markdown(
        "**Interpretation:** Strong USD = global risk aversion, capital flows to safety (Risk-Off). "
        "Weak USD = more risk appetite (Risk-On).  \n"
        "📈 **Ticker used:** DX-Y.NYB (US Dollar Index)."
    )

# --- Global Market Regime Score (Single Value Display) ---
with tab_score:
    st.subheader("Global Market Regime Score")
    display_market_regime_score()  # <--- Appelle la fonction que je t’ai donnée
    st.markdown(
        "**Interpretation:** This indicator summarizes credit spreads, yield curve, volatility, "
        "and USD strength into a single score.  \n"
        "🟢 Positive = Risk-On  |  🟡 Neutral = No clear signal  |  🔴 Negative = Risk-Off."
    )
    

# ==========================================================
# Section: Macroeconomic Indicators
# ==========================================================

st.header("Macroeconomic Indicators")

tab_growth, tab_inflation = st.tabs([
    "Growth", 
    "Inflation"
])


# --- Growth Proxies ---
with tab_growth:
    st.subheader("Growth Proxies")

    # --- Proxies disponibles ---
    growth_proxies = {
        "S&P500 (Equities Proxy)": "^GSPC",
        "DJT (Dow Jones Transportation)": "DJT",
        "EEM (Emerging Markets ETF)": "EEM",
        "IWM (US Small Caps)": "IWM",
        "XLI (Industrials ETF)": "XLI",
        "XLK (Technology ETF)": "XLK"
    }

    # Multiselect avec ^GSPC comme défaut
    selected_growth = st.multiselect(
        "Select growth proxies to display",
        options=list(growth_proxies.keys()),
        default=["S&P500 (Equities Proxy)"]
    )

    if selected_growth:
        tickers = [growth_proxies[label] for label in selected_growth]
        df_growth = get_multi_price_data(tickers, start="2000-01-01", end=datetime.today(), interval="1d")

        if not df_growth.empty:
            # Sélecteur de période dynamique
            min_date, max_date = df_growth.index.min().date(), df_growth.index.max().date()
            start_growth = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_growth = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            # Filtrer les données
            df_growth_filtered = df_growth.loc[start_growth:end_growth]

            # Normaliser (base 100)
            df_growth_norm = normalize_data(df_growth_filtered.dropna(axis=1, how="all"))

            # Message en haut
            st.markdown(
                "⚠️ *Curves will not display if the index is not available in the selected period.*"
            )

            if not df_growth_norm.empty:
                fig_growth = px.line(
                    df_growth_norm,
                    title="Growth Proxies (Normalized Performance)",
                    template=template_choice
                )
                fig_growth.update_layout(yaxis_title="Index (Base 100)", xaxis_title="Date")
                st.plotly_chart(fig_growth, use_container_width=True)

            # Explication
            st.markdown("""
            **Why these proxies?**  
            - **S&P500**: broad equity market → baseline growth barometer.  
            - **DJT**: transports reflect goods movement → economic activity.  
            - **EEM**: emerging markets, cyclical & growth-sensitive.  
            - **IWM**: US small caps → domestic growth.  
            - **XLI**: industrials → production & investment cycles.  
            - **XLK**: technology → innovation-driven growth.  
            """)
        else:
            st.warning("No data available for the selected proxies.")
            
# --- Inflation Proxy (CPI from FRED) ---
with tab_inflation:
    st.subheader("Inflation Proxy (CPI)")

    try:
        # Download CPI data from FRED (monthly data)
        cpi = web.DataReader("CPIAUCSL", "fred", start="1960-01-01", end=datetime.today())

        # --- Date range selector (flexible) ---
        start_cpi = st.date_input(
            "Start Date",
            datetime.today() - timedelta(days=365)  # par défaut = 1 an avant aujourd'hui
        )
        end_cpi = st.date_input("End Date", datetime.today())

        # Filtrer les données selon les dates choisies
        cpi_filtered = cpi.loc[(cpi.index >= str(start_cpi)) & (cpi.index <= str(end_cpi))]

        # --- Choix d'affichage ---
        view_mode = st.radio(
            "Select View",
            ["Index (Base 100)", "Year-over-Year % Change"],
            index=0,
            horizontal=True
        )

        if view_mode == "Index (Base 100)":
            cpi_norm = cpi_filtered / cpi_filtered.iloc[0] * 100
            fig_cpi = px.line(
                cpi_norm,
                title=f"US CPI Index ({start_cpi} → {end_cpi})",
                template=template_choice
            )
            fig_cpi.update_layout(yaxis_title="Index (Base 100)", xaxis_title="Date")

        else:  # YoY % change
            cpi_yoy = cpi_filtered.pct_change(periods=12) * 100
            fig_cpi = px.line(
                cpi_yoy,
                title=f"US CPI Year-over-Year % Change ({start_cpi} → {end_cpi})",
                template=template_choice
            )
            fig_cpi.update_layout(yaxis_title="YoY % Change", xaxis_title="Date")

        # Display chart
        st.plotly_chart(fig_cpi, use_container_width=True)

        # Add interpretation
        st.markdown(
            "**Interpretation:**  \n"
            "- *Index (Base 100)* = long-term inflation trend.  \n"
            "- *YoY % Change* = actual inflation rate.  \n"
            f"Displayed period: **{start_cpi} → {end_cpi}**"
        )

    except Exception as e:
        st.warning(f"Could not load CPI data from FRED: {e}")
        
# ==========================================================
# FX & Central Banks Section
# ==========================================================
st.header("FX & Central Banks")

tab_fx, tab_rates = st.tabs([
    "FX Majors",
    "Central Bank Policy Rates"
])


# ----------------------------------------------------------
# Tab 1: FX Majors
# ----------------------------------------------------------
with tab_fx:
    st.subheader("FX Majors")

    # --- FX tickers map ---
    FX_TICKERS = {
        "US Dollar Index (DXY)": "DX-Y.NYB",
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X",
        "GBP/USD": "GBPUSD=X",
        "USD/CNY": "USDCNY=X"
    }

    # --- User selection for cumulative returns ---
    selected_fx = st.multiselect(
        "Select FX pairs to display:",
        list(FX_TICKERS.keys()),
        default=["US Dollar Index (DXY)", "EUR/USD"]
    )

    if selected_fx:
        tickers = [FX_TICKERS[pair] for pair in selected_fx]
        fx_data = get_multi_price_data(
            tickers,
            start="2010-01-01",
            end=datetime.today(),
            interval="1d"
        )
        fx_data = check_and_fill_missing(fx_data)

        if not fx_data.empty:
            # --- Cumulative returns chart ---
            cum_returns = calculate_cumulative_returns(fx_data)
            fig_fx = px.line(
                cum_returns,
                title="Cumulative Returns of Selected FX Pairs",
                template=template_choice
            )
            fig_fx.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date")
            st.plotly_chart(fig_fx, use_container_width=True)

            # --- Current FX levels (only selected pairs) ---
            st.markdown("### Current FX Levels")
            last_day = fx_data.iloc[-1]
            prev_day = fx_data.iloc[-2]

            cols = st.columns(len(selected_fx))
            for i, pair in enumerate(selected_fx):
                ticker = FX_TICKERS[pair]
                latest = last_day[ticker]
                change_pct = ((last_day[ticker] - prev_day[ticker]) / prev_day[ticker]) * 100

                with cols[i]:
                    st.metric(
                        label=pair,
                        value=f"{latest:.2f}",
                        delta=f"{change_pct:.2f} %"
                    )
        else:
            st.warning("No FX data available for the selected pairs and date range.")


# ----------------------------------------------------------
# Tab 2: Central Bank Policy Rates
# ----------------------------------------------------------
with tab_rates:
    st.subheader("Central Bank Policy Rates")

    st.markdown(
        """
        Below are the **current policy rates** of the main central banks.  
        """
    )

    col1, col2, col3 = st.columns(3)

    # --- Federal Reserve (FED) ---
    try:
        fed_rate = web.DataReader("FEDFUNDS", "fred", start="2000-01-01", end=datetime.today()).iloc[-1].values[0]
        col1.metric("🇺🇸 Federal Reserve (FED)", f"{fed_rate:.2f}%")
    except:
        col1.metric("🇺🇸 Federal Reserve (FED)", "N/A")

    # --- European Central Bank (ECB) ---
    try:
        ecb_rate = web.DataReader("ECBDFR", "fred", start="2000-01-01", end=datetime.today()).iloc[-1].values[0]
        col2.metric("🇪🇺 European Central Bank (ECB)", f"{ecb_rate:.2f}%")
    except:
        col2.metric("🇪🇺 European Central Bank (ECB)", "N/A")

    # --- Bank of England (BoE) ---
    try:
        boe_rate = web.DataReader("BOERUKM", "fred", start="2000-01-01", end=datetime.today()).iloc[-1].values[0]
        col3.metric("🇬🇧 Bank of England (BoE)", f"{boe_rate:.2f}%")
    except:
        col3.metric("🇬🇧 Bank of England (BoE)", "N/A")

    col4, col5, col6 = st.columns(3)

    # --- Bank of Japan (BoJ) ---
    try:
        boj_rate = web.DataReader("INTDSRJPM193N", "fred", start="2000-01-01", end=datetime.today()).iloc[-1].values[0]
        col4.metric("🇯🇵 Bank of Japan (BoJ)", f"{boj_rate:.2f}%")
    except:
        col4.metric("🇯🇵 Bank of Japan (BoJ)", "N/A")

    # --- Swiss National Bank (SNB) ---
    try:
        snb_rate = web.DataReader("IR3TIB01CHM156N", "fred", start="2000-01-01", end=datetime.today()).iloc[-1].values[0]
        col5.metric("🇨🇭 Swiss National Bank (SNB)", f"{snb_rate:.2f}%")
    except:
        col5.metric("🇨🇭 Swiss National Bank (SNB)", "N/A")

    # --- Bank of Canada (BoC) ---
    try:
        boc_rate = web.DataReader("IR3TIB01CAM156N", "fred", start="2000-01-01", end=datetime.today()).iloc[-1].values[0]
        col6.metric("🇨🇦 Bank of Canada (BoC)", f"{boc_rate:.2f}%")
    except:
        col6.metric("🇨🇦 Bank of Canada (BoC)", "N/A")
