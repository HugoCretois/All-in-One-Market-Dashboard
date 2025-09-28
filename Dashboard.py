# ==========================================================
# All-in-One Market Dashboard
# MSc in Financial Markets and Investments - Skema Business School
# Python Programming for Finance
# Academic Year 2025/26
# ==========================================================

# ==========================================================
# SOMMAIRE – ALL-IN-ONE MARKET DASHBOARD
# ----------------------------------------------------------
# 1. Imports & Setup
# 2. Utility Functions (data download, cleaning, metrics)
# 3. Data Preprocessing & Metrics (returns, vol, drawdown)
# 4. Data Visualization & Market Regime Indicators
# 5. Streamlit Dashboard Layout
#    5.1 Page Configuration, Navigation & Global Settings         
#    5.2 Data Preparation for Dashboard
#    5.3 Market Performance – Main Charts (Tabs)
#    5.4 Economic Situation – Market Regime Monitor
#    5.5 Macroeconomic Indicators
#    5.6 FX & Central Banks
# ===========================================================

#Lets start!

# ==========================================================
# 1. IMPORTS & SETUP
# ==========================================================

# --- Data handling ---
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dashboard framework ---
import streamlit as st

# --- Utilities ---
from datetime import datetime, timedelta


# --- Streamlit page setup ---
st.set_page_config(
    page_title="All-in-One Asset Dashboard",
    layout="wide"
)

# ==========================================================
# 2. UTILITY FUNCTIONS (DATA & METRICS)
#-----------------------------------------------------------
# These functions handle the basic operations needed 
# to prepare market data:
# - Downloading prices for one or several tickers
# - Cleaning missing values
# - Organizing data into DataFrames by asset class
# ==========================================================

#---------------------------------------------------
# Download historical price data for a single ticker
#---------------------------------------------------

def get_price_data(ticker, start="1800-01-01", end=datetime.today().strftime("%Y-%m-%d"), interval="1d"):
  
    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        print(f"[WARNING] No data for {ticker}")
        return pd.DataFrame()

    if "Adj Close" in df.columns:
        series = df["Adj Close"]
    elif "Close" in df.columns:
        series = df["Close"]
    else:
        print(f"[WARNING] No 'Adj Close' or 'Close' column for {ticker}")
        return pd.DataFrame()
    
#-----------------------------------------------------------------------
# Ensure the output is always a DataFrame with the ticker as column name
#-----------------------------------------------------------------------

    if isinstance(series, pd.Series):
        return series.to_frame(name=ticker)
    else:
        return series.rename(columns={series.columns[0]: ticker})

#---------------------------------------------------    
# Download historical price data for multiple ticker
#---------------------------------------------------

def get_multi_price_data(tickers, start="1800-01-01", end=datetime.today().strftime("%Y-%m-%d"), interval="1d"):

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

#------------------------------------------------------
# Ensure output is always a DataFrame with correct name
#------------------------------------------------------

        if isinstance(series, pd.Series):
            series = series.to_frame(name=ticker)
        elif isinstance(series, pd.DataFrame):
            series = series.rename(columns={series.columns[0]: ticker})

        data = pd.concat([data, series], axis=1)

    return data

#------------------------------------------
# Function to check and fill missing values 
#------------------------------------------

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

#-------------------------------
# Data Collection for All Assets 
#-------------------------------

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
# 3. DATA PREPROCESSING & METRICS
#-----------------------------------------------------------
# These functions transform raw market data into meaningful
# indicators that can later be visualized:
# - Returns (daily, cumulative)
# - Normalization (base 100)
# - Volatility (rolling, annualized)
# - Drawdown (series + max drawdown)
# - Portfolio returns (given weights)
# ==========================================================


#----------------------------------------
# Calculate daily percentage returns
#----------------------------------------

def calculate_returns(price_df):
    returns = price_df.pct_change().dropna()
    return returns

#----------------------------------------
# Calculate cumulative returns over time
#----------------------------------------

def calculate_cumulative_returns(price_df):
    returns = calculate_returns(price_df)
    cumulative = (1 + returns).cumprod()
    return cumulative


#----------------------------------------
# Normalize price series to start at 100
#----------------------------------------

def normalize_data(price_df):
    normalized = price_df / price_df.iloc[0] * 100
    return normalized


#----------------------------------------
# Calculate rolling annualized volatility
#----------------------------------------

def calculate_volatility(price_df, window=30):
    returns = calculate_returns(price_df)
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

#----------------------------------------
# Calculate drawdown and maximum drawdown
#----------------------------------------

def calculate_drawdown(price_df):
    cum_returns = calculate_cumulative_returns(price_df)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown


#------------------------------------------------------
# Calculate portfolio returns given weights of assets
#------------------------------------------------------

def calculate_portfolio_returns(price_df, weights):
    returns = calculate_returns(price_df)
    weighted_returns = (returns * pd.Series(weights)).sum(axis=1)
    portfolio_cum = (1 + weighted_returns).cumprod()
    return portfolio_cum, weighted_returns


# ==========================================================
# 4. DATA VISUALIZATION & MARKET REGIME INDICATORS
#-----------------------------------------------------------
# These functions take processed metrics and create visual
# outputs to explore the data:
# - Time series plots (normalized, cumulative, volatility)
# - Correlation analysis (heatmap, scatter/regression)
# - Market Regime indicators (breadth, yield curve, spreads,
#   volatility, USD, composite score)
# ==========================================================

#----------------------------------------
# Plot normalized time series (base 100)
#----------------------------------------

def plot_normalized_series(price_df, title="Normalized Performance (Base 100)"):
    normalized = normalize_data(price_df)
    fig = px.line(normalized, title=title)
    fig.update_layout(yaxis_title="Index (Base 100)", xaxis_title="Date")
    return fig


#----------------------------------------
# Plot cumulative returns
#----------------------------------------

def plot_cumulative_returns(price_df, title="Cumulative Returns"):
    cum_returns = calculate_cumulative_returns(price_df)
    fig = px.line(cum_returns, title=title)
    fig.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date")
    return fig


#----------------------------------------
# Plot rolling volatility
#----------------------------------------

def plot_volatility(price_df, window=30, title="Rolling Volatility (30d)"):
    vol = calculate_volatility(price_df, window=window)
    fig = px.line(vol, title=title)
    fig.update_layout(yaxis_title="Volatility (annualized)", xaxis_title="Date")
    return fig


#----------------------------------------
# Plot correlation heatmap
#----------------------------------------

def plot_correlation_heatmap(price_df, title="Correlation Heatmap"):
    returns = calculate_returns(price_df)
    corr_matrix = returns.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(title)
    return fig

#----------------------------------------
# Plot correlation between two assets
#----------------------------------------
def plot_correlation_period(price_df, asset_x, asset_y, start_date, end_date, title=None):
    returns = calculate_returns(price_df)
    period_data = returns.loc[start_date:end_date, [asset_x, asset_y]].dropna()
    corr_value = period_data[asset_x].corr(period_data[asset_y])

    if title is None:
        title = f"Correlation between {asset_x} and {asset_y} ({start_date} to {end_date})"

    fig = px.scatter(
        period_data,
        x=asset_x,
        y=asset_y,
        trendline="ols",
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

#----------------------------------------
# Breadth Indicator
#----------------------------------------

def plot_breadth(start=None, end=None, window=200):
    sp500 = get_price_data("^GSPC", start=start, end=end)
    if sp500 is None or sp500.empty:
        st.warning("No data available for ^GSPC.")
        return px.line()

    ma = sp500["^GSPC"].rolling(window).mean()
    breadth = (sp500["^GSPC"] / ma - 1) * 100

    fig = px.line(breadth, title="Breadth Indicator (^GSPC vs 200d MA)")
    fig.update_layout(yaxis_title="% Above 200d MA", xaxis_title="Date")
    return fig



#----------------------------------------
# Yield Curve (10Y - 3M)
#----------------------------------------

def plot_us_yield_curve(start=None, end=None):
    t10y = get_price_data("^TNX", start=start, end=end)
    t3m = get_price_data("^IRX", start=start, end=end)
    if t10y is None or t3m is None or t10y.empty or t3m.empty:
        st.warning("No data available for US yield curve (^TNX, ^IRX).")
        return px.line()

    spread = (t10y["^TNX"]/100) - (t3m["^IRX"]/100)
    fig = px.line(spread, title="US Yield Curve (10Y - 3M Spread)")
    fig.update_layout(yaxis_title="Spread (%)", xaxis_title="Date")
    return fig


#----------------------------------------
# Credit Spread (HYG - LQD)
#----------------------------------------

def plot_credit_spread(start=None, end=None):
    hyg = get_price_data("HYG", start=start, end=end)
    lqd = get_price_data("LQD", start=start, end=end)

    if hyg is None or lqd is None or hyg.empty or lqd.empty:
        st.warning("No data available for HYG or LQD.")
        return px.line()

    # Align the two time series
    df = pd.concat([hyg, lqd], axis=1).dropna()
    df.columns = ["HYG", "LQD"]

    # Compute relative performance (spread)
    hyg_ret = df["HYG"].pct_change().fillna(0)
    lqd_ret = df["LQD"].pct_change().fillna(0)
    spread = (hyg_ret - lqd_ret).cumsum()

    fig = px.line(spread, title="Credit Spread (HYG - LQD, cumulative)")
    fig.update_layout(yaxis_title="Relative Performance", xaxis_title="Date")
    return fig


#----------------------------------------
# Volatility Regime (VIX Index)
#----------------------------------------

def plot_volatility_regime(start=None, end=None):
    vix = get_price_data("^VIX", start=start, end=end)
    if vix is None or vix.empty:
        st.warning("No data available for ^VIX.")
        return px.line()

    fig = px.line(vix, title="Volatility Regime (VIX Index)")
    fig.update_layout(yaxis_title="VIX Level", xaxis_title="Date")
    return fig


#----------------------------------------
# USD Strength (DXY Index)
#----------------------------------------

def plot_usd_strength(start=None, end=None):
    dxy = get_price_data("DX-Y.NYB", start=start, end=end)
    if dxy is None or dxy.empty:
        st.warning("No data available for DXY (DX-Y.NYB).")
        return px.line()

    fig = px.line(dxy, title="USD Strength (DXY Index)")
    fig.update_layout(yaxis_title="Index Level", xaxis_title="Date")
    return fig

#----------------------------------------
# Global Market Regime Score (with min-max bar)
#----------------------------------------

def display_market_regime_score(show_chart=True):
    """
    Display the Global Market Regime Score with historical fixed min/max.
    Score of the day compared against history since 2000.
    """

    # --- Télécharger données depuis 2000 ---
    sp500 = get_price_data("^GSPC", start="2000-01-01")
    hyg = get_price_data("HYG", start="2000-01-01")
    lqd = get_price_data("LQD", start="2000-01-01")
    t10y = get_price_data("^TNX", start="2000-01-01")
    t3m = get_price_data("^IRX", start="2000-01-01")
    vix = get_price_data("^VIX", start="2000-01-01")
    dxy = get_price_data("DX-Y.NYB", start="2000-01-01")

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

    score = -norm_credit + norm_yield - norm_vix - norm_dxy

    # --- Score du jour ---
    latest_score = score.iloc[-1]

    # --- Min et Max historiques ---
    min_score = score.min()
    max_score = score.max()
    min_date = score.idxmin().strftime("%Y-%m-%d")
    max_date = score.idxmax().strftime("%Y-%m-%d")

    # --- Barre style thermomètre ---
    if show_chart:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_score,
            title={'text': "Global Market Regime Score"},
            gauge={
                'axis': {'range': [min_score, max_score]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [min_score, (min_score+max_score)/3], 'color': "crimson"},
                    {'range': [(min_score+max_score)/3, (2*max_score+min_score)/3], 'color': "gold"},
                    {'range': [(2*max_score+min_score)/3, max_score], 'color': "seagreen"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # --- Infos texte ---
    st.markdown(f"""
    **📉 Minimum historique :** {min_score:.2f} atteint le {min_date}  
    **📈 Maximum historique :** {max_score:.2f} atteint le {max_date}  
    **📊 Score actuel :** {latest_score:.2f}  

    **Interpretation :**  
    - Le score combine **credit spreads, yield curve, VIX et USD**.  
    - Valeurs proches du **minimum** = stress extrême, *Risk-Off*.  
    - Valeurs proches du **maximum** = confiance élevée, *Risk-On*.  
    """)


# ==========================================================
# 5. STREAMLIT DASHBOARD LAYOUT
# ==========================================================


# ==========================================================
# 5.1 PAGE CONFIGURATION, NAVIGATION & GLOBAL SETTINGS
#-----------------------------------------------------------
# This section sets up the Streamlit application with:
# - Page title, layout, and sidebar behavior
# - Global CSS (fonts, style customization)
# - Navigation menu (Table of Contents in the sidebar)
# - Home page with banner and introduction
# - Sidebar filters (asset classes & manual date range)
# - Global mappings for tickers and human-readable labels
# ==========================================================

#---------------------------------------------------
# Page setup
#---------------------------------------------------

st.set_page_config(
    page_title="All-in-One Market Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"  # 👈 ferme la sidebar par défaut
)

#---------------------------------------------------
# LinkedIn icon (top-right corner, global)
#---------------------------------------------------

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 20px;
        z-index: 100;
        display: flex;
        align-items: center;
        font-size: 14px;
        color: #555;
    }
    .footer img {
        width: 20px;
        margin-right: 8px;
    }
    </style>

    <div class="footer">
        <a href="https://www.linkedin.com/in/hugo-cretois" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
        </a>
        <span>Author: <b>Hugo Crétois</b>, MSc FMI – Skema Business School</span>
    </div>
    """,
    unsafe_allow_html=True
)



#---------------------------------------------------
# Navigation (Table of Contents) 
#---------------------------------------------------

st.sidebar.header("📌 Navigation")   # titre principal

st.sidebar.subheader("Go to section:")

menu = st.sidebar.radio(
    "",
    [
        "Home",                   # Page d'accueil
        "Market performance",        # 5.2 + 5.3
        "Economic Situation",        # 5.4
        "Macroeconomic Indicators",  # 5.5
        "FX & Central Banks"         # 5.6
    ],
    index=0  # Default = Home
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if menu == "Home":
    st.markdown(
        """
        <div style="
            position: relative;
            text-align: center;
            color: white;
            padding: 80px 20px;
            background-image: url('https://th.bing.com/th/id/R.bef484c1c2679790df69058bcd570d53?rik=x%2fEr%2bxzkWWpCIQ&riu=http%3a%2f%2fwww.pixelstalk.net%2fwp-content%2fuploads%2f2016%2f04%2fBlue-Night-HD-Wallpaper-Snow-Mountains-Image-Widescreen-Picture.jpg&ehk=h6h4kxpnAkePDVvHotBgPyDCKU48RSwfJJm5bRknob4%3d&risl=&pid=ImgRaw&r=0');
            background-size: cover;
            background-position: center;
            border-radius: 12px;
        ">
            <h1 style="font-size: 52px; font-weight: bold; margin: 0;">
                 All-in-One Asset Dashboard
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
 # --- Ligne de séparation ---
    st.markdown("---")
    
    st.markdown(
        """
        <p style='text-align: center; font-size:18px;'>
        Welcome to your dashboard!  
        This project is an exercise to practice <b>Python data manipulation</b>  
        while exploring <b>financial market insights</b>.  
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Clean summary
    st.subheader("Table of Contents")
    st.markdown(
        """
        - **Market performance**   
          Explore equities, bonds, commodities, currencies, gold & energy.  
        
        - **Economic Situation**   
          Monitor market regimes: breadth, credit spreads, yield curve, volatility, USD strength.  

        - **Macroeconomic Indicators**   
          Track growth (equities, transports, small caps, EM) and inflation (CPI).  

        - **FX & Central Banks**   
          Compare FX majors and central bank policy rates (Fed, ECB, BoE, BoJ, SNB, BoC).  
        """
    )

    st.markdown("---")

    st.info("➡️ Use the **sidebar menu** to navigate directly to each section.")
    
#---------------------------------------------------
# Sidebar filters
#---------------------------------------------------

st.sidebar.header("Filters")

asset_classes = st.sidebar.multiselect(
    "Select Asset Classes",
    ["Equities", "Commodities", "Currencies", "Bonds", "Gold", "Energy"],
    default=["Equities"]
)


start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("1950-01-01"),   # 👈 par défaut en 1950
    min_value=pd.to_datetime("1950-01-01"),  # 👈 limite basse
    max_value=datetime.today()                 # 👈 limite haute
)

end_date = st.sidebar.date_input(
    "End Date",
    datetime.today()
)

#---------------------------------------------------
# Visualization template
#---------------------------------------------------

template_choice = "plotly_white"  # ou "plotly_dark" si tu préfères

#---------------------------------------------------
# Asset class to ticker mapping
#---------------------------------------------------

ASSET_CLASS_MAP = {
    "Equities": ["^GSPC"],       # S&P500
    "Commodities": ["ZW=F"],     # Wheat
    "Currencies": ["DX-Y.NYB"],  # Dollar Index
    "Bonds": ["^TNX"],           # US 10Y Treasury Yield
    "Gold": ["GC=F"],            # Gold futures
    "Energy": ["CL=F"]           # Crude Oil
}

# Collect tickers
tickers = []
for cls in asset_classes:
    tickers.extend(ASSET_CLASS_MAP[cls])

# Download + clean
raw_data = get_multi_price_data(tickers, start=start_date, end=end_date, interval="1d")
data = check_and_fill_missing(raw_data)

# Pre-calc metrics
returns = calculate_returns(data)
cumulative = calculate_cumulative_returns(data)

#---------------------------------------------------
# Ticker to label mapping
#---------------------------------------------------

TICKER_LABEL_MAP = {
    "^GSPC": "Equities – S&P 500",
    "ZW=F": "Commodities – Wheat",
    "DX-Y.NYB": "Currencies – Dollar Index",
    "^TNX": "Bonds – US 10Y Treasury Yield",
    "GC=F": "Gold – Gold Futures",
    "CL=F": "Energy – Crude Oil"
}

# ==========================================================
# 5.2 DATA PREPARATION FOR DASHBOARD
#-----------------------------------------------------------
# This step prepares the market data for visualizations:
# - Collect tickers from selected asset classes
# - Download data via utility functions
# - Clean missing values
# - Compute main performance metrics
# ==========================================================

#---------------------------------------------------
# Collect tickers from selected asset classes
#---------------------------------------------------

tickers = []
for cls in asset_classes:
    tickers.extend(ASSET_CLASS_MAP[cls])

#---------------------------------------------------
# Download and clean price data
#---------------------------------------------------

raw_data = get_multi_price_data(
    tickers, start=start_date, end=end_date, interval="1d"
)
data = check_and_fill_missing(raw_data)

#---------------------------------------------------
# Calculate performance metrics
#---------------------------------------------------

returns = calculate_returns(data)
cumulative = calculate_cumulative_returns(data)

# ==========================================================
# 5.3 DASHBOARD – MAIN CHARTS (Tabs)
#-----------------------------------------------------------
# Interactive dashboard section with 5 main tabs:
# - Cumulative Returns
# - Rolling Volatility
# - Correlation Heatmap
# - Correlation Chart (pairwise)
# - Portfolio Simulation
# ==========================================================

if menu == "Market performance":

    # --- Section Title ---
    st.header("Market performance")

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cumulative Returns",
        "Volatility",
        "Correlation Heatmap",
        "Correlation Chart",
        "Portfolio Simulation"
    ])

    #---------------------------------------------------
    # Tab 1: Cumulative Returns
    #---------------------------------------------------
    with tab1:
        st.subheader("Cumulative Returns")
        fig_cum = plot_cumulative_returns(data)
        fig_cum.update_layout(template=template_choice)
        st.plotly_chart(fig_cum, use_container_width=True, key="cum_chart")

    #---------------------------------------------------
    # Tab 2: Volatility
    #---------------------------------------------------
    with tab2:
        st.subheader("Rolling Volatility (30 days)")
        fig_vol = plot_volatility(data, window=30)
        fig_vol.update_layout(template=template_choice)
        st.plotly_chart(fig_vol, use_container_width=True, key="vol_chart")

    #---------------------------------------------------
    # Tab 3: Correlation Heatmap
    #---------------------------------------------------
    with tab3:
        st.subheader("Correlation Heatmap of Returns")

        if data.empty or len(data.columns) < 2:
            st.warning("Not enough assets selected to compute a correlation heatmap.")
        else:
            fig_corr = plot_correlation_heatmap(data)
            st.pyplot(fig_corr)

    #---------------------------------------------------
    # Tab 4: Correlation Chart (pairwise)
    #---------------------------------------------------
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

    #---------------------------------------------------
    # Tab 5: Portfolio Simulation
    #---------------------------------------------------
    with tab5:
        st.subheader("Portfolio Simulation")

        if len(data.columns) >= 2:
            st.write("Select assets and adjust weights to simulate a portfolio.")

            # Build readable labels
            ticker_to_label = {ticker: TICKER_LABEL_MAP.get(ticker, ticker) for ticker in data.columns}
            label_to_ticker = {v: k for k, v in ticker_to_label.items()}
            available_labels = list(ticker_to_label.values())

            # Choose assets
            selected_labels = st.multiselect(
                "Select assets for your portfolio",
                available_labels,
                default=available_labels[:2]
            )
            selected_tickers = [label_to_ticker[label] for label in selected_labels]

            # Set weights
            weights = {}
            total_weight = 0
            for label in selected_labels:
                w = st.slider(
                    f"Weight for {label}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(selected_labels),
                    step=0.05
                )
                weights[label_to_ticker[label]] = w
                total_weight += w

            # Normalize weights (sum = 1)
            weights = {k: v/total_weight for k, v in weights.items()}

            # Portfolio calculation
            portfolio_cum, weighted_returns = calculate_portfolio_returns(data[selected_tickers], weights)

            # Portfolio metrics
            st.markdown("### Portfolio Metrics")
            st.write(f"Volatility (annualized): {weighted_returns.std() * (252**0.5):.2%}")
            st.write(f"Sharpe Ratio (r=0): {weighted_returns.mean()/weighted_returns.std() * (252**0.5):.2f}")
            st.write(f"Max Drawdown: {((portfolio_cum / portfolio_cum.cummax()) - 1).min():.2%}")

            # Portfolio chart
            fig_port = px.line(
                portfolio_cum,
                title="Portfolio Cumulative Performance",
                template=template_choice
            )
            st.plotly_chart(fig_port, use_container_width=True, key="portfolio_chart")

        else:
            st.warning("You need at least two assets to build a portfolio.")

        

# ==========================================================
# 5.4 ECONOMIC SITUATION – MARKET REGIME MONITOR
#-----------------------------------------------------------
# This section monitors the overall health of the markets 
# through regime indicators:
# - Breadth Indicator (S&P500 > 200d MA)
# - Credit Spread (HYG vs LQD)
# - US Yield Curve (10Y - 3M)
# - Volatility Regime (VIX Index)
# - USD Strength (DXY Index)
# - Global Market Regime Score (composite indicator)
# ==========================================================

if menu == "Economic Situation":

    # --- Section Title ---
    st.header("Economic Situation")

    # --- Tabs for regime indicators ---
    tab_breadth, tab_credit, tab_yield, tab_vol, tab_usd, tab_score = st.tabs([
        "Breadth Indicator",
        "Credit Spread",
        "Yield Curve",
        "Volatility Regime",
        "USD Strength",
        "Global Score"
    ])

    # ----------------------------------------------------------
    # Breadth Indicator (S&P500)
    # ----------------------------------------------------------
    with tab_breadth:
        st.subheader("Breadth Indicator (S&P500)")
        fig_breadth = plot_breadth(start=start_date, end=end_date)  # 🔗 filtres dates
        st.plotly_chart(fig_breadth, use_container_width=True, key="breadth_chart")
        st.markdown(
            "**Interpretation:** Shows % of S&P500 stocks trading above their 200-day moving average. "
            "High breadth (>50%) = market strength (Risk-On). Low breadth (<50%) = weakness (Risk-Off).  \n"
            "📈 **Ticker used:** ^GSPC (S&P500 index)."
        )

    # ----------------------------------------------------------
    # Credit Spread (HYG vs LQD)
    # ----------------------------------------------------------
    with tab_credit:
        st.subheader("Credit Spread (HYG vs LQD)")
        fig_credit = plot_credit_spread(start=start_date, end=end_date)  # 🔗 filtres dates
        st.plotly_chart(fig_credit, use_container_width=True, key="credit_chart")
        st.markdown(
            "**Interpretation:** Spread between High Yield (HYG) and Investment Grade (LQD) bonds. "
            "Wider spread = more credit stress (Risk-Off). Narrow spread = improving credit conditions (Risk-On).  \n"
            "📈 **Tickers used:** HYG (High Yield ETF), LQD (Investment Grade ETF)."
        )

    # ----------------------------------------------------------
    # Yield Curve (US 10Y - 3M)
    # ----------------------------------------------------------
    with tab_yield:
        st.subheader("US Yield Curve (10Y - 3M Spread)")
        fig_us_yield = plot_us_yield_curve(start=start_date, end=end_date)  # 🔗 filtres dates
        st.plotly_chart(fig_us_yield, use_container_width=True, key="us_yield_chart")
        st.markdown(
            "**Interpretation (US):** Spread between 10Y and 3M US Treasuries. "
            "Normal (10Y > 3M) = healthy growth. Inverted (10Y < 3M) = recession risk."
        )

    # ----------------------------------------------------------
    # Volatility Regime (VIX Index)
    # ----------------------------------------------------------
    with tab_vol:
        st.subheader("Volatility Regime (VIX Index)")
        fig_vol_regime = plot_volatility_regime(start=start_date, end=end_date)  # 🔗 filtres dates
        st.plotly_chart(fig_vol_regime, use_container_width=True, key="vol_regime_chart")
        st.markdown(
            "**Interpretation:** Market volatility proxy. Low VIX = stable, Risk-On. High VIX = stress, Risk-Off.  \n"
            "📈 **Ticker used:** ^VIX (CBOE Volatility Index)."
        )

    # ----------------------------------------------------------
    # USD Strength (DXY Index)
    # ----------------------------------------------------------
    with tab_usd:
        st.subheader("USD Strength (Dollar Index)")
        fig_usd = plot_usd_strength(start=start_date, end=end_date)  # 🔗 filtres dates
        st.plotly_chart(fig_usd, use_container_width=True, key="usd_strength_chart")
        st.markdown(
            "**Interpretation:** Strong USD = global risk aversion, capital flows to safety (Risk-Off). "
            "Weak USD = more risk appetite (Risk-On).  \n"
            "📈 **Ticker used:** DX-Y.NYB (US Dollar Index)."
        )

    # ----------------------------------------------------------
    # Global Market Regime Score (Composite Indicator)
    # ----------------------------------------------------------
    
    with tab_score:
        st.subheader("Global Market Regime Score")

        # Display chart + current score
        display_market_regime_score()   # 👈 enlève start et end si ta fonction ne les attend pas

        # ✅ Interpretation (colle ici le bloc markdown que tu avais écrit)
        st.markdown(
        """
        ### Interpretation of the Global Market Regime Score  

        - **Historical Minimum: -15.21 (March 17, 2020)**  
          This point reflects the peak of the **Covid-19 crisis**, when volatility (VIX) spiked,  
          the USD surged, and credit spreads widened dramatically.  
          Markets were in a state of **extreme Risk-Off**, driven by panic selling.  

        - **Historical Maximum: 9.29 (September 30, 2008)**  
          This peak occurred during the **Global Financial Crisis**.  
          The surge was largely driven by the explosion in **credit spreads (HYG – LQD)**,  
          which outweighed the other components in the formula.  
          It shows that the score should be read as a **relative measure of market stress**,  
          not as a strict Risk-On signal.  

        ---
        👉 **In summary:**  
        - The score ranges from **extreme stress (-15 in 2020)** to **credit-driven peaks (+9 in 2008)**.  
        """
    )

# ==========================================================
# 5.5 MACROECONOMIC INDICATORS
#-----------------------------------------------------------
# This section tracks macroeconomic drivers that influence
# market regimes and asset prices:
# - Growth proxies (equities, transports, emerging, small caps)
# - Inflation proxy (CPI from FRED)
# ==========================================================

# ==========================================================
# 5.5 MACROECONOMIC INDICATORS
#-----------------------------------------------------------
# This section tracks macroeconomic drivers that influence
# market regimes and asset prices:
# - Growth proxies (equities, transports, emerging, small caps)
# - Inflation proxy (CPI from FRED)
# ==========================================================

if menu == "Macroeconomic Indicators":

    # --- Section Title ---
    st.header("Macroeconomic Indicators")

    # --- Tabs for macro indicators ---
    tab_growth, tab_inflation = st.tabs([
        "Growth", 
        "Inflation"
    ])

    # ----------------------------------------------------------
    # Growth Proxies
    # ----------------------------------------------------------
    with tab_growth:
        st.subheader("Growth Proxies")

        # Available growth proxies
        growth_proxies = {
            "S&P500 (Equities Proxy)": "^GSPC",
            "DJT (Dow Jones Transportation)": "DJT",
            "EEM (Emerging Markets ETF)": "EEM",
            "IWM (US Small Caps)": "IWM",
            "XLI (Industrials ETF)": "XLI",
            "XLK (Technology ETF)": "XLK"
        }

        # Multiselect with S&P500 as default
        selected_growth = st.multiselect(
            "Select growth proxies to display",
            options=list(growth_proxies.keys()),
            default=["S&P500 (Equities Proxy)"]
        )

        if selected_growth:
            tickers = [growth_proxies[label] for label in selected_growth]
            df_growth = get_multi_price_data(
                tickers, start="2000-01-01", end=datetime.today(), interval="1d"
            )

            if not df_growth.empty:
                # Date range selector
                min_date, max_date = df_growth.index.min().date(), df_growth.index.max().date()
                start_growth = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                end_growth = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

                # Filter and normalize (base 100)
                df_growth_filtered = df_growth.loc[start_growth:end_growth]
                df_growth_norm = normalize_data(df_growth_filtered.dropna(axis=1, how="all"))

                # Info message
                st.markdown("⚠️ *Curves will not display if the index is not available in the selected period.*")

                if not df_growth_norm.empty:
                    fig_growth = px.line(
                        df_growth_norm,
                        title="Growth Proxies (Normalized Performance)",
                        template=template_choice
                    )
                    fig_growth.update_layout(yaxis_title="Index (Base 100)", xaxis_title="Date")
                    st.plotly_chart(fig_growth, use_container_width=True)

                # Explanation
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

    # ----------------------------------------------------------
    # Inflation Proxy (CPI from FRED)
    # ----------------------------------------------------------
    
    with tab_inflation:
        st.subheader("Inflation Proxy (CPI)")

        try:
            # Download CPI data from FRED
            cpi = web.DataReader("CPIAUCSL", "fred", start="1960-01-01", end=datetime.today())

            # Date range selector
            start_cpi = st.date_input(
                "Start Date",
                datetime.today() - timedelta(days=365)  # default = 1 year back
            )
            end_cpi = st.date_input("End Date", datetime.today())

            cpi_filtered = cpi.loc[(cpi.index >= str(start_cpi)) & (cpi.index <= str(end_cpi))]

            # View mode selection
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
            else:
                cpi_yoy = cpi_filtered.pct_change(periods=12) * 100
                fig_cpi = px.line(
                    cpi_yoy,
                    title=f"US CPI Year-over-Year % Change ({start_cpi} → {end_cpi})",
                    template=template_choice
                )
                fig_cpi.update_layout(yaxis_title="YoY % Change", xaxis_title="Date")

            st.plotly_chart(fig_cpi, use_container_width=True)

            # Interpretation
            st.markdown(
                "**Interpretation:**  \n"
                "- *Index (Base 100)* = long-term inflation trend.  \n"
                "- *YoY % Change* = actual inflation rate.  \n"
                f"Displayed period: **{start_cpi} → {end_cpi}**"
            )

        except Exception as e:
            st.warning(f"Could not load CPI data from FRED: {e}")
        
# ==========================================================
# 5.6 FX & CENTRAL BANKS
#-----------------------------------------------------------
# This section focuses on FX markets and monetary policy:
# - FX Majors (cumulative returns, current levels)
# - Central Bank Policy Rates (Fed, ECB, BoE, BoJ, SNB, BoC)
# ==========================================================

if menu == "FX & Central Banks":

    # --- Section Title ---
    st.header("FX & Central Banks")

    # --- Tabs for FX & Rates ---
    tab_fx, tab_rates = st.tabs([
        "FX Majors",
        "Central Bank Policy Rates"
    ])

    # ----------------------------------------------------------
    # FX Majors
    # ----------------------------------------------------------
    
    with tab_fx:
        st.subheader("FX Majors")

        # Mapping of FX pairs
        FX_TICKERS = {
            "US Dollar Index (DXY)": "DX-Y.NYB",
            "EUR/USD": "EURUSD=X",
            "USD/JPY": "JPY=X",
            "GBP/USD": "GBPUSD=X",
            "USD/CNY": "USDCNY=X"
        }

        # User selection
        selected_fx = st.multiselect(
            "Select FX pairs to display:",
            list(FX_TICKERS.keys()),
            default=["US Dollar Index (DXY)", "EUR/USD"]
        )

        if selected_fx:
            tickers = [FX_TICKERS[pair] for pair in selected_fx]
            fx_data = get_multi_price_data(
                tickers, start="2010-01-01", end=datetime.today(), interval="1d"
            )
            fx_data = check_and_fill_missing(fx_data)

            if not fx_data.empty:
                # --- Cumulative returns ---
                cum_returns = calculate_cumulative_returns(fx_data)
                fig_fx = px.line(
                    cum_returns,
                    title="Cumulative Returns of Selected FX Pairs",
                    template=template_choice
                )
                fig_fx.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date")
                st.plotly_chart(fig_fx, use_container_width=True)

                # --- Current FX levels ---
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
    # Central Bank Policy Rates
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

