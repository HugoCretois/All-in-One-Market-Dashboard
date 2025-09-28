\# 📊 All-in-One Asset Dashboard  



This project is a \*\*training exercise in Python programming for finance\*\*, built with \*\*Streamlit\*\*.  

It provides an interactive dashboard to explore \*\*financial markets, macroeconomic indicators, and central banks data\*\*.  



All-in-One-Asset-Dashboard/

│── Dashboard.py          # Main Streamlit app

│── requirements.txt      # Dependencies

│── README.md             # Project description



\## 🚀 Features  



\- \*\*Market Performance\*\*  

&nbsp; - Cumulative returns  

&nbsp; - Rolling volatility  

&nbsp; - Correlation heatmap  

&nbsp; - Pairwise correlation chart  

&nbsp; - Portfolio simulation  



\- \*\*Economic Situation (Regime Monitor)\*\*  

&nbsp; - Breadth indicator (S\&P500)  

&nbsp; - Credit spread (HYG vs LQD)  

&nbsp; - Yield curve (10Y–3M)  

&nbsp; - Volatility regime (VIX)  

&nbsp; - USD strength (DXY)  

&nbsp; - Global market regime score  



\- \*\*Macroeconomic Indicators\*\*  

&nbsp; - Growth proxies (equities, transports, EM, small caps, technology, industrials)  

&nbsp; - Inflation proxy (US CPI – index \& YoY change from FRED)  



\- \*\*FX \& Central Banks\*\*  

&nbsp; - FX majors performance  

&nbsp; - Central bank policy rates (Fed, ECB, BoE, BoJ, SNB, BoC)  





\## 🛠 Installation  



\### 1. Clone the repository  

```bash

git clone https://github.com/your-username/All-in-One-Asset-Dashboard.git

cd All-in-One-Asset-Dashboard





\### 2. Create a virtual environment (recommended)

python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\\Scripts\\activate      # Windows



\### 3. Install dependencies



Make sure you have Python 3.9+ installed.

Then run:



pip install -r requirements.txt



Run the dashboard with:



streamlit run Dashboard.py

