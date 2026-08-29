
import pandas as pd
import yfinance as yf
from datetime import datetime

symbols = [
    "USDJPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "NZDJPY=X", "CADJPY=X",
    "USDCAD=X", "GBPUSD=X", "EURUSD=X", "AUDUSD=X", "NZDUSD=X", "EURGBP=X",
    "USDCHF=X", "EURCAD=X", "SEKJPY=X", "SGDJPY=X", "EURAUD=X", "GBPCAD=X",
    "EURNZD=X", "CADCHF=X", "GBPCHF=X", "AUDCAD=X", "AUDCHF=X", "CHFJPY=X",
    "NZDCHF=X", "ZARJPY=X", "EURNOK=X", "EURCHF=X", "CHFSEK=X", "GBPNOK=X", "NZDNOK=X"
]

end_date = datetime.now()
start_date = end_date - pd.DateOffset(years=1)
raw_data = yf.download(symbols, start=start_date, end=end_date, progress=False)
data_close = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']

for col in data_close.columns:
    col_name = col[-1] if isinstance(col, tuple) else col
    last_val = data_close[col].iloc[-1]
    missing = data_close[col].isna().sum()
    if pd.isna(last_val) or missing > 5:
        print(f"Pair: {col_name} | Last: {last_val} | Missing: {missing}")
