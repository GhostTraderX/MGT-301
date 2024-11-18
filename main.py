import math
import pandas as pd


risk_free_rate = 0.015

# Import data
spy_data = pd.read_csv('SPY.csv')['adjClose']
ewl_data = pd.read_csv('EWL.csv')['adjClose']
ief_data = pd.read_csv('IEF.csv')['adjClose']
df = pd.DataFrame({'SPY': spy_data, 'EWL': ewl_data, 'IEF':ief_data})

# Calculate annualized expected returns (aer)
def get_annualized_expected_returns(data):
    pct_change = data.pct_change().dropna()
    mean = pct_change.mean()
    return mean * 52

spy_aer = get_annualized_expected_returns(spy_data)
ewl_aer = get_annualized_expected_returns(ewl_data)
ief_aer = get_annualized_expected_returns(ief_data)

# Calculate annualized volatility
def get_annualized_volatility(data):
    pct_change = data.pct_change().dropna()
    var = pct_change.var()
    return math.sqrt(var * 52)

spy_vol = get_annualized_volatility(spy_data)
ewl_vol = get_annualized_volatility(ewl_data)
ief_vol = get_annualized_volatility(ief_data)

# calculate pairwise correlation
spy_ewl_corr = df.corr().loc['SPY', 'EWL']
spy_ief_corr = df.corr().loc['SPY', 'IEF']
ewl_ief_corr = df.corr().loc['EWL', 'IEF']
