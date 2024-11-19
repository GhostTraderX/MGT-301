import pandas as pd
import numpy as np

risk_free_rate = 0.015
ones = np.ones(3)

# Import data
spy_data = pd.read_csv('SPY.csv')['adjClose']
ewl_data = pd.read_csv('EWL.csv')['adjClose']
ief_data = pd.read_csv('IEF.csv')['adjClose']
weekly_returns = pd.DataFrame({'SPY': spy_data, 'EWL': ewl_data, 'IEF':ief_data}).pct_change().dropna()

# ====== QUESTION 2 ======

# Calculate annualized expected returns (aer)
def get_annualized_expected_returns(data):
    pct_change = data.pct_change().dropna()
    mean = pct_change.mean()
    return mean * 52

spy_aer = get_annualized_expected_returns(spy_data)
print('SPY Annualized Expected Returns: ', spy_aer.round(3))
ewl_aer = get_annualized_expected_returns(ewl_data)
print('EWL Annualized Expected Returns: ', ewl_aer.round(3))
ief_aer = get_annualized_expected_returns(ief_data)
print('IEF Annualized Expected Returns: ', ief_aer.round(3))
print()

# Calculate annualized volatility
def get_annualized_volatility(data):
    pct_change = data.pct_change().dropna()
    var = pct_change.var()
    return np.sqrt(var * 52)

spy_vol = get_annualized_volatility(spy_data)
print('SPY Annualized Volatility: ', spy_vol.round(3))
ewl_vol = get_annualized_volatility(ewl_data)
print('EWL Annualized Volatility: ', ewl_vol.round(3))
ief_vol = get_annualized_volatility(ief_data)
print('IEF Annualized Volatility: ', ief_vol.round(3))
print()

vol = np.array([spy_vol, ewl_vol, ief_vol])


# calculate pairwise correlation
corr_matrix = weekly_returns.corr()
spy_ewl_corr = corr_matrix.loc['SPY', 'EWL']
print('SPY/EWL Correlation: ', spy_ewl_corr.round(3))
spy_ief_corr = corr_matrix.loc['SPY', 'IEF']
print('SPY/IEF Correlation: ', spy_ief_corr.round(3))
ewl_ief_corr = corr_matrix.loc['EWL', 'IEF']
print('EWL/IEF Correlation: ', ewl_ief_corr.round(3))
print()

# ====== QUESTION 3 ======

# Calculate Cov Matrix = Corr Matrix * Volatility Vector
vols = np.array([[spy_vol, ewl_vol, ief_vol]])
vols_matrix = vols.T @ vols

Sigma = corr_matrix * vols_matrix
Sigma_inv = np.linalg.inv(Sigma)
print("Simga matrix :\n", Sigma.round(3))
print()

# Returns vectors
mu = np.array([[spy_aer], [ewl_aer], [ief_aer]])
R0 = np.array([[risk_free_rate], [risk_free_rate], [risk_free_rate]])

excess_returns = mu - R0

# Risk aversion coefficient
a = 1

risky_weights = np.linalg.inv(a * Sigma) @ excess_returns
rf_weight = 1 - ones @ risky_weights

C = ones @ Sigma_inv @ excess_returns

portfolio_weights = 1 / C * (Sigma_inv @ excess_returns)
print(portfolio_weights)