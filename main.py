import math
from operator import length_hint

import pandas as pd
import numpy as np


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

sigma = df.cov()
sigma_inv = np.linalg.inv(sigma)

mu = np.array([[spy_aer], [ewl_aer], [ief_aer]])

a = 1
mu_rf = (mu - risk_free_rate * np.ones((len(mu), 1)))
print('mu_Ref', mu_rf.shape)

weights = np.linalg.inv(a * sigma) @ mu_rf

w_0 = 1 - (np.ones(len(mu)) @ weights)

C = np.ones(len(mu)) @ sigma_inv @ mu_rf
w_tan = 1/C * sigma_inv @ mu_rf

sharpe_ratio = np.sqrt(mu_rf.T @ sigma_inv @ mu_rf)
print(sharpe_ratio)
