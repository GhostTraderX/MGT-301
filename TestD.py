import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from main import risk_free_rate

#importing data
ticker = 'SPY','EWL', 'IEF'
spy_data = pd.read_csv('SPY.csv')['adjClose']
ewl_data = pd.read_csv('EWL.csv')['adjClose']
ief_data = pd.read_csv('IEF.csv')['adjClose']

prices_df = pd.DataFrame({'SPY': spy_data, 'EWL': ewl_data, 'IEF': ief_data})
returns_df = prices_df.pct_change()[1:]

# return vector and covariance matrix
r = ((1+returns_df).prod())**(52/len(returns_df))
cov = returns_df.cov()*52
e = np.ones(len(r))

print(cov)

# Variance portfolio
def variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Std portfolio
def standard_deviation(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

# Expected return portfolio
def expected_return(weights, returns):
    return np.sum(returns.mean() * weights)*52

# Sharpe Ratio portfolio
def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, returns) - risk_free_rate)/ standard_deviation(weights, cov_matrix)

risk_free_rate = 0.015

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1}
bounds =[(0, 1) for _ in range(len(ticker))] # putting the max

# Set initial weights (all the same)
initial_weights = np.array([1/len(ticker)]*len(ticker))
print(initial_weights)

#Optimized results
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(returns_df, cov, risk_free_rate), method='SLSQP', constraints=constraints)

# Optimized weights
optimal_weights = optimized_results.x

# Display everything
print(optimal_weights)

optimal_portfolio_return = expected_return(optimal_weights, returns_df)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, returns_df, cov, risk_free_rate)
optimal_variance = variance(optimal_weights, cov)

print(f"Expected Annual Return: {optimal_portfolio_return: .4f}")
print(f"Expected Variance: {optimal_variance: .4f}")
print (f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")
