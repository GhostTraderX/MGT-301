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
print('\n \n')


# For question 5, we want to minimize variance while gettint 12% annual return
target_return = 0.12

def minimize_variance(weights, cov_matrix):
    return variance(weights, cov_matrix)

def minimize_volatility(weights, cov_matrix):
    return standard_deviation(weights, cov_matrix)

# We update the constraints
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Sum = 1
    {'type': 'eq', 'fun': lambda weights: expected_return(weights, returns_df) -  target_return},
]

# We optimize for minimum -SR with the new constraints
optimized_target_return = minimize(minimize_volatility, optimal_weights, args= cov,
                                   method='SLSQP', constraints=constraints)

if optimized_target_return.success:
    target_return_weights = optimized_target_return.x
else:
    print("Optimization failed.")

# Gettint the things we wanted
target_portfolio_return = expected_return(target_return_weights, r)
target_portfolio_volatility = standard_deviation(target_return_weights, cov)
target_portfolio_variance = variance(target_return_weights, cov)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, returns_df, cov, risk_free_rate)


print(f"Optimal Weights: {target_return_weights}")
print(f"Expected Annual Return: {target_portfolio_return:.4f}")
print(f"Portfolio Variance: {target_portfolio_variance:.4f}")
print(f"Portfolio Volatility: {target_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")