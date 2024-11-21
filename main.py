import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

risk_free_rate = 0.015
ones = np.ones(3)

# Define the tickers and date range
tickers = ["SPY", "EWL", "IEF"]
start_date = "2012-12-31"
end_date = "2024-06-30"

# Download the data with weekly frequency and adjust for dividends
data = yf.download(tickers, start=start_date, end=end_date, interval="1wk")["Adj Close"]

# Import data
spy_data = data['SPY']
ewl_data = data['EWL']
ief_data = data['IEF']
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

# Calculate Tangency Portfolio Weights
risky_weights = np.linalg.inv(a * Sigma) @ excess_returns
rf_weight = 1 - ones @ risky_weights

C = ones @ Sigma_inv @ excess_returns

portfolio_weights = 1 / C * (Sigma_inv @ excess_returns)
print(f'Tangency Portfolio Weights\n SPY: {portfolio_weights[0][0].round(3)}\n EWL: {portfolio_weights[1][0].round(3)}\n IEF: {portfolio_weights[2][0].round(3)}')

# Portfolio return
portfolio_return = portfolio_weights.T @ mu
print(f"Tangency Portfolio Return: {portfolio_return[0][0].round(3)}")

# Portfolio variance and std
portfolio_var = portfolio_weights.T @ Sigma @ portfolio_weights
print(f"Tangency Portfolio Variance: {portfolio_var.iloc[0, 0].round(3)}")
portfolio_std = np.sqrt(portfolio_var)
print("Portfolio Standard Deviation: ", portfolio_std.iloc[0, 0].round(3))

# Portfolio Sharpe Ratio
portfolio_shape_ratio = np.sqrt(excess_returns.T @ Sigma_inv @ excess_returns)
print("Portfolio Sharpe Ratio:", portfolio_shape_ratio[0][0].round(3))
print()

# ====== Question 4 ======
def generate_three_numbers_sum_to_one():
    weight_1 = np.random.uniform(0, 1)
    weight_2 = np.random.uniform(0, 1)
    weight_3 = 1 - weight_1 - weight_2

    return np.array([[weight_1], [weight_2], [weight_3]])


returns = []
std_dev = []

for i in range(1000):
    weights = generate_three_numbers_sum_to_one()
    returns.append((weights.T @ mu)[0][0])
    std_dev.append(np.sqrt((weights.T @ Sigma @ weights).iloc[0, 0]))


portfolios = zip(std_dev, returns)
plt.figure(dpi=600)
plt.title("Efficient frontier")
plt.xlabel("Standard deviation")
plt.ylabel("Mean return")
plt.scatter(std_dev, returns, s=0.5)
# risk_free_rate + portfolio_shape_ratio[0][0] * x

plt.show()

# ====== Question 5 ======

# Target return
Ra = 0.12

C = a * (Ra - R0) / (portfolio_return - R0)

target_weights = C / a * portfolio_weights
print(f'Target Weights\n SPY: {target_weights[0][0].round(3)}\n EWL: {target_weights[1][0].round(3)}\n IEF: {target_weights[2][0].round(3)}')
print()

# Calculate implied risk aversion
gamma = (portfolio_return[0][0] - risk_free_rate) / (portfolio_var.iloc[0, 0])
print("Implied risk aversion coefficient:", gamma.round(3))