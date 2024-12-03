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
print("Sigma matrix :\n", Sigma.round(3))
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

def calculate_portfolio_stats(weights, annualized_return, cov_matrix_annual):
    portfolio_mean = np.dot(weights, annualized_return)
    portfolio_variance = weights.T @ cov_matrix_annual @ weights
    return portfolio_mean, portfolio_variance


def generate_efficient_frontier(annualized_return, cov_matrix_annual, num_points, risk_free_rate):
    weights_spy = np.linspace(-1, 2, num_points)
    frontier_means, frontier_variances = [], []

    for w_spy in weights_spy:
        for w_ewl in np.linspace(-1, 2 - w_spy, num_points):
            w_ief = 1 - w_spy - w_ewl
            weights = np.array([w_spy, w_ewl, w_ief])
            portfolio_mean, portfolio_variance = calculate_portfolio_stats(weights, annualized_return,
                                                                           cov_matrix_annual)
            frontier_means.append(portfolio_mean)
            frontier_variances.append(portfolio_variance)

    frontier_means = np.array(frontier_means)
    frontier_std_devs = np.sqrt(frontier_variances)
    sorted_indices = np.argsort(frontier_std_devs)
    frontier_means = frontier_means[sorted_indices]
    frontier_std_devs = frontier_std_devs[sorted_indices]

    efficient_means, efficient_std_devs = [], []
    for std, mean in zip(frontier_std_devs, frontier_means):
        if len(efficient_means) == 0 or mean > efficient_means[-1]:
            efficient_means.append(mean)
            efficient_std_devs.append(std)

    return np.array(efficient_means), np.array(efficient_std_devs)


def calculate_tangency_portfolio(efficient_means, efficient_std_devs, risk_free_rate):
    sharpe_ratios = (efficient_means - risk_free_rate) / efficient_std_devs
    max_sharpe_idx = np.argmax(sharpe_ratios)
    return efficient_means[max_sharpe_idx], efficient_std_devs[max_sharpe_idx], sharpe_ratios[max_sharpe_idx]


def plot_efficient_frontier_and_cml(efficient_means, efficient_std_devs, tangency_mean, tangency_std_dev,
                                    risk_free_rate, sharpe_ratio):
    slope = sharpe_ratio
    weights_cml = np.linspace(0, 2, 200)
    cml_means = risk_free_rate + weights_cml * slope * tangency_std_dev
    cml_std_devs = weights_cml * tangency_std_dev


    plt.figure(dpi=600)
    plt.plot(efficient_std_devs, efficient_means, 'k--', label='Efficient frontier')
    plt.plot(cml_std_devs, cml_means, 'b-', label='Capital market line')
    plt.scatter(tangency_std_dev, tangency_mean, color='orange', label='Tangency portfolio')
    plt.xlabel("Standard Deviation")
    plt.ylabel("Expected Return")
    plt.title("Mean Variance Efficient Frontier")
    plt.legend()
    plt.show()


num_points = 300
annualized_return = weekly_returns.mean() * 52
efficient_means, efficient_std_devs = generate_efficient_frontier(annualized_return, Sigma, num_points,
                                                                  risk_free_rate)
tangency_mean, tangency_std_dev, sharpe_ratio = calculate_tangency_portfolio(efficient_means, efficient_std_devs,
                                                                             risk_free_rate)
plot_efficient_frontier_and_cml(efficient_means, efficient_std_devs, tangency_mean, tangency_std_dev,
                                risk_free_rate, sharpe_ratio)

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

# ====== Question 6 ======

# Calculate volatility
target_var = target_weights.T @ Sigma @ target_weights
print(f"Target Portfolio Variance: {target_var.iloc[0, 0].round(3)}")
target_std = np.sqrt(target_var)
print("Target Portfolio Standard Deviation: ", target_std.iloc[0, 0].round(3))

# Calculate Sharpe Ratio
target_sharpe_ratio = (Ra - risk_free_rate)/target_std
print("Target Portfolio Sharpe Ratio:", target_sharpe_ratio[0][0].round(3))
