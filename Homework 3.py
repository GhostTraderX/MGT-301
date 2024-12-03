import numpy as np

# ======== Question 1 =======
S0 = 1              # Initial risky price
r = 0.03            # Riskfree rate
N = 100             # Number of periods
delta = 1 / N       # Lenght of periods
U = np.exp(0.02)    # Up factor
D = 1 / U           # Down factor

def h(a):
    return 1

def payoff(k, h, S0, r, N, delta, U, D, q_N, n):
    if k > n > N - 1:
        return h(1)
    else:
        return q_N * (payoff(k+1, h, S0, r, N, delta, U, D, q_N, n+1) / (np.exp(r * delta))) + (1 - q_N) * payoff(k, h, S0, r, N, delta, U, D, q_N, n+1) / (np.exp(r * delta))

print(payoff(1, h, S0, r, N, delta, U, D, 0.5, 1))