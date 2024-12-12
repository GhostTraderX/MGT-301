import numpy as np

# ======== Question 1 =======
S0 = 1              # Initial risky price
k = 1               # Strike price
r = 0.03            # Riskfree rate
N = 100             # Number of periods
delta = 1 / N       # Lenght of periods
U = np.exp(0.02)    # Up factor
D = 1 / U           # Down factor

# We are using a European call option, which means:
def h(S, K = 1):
    return max(S - K, 0)

def binomial_pricing(S0, k, r, N, delta, U, D, h):

    # We start by computing important constants
    q = (np.exp(r * delta) - D) / (U - D)
    coeff = 1 / np.exp(r * delta)

    # We start the algorithm at the end, so we have to create the array to receive the whole tree
    stock_prices = np.array([S0 * (U**j) * (D**(N - j)) for j in range(N + 1)])
    P_t = np.array([h(price) for price in stock_prices]) # This is the payoff function

    # Now we do the Backward induction
    for n in range(N, 0, -1): # Go from N to 0, using a step of -1, meaning we go backwards
        for j in range(n):    # Passing by every branch of the tree
            P_t[j] = coeff * (q * P_t[j + 1] + (1 - q) * P_t[j])

    # We can now compute the replicating portfolio
    b0 = (P_t[1] - P_t[0]) / (S0 * (U - D))  # option_values[1] means stock goes up and [0] stock goes down
    a0 = (U * P_t[0] - D * P_t[1]) / (np.exp(r * delta) * (U - D))
    #a0 = P_t[0] - b0 * S0  # Derived from C0 = a0 + b0 * S0

    return P_t[0], (a0, b0)

# Code test
C0, (a0, b0) = binomial_pricing(S0, k, r, N, delta, U, D, h)

print("Initial price of derivative:", C0)
print("Initial replicating portfolio:", (a0, b0))