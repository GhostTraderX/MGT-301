import numpy as np
import matplotlib.pyplot as plt

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
    S_n = [S0 * (U**j) * (D**(N - j)) for j in range(N + 1)]
    P_t = [h(price) for price in S_n] # This is the payoff function

    # Now we do the Backward induction
    for n in range(N, 0, -1): # Go from N to 0, using a step of -1, meaning we go backwards
        for j in range(n):    # Passing by every branch of the tree
            P_t[j] = coeff * (q * P_t[j + 1] + (1 - q) * P_t[j])

    # We can now compute the replicating portfolio
    b0 = (P_t[1] - P_t[0]) / (S0 * (U - D))  # option_values[1] means stock goes up and [0] stock goes down
    a0 = (U * P_t[0] - D * P_t[1]) / (np.exp(r * delta) * (U - D))
    #a0 = P_t[0] - b0 * S0  # Derived from C0 = a0 + b0 * S0

    return P_t[0], a0, b0

# Code test
C0, a0, b0 = binomial_pricing(S0, k, r, N, delta, U, D, h)

print("Initial price of derivative:", C0)
print("Initial replicating portfolio:", a0, b0)

#========== Question 2 ==========
# Initial values
S0 = 30
r = 0.05
k = 30
T = 1 / 12 # 1 month in years

#To store the results
prices = []
a_val = []
b_val = []

# Loop to get every result + verify if arbitrage free
# Our goal is to change N and verify if it works
for N in range(1, 101):
    delta = T / N
    U = np.exp(0.2 * np.sqrt(delta))
    D = 1 / U

    # Arbitrage-free condition
    if D < np.exp(r * delta) < U:
        C0, a0, b0 = binomial_pricing(S0, k, r, N, delta, U, D, h)
        prices.append(C0)
        a_val.append(a0)
        b_val.append(b0)
    else:
        print("Model with N= ", N, " not arbitrage-free!")

# Plot prices
N_values = range(1, 101)

plt.figure()
plt.plot(N_values, prices)
plt.title("Option prices as function of N")
plt.xlabel("Option price")
plt.ylabel("N")
plt.show()


