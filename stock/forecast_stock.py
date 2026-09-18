import numpy as np
import matplotlib.pyplot as plt

# ------ Simulating a stock value over one year ------

# Parameters
S0 = 10000          # Starting price
mu = 0.1          # Annual expected return (10%)
sigma = 0.3       # Annual volatility (30%)
n_years = 10000   # Number of simulated year trials

# 2. Generating the Random Variables (Log-Returns)
# Total number of "events" (trading days across all simulated year trials)
total_days = n_years * 365

# Calculate daily drift and volatility based on Geometric Brownian Motion
dt = 1 / 365
daily_mu = (mu - 0.5 * sigma**2) * dt #Ito correction for drift
daily_sigma = sigma * np.sqrt(dt)

# Generate all random daily returns at once (similar to U and E)
# Daily returns follow a normal distribution, mean = daily_mu, std = daily_sigma, need total_days of them
r = np.random.normal(loc=daily_mu, scale=daily_sigma, size=total_days)


indices = np.arange(0, total_days, 365)
yearly_log_returns = np.add.reduceat(r, indices)

# Calculate ending prices: S_T = S_0 * e^(sum of log returns)
ending_prices = S0 * np.exp(yearly_log_returns)

# Visualization
plt.figure(figsize=(8,4))
plt.hist(ending_prices, bins=100, color='skyblue', edgecolor='black')
plt.xlabel('Stock Price after 1 Year', fontsize=14)
plt.ylabel('# Occurrences', fontsize=14)
plt.title('Stock Price Distribution after 1 Year (100,000 Simulations)', fontsize=18)
plt.show()

prob_profit = np.sum(ending_prices > S0) / len(ending_prices)
expected_val = np.mean(ending_prices)

print("Probability of making a profit: " + str(prob_profit*100))
print("Expected stock price: " + str(expected_val))

# ------
