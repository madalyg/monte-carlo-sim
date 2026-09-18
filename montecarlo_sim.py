import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp

x = np.linspace(0,3,100)
f = 2*np.exp(-2*x)
F = 1-np.exp(-2*x)

Us = np.random.rand(10000)

F_inv_Us = -np.log(1-Us)/2

#Solve for the inverse CDF
x, y, F1, F2, E1, E2 = smp.symbols('x y F_1 F_2 E_1 E_2', real=True, positive=True)
fs = F1*smp.exp(-smp.sqrt(x/E1)) + F2*smp.exp(-smp.sqrt(x/E2))
fs

Fs = smp.integrate(fs, (x,0,y)).doit()
Fs

Fn = smp.lambdify((y, E1, E2, F1, F2), Fs)
fn = smp.lambdify((x, E1, E2, F1, F2), fs)

# ------ Test the inverse CDF method by simulating 10,000 energies from the distribution defined by f(x) and F(x) ------
E1 = E2 = 0.2
F1 = 1.3
F2 = 1.4
x = np.linspace(0,5,1000) # Create a "look-up table" by calculating the CDF for a thousand specific energy values
f = fn(x, E1, E2, F1, F2)
F = Fn(x, E1, E2, F1, F2)

F_inv_Us = x[np.searchsorted(F[:-1], Us)] # Looks through the sorted list of CDF values for a random number U and finds 
# the corresponding energy value in x

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.hist(F_inv_Us, histtype='step', color='red', density='norm', bins=100, label='$F^{-1}(u)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
plt.xlim(0,2)
plt.show()

# ------

# ------ Simulating 100,000 10-second experiments as 100,000 samples from a Poisson distribution ------
N = 100000
X = np.random.poisson(lam=4, size=N)

x = np.linspace(0,5,1000)
F = Fn(x, E1, E2, F1, F2)
Us = np.random.rand(X.sum())
E = x[np.searchsorted(F[:-1], Us)]

# Sum of how many particles are detected total after all N experiments
idx = np.insert(X.cumsum(), 0, 0)[:-1]
idx[0:10]

# Sum the energies of the particles detected in each experiment
E_10s_sum = np.add.reduceat(E, idx)

# Plot the distribution of energies detected in all of the 10-second experiments
plt.figure(figsize=(5,3))
plt.hist(E_10s_sum, bins=100)
plt.xlabel('Energy [GeV]', fontsize=20)
plt.ylabel('# Occurences')
plt.show()

# ------


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

