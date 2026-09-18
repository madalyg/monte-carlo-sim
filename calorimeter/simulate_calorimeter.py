import numpy as np
import matplotlib.pyplot as plt
import sympy as smp

x = np.linspace(0,3,100)
f = 2*np.exp(-2*x)
F = 1-np.exp(-2*x)

Us = np.random.rand(10000)

F_inv_Us = -np.log(1-Us)/2

# Solve for the inverse CDF
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
