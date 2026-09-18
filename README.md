# Monte Carlo simulations

Monte Carlo scripts for various applications. NumPy drives the sampling, Matplotlib used for histograms.

## Layout

```
calorimeter/simulate_calorimeter.py   Particle-energy calorimeter model
stock/forecast_stock.py               One-year stock price (GBM)
montecarlo_sim.py                     Calorimeter + stock in one script
requirements.txt
```

## Setup

Python 3 with pip.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `sympy`, `scipy` (see `requirements.txt`).

## Run

From the repo root:

```bash
python calorimeter/simulate_calorimeter.py
python stock/forecast_stock.py
python montecarlo_sim.py
```

Each script blocks on `plt.show()` until you close the figure windows. The stock script also prints profit probability and mean ending price to stdout.

## Calorimeter (`calorimeter/simulate_calorimeter.py`)

1. **Inverse CDF check** — Samples 10,000 uniform draws and maps them through a discrete inverse CDF (lookup via `searchsorted`) for a two-component energy PDF built in SymPy.
2. **10 s experiments** — Draws 100,000 Poisson counts (`λ=4`), samples one energy per detected particle from the same CDF, and histograms total energy per experiment.

Parameters (`E1`, `E2`, `F1`, `F2`, grid size, `N`) are set in the script.

## Stock (`stock/forecast_stock.py`)

Simulates `n_years` independent one-year paths with geometric Brownian motion: 365 daily log-returns per path, Ito-adjusted drift, vectorized with `numpy.add.reduceat`. Default inputs: `S0=10000`, `μ=0.1`, `σ=0.3`, `n_years=10000`. Output is a histogram of terminal prices plus summary statistics.

## Combined script (`montecarlo_sim.py`)

Same logic as the two folder scripts, run back-to-back. This was my initial source code for my Computational Theoretical Modelling in Physics class. Use the indepdent scripts in each folder if you only need one model.
