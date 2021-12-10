# Instructions
* Store raw data in a folder named 'data'

* Run compute_bsm.py
Takes in raw CBOE options price data from 2006-2021, separates calls from puts, tabulates quote date, option price, underlying price, strike price, calculates risk-free rate from US Treasury yield data, time to maturity to the nearest minute, and implied volatility using Brent's method root finding using BSM option pricing equation.
The output of BSM parameters calculations are, by default stored as .feather files, 'calls_bsm.feather' and 'puts_bsm.feather'

* Create folders 'poly_res', 'cubic_res' and 'svi_res' to store surface fitting results.
* Run surface_fitting.ipynb
Fits a surface to the implied volatilities of each day, using 3 models: SVI, surface polynomial and cubic polynomial.
Stores the output object as pickle files, one file for each model for each trading day.

* Create folder 'plots' to store plotting results
* Run kalman_filtering.ipynb

The code base also includes 'utils' folder which has two modules: 'bsm' and 'ivs'.
Module 'bsm' contains functions to calculate Black-Scholes-Merton parameters.
Module 'ivs' contains classes and methods to fit and evaluate an IVS parametric model.
