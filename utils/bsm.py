import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from scipy.optimize import brentq

print("Loading risk-free rate data...", end='\r')
rfr = pd.read_csv('data/risk_free_rate.csv', index_col=0)
rfr.index = pd.to_datetime(rfr.index, format='%d-%m-%y')
rfr.columns = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
rfr /= 100
rfr['cs'] = rfr.apply(lambda x: CubicSpline(x.dropna().index, x.dropna()), axis=1)
print("Risk-free rate data loaded successfully")

def bsm_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + sigma**2/2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def calc_iv_call(row, verbose=False):
    if verbose:
        print(row.name, end='\r')
    fun = lambda sigma, C, S, K, r, t: C - bsm_call(S, K, sigma, r, t)
    try:
        return brentq(fun, 1e-6, 1e2, args=(row['C'], row['S'], row['K'], row['r'], row['t']))
    except:
        return np.nan

def bsm_put(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + sigma**2/2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return K * np.exp(-r*t) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calc_iv_put(row, verbose=False):
    if verbose:
        print(row.name, end='\r')
    fun = lambda sigma, P, S, K, r, t: P - bsm_put(S, K, sigma, r, t)
    try:
        return brentq(fun, 1e-6, 1e2, args=(row['P'], row['S'], row['K'], row['r'], row['t']))
    except:
        return np.nan

def get_time_to_maturity(row):
    '''
        * Prices are quoted at close, i.e. 3.45pm CST
        * If ticker is SPX, option is AM-settled, i.e. 9.30am CST
        * If ticker is SPXW, option is PM-settled, i.e. 4.00pm CST
        * Otherwise, assume PM-settled
    '''
    current_time = pd.to_datetime(row.quote_date).replace(hour=15, minute=45)
    
    if row.root == 'SPX':
        expiry_time = pd.to_datetime(row.expiration).replace(hour=9, minute=30)
    else:
        expiry_time = pd.to_datetime(row.expiration).replace(hour=16)
        
    timedelta = expiry_time - current_time
    
    return timedelta.days / 365 + timedelta.seconds / 365 / 24 / 3600

def get_risk_free_rate(row, rfr=rfr):
    '''
        * Use treasury yield curve for risk-free rate. 
        * Use cubic spline interpolation to calculate the risk-free rate depending on time-to-maturity
    '''
    try:
        date = datetime.strptime(row.quote_date, '%d-%m-%y')
    except:
        date = datetime.strptime(row.quote_date, '%Y-%m-%d')
    while date not in rfr.index:
        date -= timedelta(days=1)
    return rfr.loc[date, 'cs'](row.time_to_maturity).item()

def get_parameters_df(year, data_folder=None, verbose=False):
    assert data_folder is not None
    
    if verbose:
        print(f"Calculating Black-Scholes-Merton parameters for SPX options in {year}")
    
    df = pd.read_feather(f'{data_folder}/spx{year}.feather')
    
    df['underlying'] = (df['underlying_bid_1545'] + df['underlying_ask_1545']) / 2
    df['option_price'] = (df['bid_1545'] + df['ask_1545']) / 2
    df['time_to_maturity'] = df.apply(get_time_to_maturity, axis=1)
    df = df[df.time_to_maturity > 0]
    df['risk_free_rate'] = df.apply(get_risk_free_rate, axis=1)
    
    call_df = df[df.option_type == 'C'][
        ['quote_date','option_price', 'underlying', 'strike', 'risk_free_rate', 'time_to_maturity']]
    call_df.columns = ['date','C', 'S', 'K', 'r', 't']
    call_df.reset_index(drop=True, inplace=True)

    put_df = df[df.option_type == 'P'][
        ['quote_date','option_price', 'underlying', 'strike', 'risk_free_rate', 'time_to_maturity']]
    put_df.columns = ['date','P', 'S', 'K', 'r', 't']
    put_df.reset_index(drop=True, inplace=True)
    
    if verbose:
        print(f"Calculating implied volatility of calls, num. of data points = {len(call_df)}")
    call_df['sigma'] = call_df.apply(lambda row: calc_iv_call(row, verbose), axis=1)
    if verbose:
        print(f"Calculating implied volatility of puts, num. of data points = {len(put_df)}")
    put_df['sigma'] = put_df.apply(lambda row: calc_iv_put(row, verbose), axis=1)
    
    call_df = call_df.dropna().reset_index(drop=True)
    put_df = put_df.dropna().reset_index(drop=True)
    
    return call_df, put_df