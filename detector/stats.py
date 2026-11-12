import numpy as np
from scipy.stats import skew, kurtosis

test = np.array([
    [0.02,  0.01],
    [-0.01, 0.01],
    [0.03,  0.01],
    [0.00,  0.01],
])


########################
###   SHARPE RATIO   ###
########################

# HELPERS

# currently using np.mean(axis=0) over custom function as its faster, will reimplement in C++ later
def mean_per_column(returns):
    num_days = returns.shape[0]
    num_strategies = returns.shape[1]

    means = []
    for strategy in range(num_strategies):
        total = 0
        for day in range(num_days):
            total = total + returns[day][strategy]
        means.append(total / num_days)

    return np.array(means)

# SR = r_bar/sigma_hat
def sharpe_ratio(returns, annualise=False):

    #mean_returns = mean_per_column(returns)
    # get the mean of the returns 1/T Sum(rt)
    mean_returns = returns.mean(axis=0)

    # get the standard deviation of the returns
    std_returns = returns.std(axis=0, ddof=1)


    # sharpe ratio equals  create new array of (number of strategies, NaN) (gives us NaN * num strats)
    sr = np.full(len(mean_returns), np.nan)

    # essentially, did this strategy's returns change at all from day to day?
    has_volatility = std_returns > 0

    # basically a mask for those with volatility (std)
    sr[has_volatility] = mean_returns[has_volatility] / std_returns[has_volatility]
    
    if annualise:
        sr = sr * np.sqrt(252) # about 252 trading days in a year

    return sr

    


def skewness(returns):
    skew(returns, axis=0)

def kurtosis(returns):
    kurtosis(returns, axis=0, fisher=False) # fisher false makes a normal dist give 3 not 0, fisher true would be 3-3=0

#print(sharpe_ratio(test))