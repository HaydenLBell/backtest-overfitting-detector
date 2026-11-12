import numpy as np
from scipy.stats import norm


def probabilistic_sharpe_ratio(sr, sr_benchmark, gamma3, gamma4, num_days):
    """
    sr           -- observed DAILY Sharpe ratio (not annualised)
    sr_benchmark -- the Sharpe to beat, SR* (also daily)
    gamma3       -- skewness
    gamma4       -- raw kurtosis (normal = 3)
    num_days     -- number of returns, T
    """
    numerator = (sr - sr_benchmark) * np.sqrt(num_days - 1)
    denominator = np.sqrt(1 - gamma3 * sr + (gamma4 - 1) / 4 * sr ** 2)

    z = numerator / denominator
    # function phi ()
    return norm.cdf(z)



daily_sr = 1 / np.sqrt(252)    # annual 1.0 converted to daily
print(probabilistic_sharpe_ratio(daily_sr, 0, 0, 3, 2520))   # about 0.84