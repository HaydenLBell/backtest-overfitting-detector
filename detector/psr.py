import numpy as np
from scipy.stats import norm
from detector.stats import sharpe_ratio, skewness, kurtosis_raw
from detector.synthetic import daily_returns


EULER_GAMMA = 0.5772156649

# confidence percentage of sorts
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
#print(probabilistic_sharpe_ratio(daily_sr, 0, 0, 3, 2520))   # about 0.8


def expected_max_sharpe(sharpes, num_trials=None):
    """
    Estimate the Sharpe ratio the best of N strategies would get from luck alone.

    sharpes    -- array of DAILY Sharpe ratios, one per strategy tried
    num_trials -- N, number of strategies tried (defaults to len(sharpes))
    """
    # remove any nan values (strategies with zero volatility)
    valid_sharpes = sharpes[~np.isnan(sharpes)]

    # N: how many strategies were tried
    if num_trials is None:
        num_trials = len(valid_sharpes)

    # Part A: width of the luck curve
    width = np.std(valid_sharpes, ddof=1)

    # Part B: how many widths out the best of N lands
    guess_1 = norm.ppf(1 - 1 / num_trials)
    guess_2 = norm.ppf(1 - 1 / (num_trials * np.e))
    widths_out = (1 - EULER_GAMMA) * guess_1 + EULER_GAMMA * guess_2

    return width * widths_out


def deflated_sharpe_ratio(returns, num_trials=None):

    num_days = returns.shape[0]

    # daily Sharpe for every strategy
    all_sharpes = sharpe_ratio(returns)

    # find the index with the best strategy
    best_index = np.nanargmax(all_sharpes)

    # get the sharpe at position of best strat index
    best_sharpe = all_sharpes[best_index]

    # grab skew and kurtosis of the best strategy's returns (code means all days, but only w the best index)
    best_returns = returns[:, best_index]
    gamma3 = skewness(best_returns)
    gamma4 = kurtosis_raw(best_returns)

    # he Sharpe pure luck would give the best of N (expectimax)
    sr_star = expected_max_sharpe(all_sharpes, num_trials)

    # PSR using the luck benchmark
    dsr = probabilistic_sharpe_ratio(best_sharpe, sr_star, gamma3, gamma4, num_days)

    return dsr, best_index




returns = daily_returns(252, 1000, seed=42)
dsr, best = deflated_sharpe_ratio(returns)
print("Best strategy:", best)
print("DSR:", round(dsr, 3))