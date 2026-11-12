from detector.stats import sharpe_ratio
from detector.synthetic import daily_returns

import numpy as np
import matplotlib.pyplot as plt




def histogram_sharpe(sharpes):
    plt.hist(sharpes, bins=50)
    plt.axvline(np.nanmax(sharpes), color="red")
    plt.xlabel("Annualised Sharpe ratio")
    plt.ylabel("Number of strategies")
    plt.show()




num_days = 2520          # 1 year of data
num_strategies = 1000   # number of strategies "tried"

returns = daily_returns(num_days, num_strategies, seed=42)
sharpes = sharpe_ratio(returns, annualise=True)

print("Strategies tested:", num_strategies)
print("Days of data:", num_days)
print("Average Sharpe:", round(np.nanmean(sharpes), 3))
print("Best Sharpe:", round(np.nanmax(sharpes), 3))
print("Best strategy index:", np.nanargmax(sharpes))

histogram_sharpe(sharpes)