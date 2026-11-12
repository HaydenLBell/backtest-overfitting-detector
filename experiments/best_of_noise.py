from detector.stats import sharpe_ratio
from detector.synthetic import daily_returns

import numpy as np
import matplotlib.pyplot as plt




def histogram_sharpe():
    pass
    #print(np.nanmean(sharpes)) # print the average across all sharpes
    #print(np.nanmax(sharpes)) # prints the best sharp
    #print(np.nanargmax(sharpes)) # prints the column index with the best (best strategy)




num_days = 252          # 1 year of data
num_strategies = 1000   # number of strategies "tried"

returns = daily_returns(num_days, num_strategies, seed=42)
sharpes = sharpe_ratio(returns, annualise=True)

print("Strategies tested:", num_strategies)
print("Days of data:", num_days)
print("Average Sharpe:", round(np.nanmean(sharpes), 3))
print("Best Sharpe:", round(np.nanmax(sharpes), 3))
print("Best strategy index:", np.nanargmax(sharpes))