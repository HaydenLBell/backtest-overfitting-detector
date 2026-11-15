# py detect.py returns.csv
# prints a report with the DSR, PBO and a verdict

from detector.psr import deflated_sharpe_ratio
from detector.cscv import probability_of_backtest_overfitting
from detector.stats import sharpe_ratio
from detector.data_files import load_returns

import argparse

# 1. Saving and Loading return files
# in detect.data_files/.py

def verdict(dsr, pbo):

    if dsr >= 0.95 and pbo <= 0.2:
        return "LIKELY GENUINE"
    elif dsr < 0.5 or pbo >= 0.5:
        return "LIKELY OVERFIT"
    else:
        return "INCONCLUSIVE"


def print_report(returns, num_blocks=8):

    num_days = returns.shape[0]
    num_strategies = returns.shape[1]

    dsr, best_index = deflated_sharpe_ratio(returns)
    best_sharpe = sharpe_ratio(returns, annualise=True)[best_index]

    pbo, logits, winners_in, winners_out = probability_of_backtest_overfitting(returns, num_blocks)

    print("=== Backtest Overfitting Report ===")
    print("Days of data: ", num_days)
    print("Strategies tested: ", num_strategies)
    print("Best strategy: ", best_index)
    print("  > Sharpe (annual): ", round(best_sharpe, 2 ))
    print("\nDeflated Sharpe Ratio: ", round(dsr, 3))
    print("Probability of Backtest Overfitting: ", round(pbo, 3))

    print()
    print("VERDICT:", verdict(dsr, pbo))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Backtest overfitting detector")
    parser.add_argument("path", help="CSV file of returns (rows = days, columns = strategies)")
    parser.add_argument("--blocks", type=int, default=8, help="number of CSCV blocks (must be even)")
    args = parser.parse_args()

    returns = load_returns(args.path)
    print_report(returns, args.blocks)
