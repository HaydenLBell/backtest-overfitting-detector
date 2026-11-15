from detector.synthetic import returns_with_edge
from detector.psr import deflated_sharpe_ratio


num_strategies = 1000
edge = 0.002
edge_column = 0

for num_days in [252, 2520]:
    returns = returns_with_edge(num_days, num_strategies, edge, edge_column, seed=42)
    dsr, best_index = deflated_sharpe_ratio(returns)

    print("Days of data:", num_days)
    print("Best strategy:", best_index)
    print("Found the planted strategy?", best_index == edge_column)
    print("DSR:", round(dsr, 3))
    print()