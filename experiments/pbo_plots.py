from detector.synthetic import daily_returns, returns_with_edge
from detector.cscv import probability_of_backtest_overfitting
from detector.plots import plot_logits, plot_in_vs_out


# pure noise
returns = daily_returns(252, 100, seed=1)
pbo, logits, winners_in, winners_out = probability_of_backtest_overfitting(returns)
print("Noise PBO:", round(pbo, 3))
plot_logits(logits)
plot_in_vs_out(winners_in, winners_out)

# real edge
returns = returns_with_edge(2520, 100, 0.002, seed=1)
pbo, logits, winners_in, winners_out = probability_of_backtest_overfitting(returns)
print("Edge PBO:", round(pbo, 3))
plot_logits(logits)
plot_in_vs_out(winners_in, winners_out)