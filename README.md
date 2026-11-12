# backtest-overfitting-detector
A project to detect whether a backtest is overfitting using Sharpe ratios, trading strategies, and real-time information.

```math
\widehat{PSR}(SR^*) = \Phi\left( \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_{3}{\widehat{SR}} + \frac{\hat{\gamma}_{4} - 1}{4}{\widehat{SR}}^2}} \right)
```
