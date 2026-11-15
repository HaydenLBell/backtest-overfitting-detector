# backtest-overfitting-detector
A project to detect whether a backtest is overfitting using Sharpe ratios, trading strategies, and real-time information.

```math
\widehat{PSR}(SR^*) = \Phi\left( \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_{3}{\widehat{SR}} + \frac{\hat{\gamma}_{4} - 1}{4}{\widehat{SR}}^2}} \right)
```




References:

Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J. (2017) ‘The probability of backtest overfitting’, <i>Journal of Computational Finance</i>, 20(4). doi:10.21314/JCF.2016.322.

