# Backtest Overfitting Detector

A tool that estimates whether the best strategy in a backtest reflects genuine skill, or is simply the luckiest of many strategies tried.

## Why this matters

When you test many strategy variants on the same historical data and keep the best one, the winner will always look good, even if none of the strategies has any real edge.

To demonstrate this, I generated 1,000 strategies of pure random noise (true Sharpe ratio of exactly 0) and backtested each over one year of daily data. The best one had an annualised Sharpe ratio of about **3.4**, which would look like an exceptional discovery. It is just the luckiest of 1,000 coin flips.

<!-- TODO: add the histogram of the 1,000 noise Sharpe ratios here -->

This tool takes the returns of **every** strategy variant you tried (not just the winner) and estimates how much of the winner's performance is explained by luck.

## What it measures

### Deflated Sharpe Ratio (DSR)

The Probabilistic Sharpe Ratio (PSR) estimates how confident we can be that a strategy's true Sharpe ratio exceeds a benchmark $SR^*$, adjusting for the length of the track record and for skewness and kurtosis of returns:

```math
\widehat{PSR}(SR^*) = \Phi\left( \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_{3}\widehat{SR} + \frac{\hat{\gamma}_{4} - 1}{4}\widehat{SR}^2}} \right)
```

The **Deflated** Sharpe Ratio is PSR with the benchmark raised to the Sharpe ratio that the best of $N$ zero-skill strategies would be expected to reach by luck alone:

```math
SR^* = \sqrt{V\left[\{\widehat{SR}_n\}\right]} \left( (1-\gamma)\,\Phi^{-1}\left(1 - \frac{1}{N}\right) + \gamma\,\Phi^{-1}\left(1 - \frac{1}{Ne}\right) \right)
```

where $N$ is the number of strategies tried, $V[\{\widehat{SR}_n\}]$ is the variance of their Sharpe ratios, and $\gamma \approx 0.5772$ is the Euler–Mascheroni constant.

**In plain English:** DSR is the probability that the best strategy beats what pure luck would produce, given how many strategies were tried. Values near 1 suggest genuine skill; values near 0.5 or below suggest a lucky winner.

### Probability of Backtest Overfitting (PBO)

PBO is estimated with **Combinatorially Symmetric Cross-Validation (CSCV)**:

1. Split the history into $S$ equal blocks of days.
2. For every way of choosing $S/2$ blocks as in-sample (70 ways for $S = 8$), pick the strategy with the best in-sample Sharpe ratio.
3. Rank that strategy's out-of-sample Sharpe ratio among all strategies, and convert the rank to a logit $\lambda$:

```math
\bar{\omega} = \frac{\text{rank}}{N + 1}, \qquad \lambda = \ln\left(\frac{\bar{\omega}}{1 - \bar{\omega}}\right)
```

4. PBO is the fraction of splits where the in-sample winner was at or below the median out-of-sample:

```math
PBO = \frac{1}{|C_S|}\sum_{J \in C_S} \mathbf{1}\left[\lambda_J \le 0\right]
```

**In plain English:** PBO measures how often picking the best strategy on past data gives you a below-average strategy on unseen data. Near 0 means the selection process works; around 0.5 means it is no better than picking at random.

## Installation

```bash
git clone <repo-url>
cd backtest-overfitting-detector
pip install numpy scipy pandas matplotlib
pip install -e .
```

## Usage

The input is a CSV file of daily returns: **one row per day, one column per strategy variant**, with no header.

```bash
python detect.py returns.csv
python detect.py returns.csv --blocks 16
python detect.py --help
```

(On Windows, use `py` instead of `python`.)

`--blocks` sets the number of CSCV blocks (must be even, default 8). More blocks give more splits but take longer: 8 blocks gives 70 splits, 16 blocks gives 12,870.

### Example output

Run on 100 pure-noise strategies over one year:

```
=== Backtest Overfitting Report ===
Days of data: 252
Strategies tested: 100
Best strategy: 49
    Sharpe (annual): 1.77

Deflated Sharpe Ratio: 0.308
Probability of Backtest Overfitting: 0.914

VERDICT: LIKELY OVERFIT
```

### Verdict rules

| Condition | Verdict |
|---|---|
| DSR ≥ 0.95 **and** PBO ≤ 0.2 | LIKELY GENUINE |
| DSR < 0.5 **or** PBO ≥ 0.5 | LIKELY OVERFIT |
| Otherwise | INCONCLUSIVE |

These thresholds are my own choices, not values prescribed by the papers.

## Validation on synthetic data

To check the detector works, I tested it on data where the right answer is known in advance:

- **Pure noise:** every strategy has a true Sharpe ratio of 0. The detector should report overfitting.
- **Planted edge:** pure noise, plus one strategy with a genuine extra daily return added. The detector should find that strategy and report it as genuine, given enough data.

All synthetic returns are normally distributed with a daily volatility of 1%.

### Deflated Sharpe Ratio (1,000 strategies)

| Data | Days | Found planted strategy? | DSR |
|---|---|---|---|
| Pure noise | 252 | n/a | 0.537 |
| Edge of 0.2% per day | 252 | Yes | 0.746 |
| Edge of 0.2% per day | 2,520 | Yes | 1.000 |
| Edge of 0.02% per day | 2,520 | <!-- TODO --> | 0.71 |

<!-- TODO: add PSR vs a benchmark of 0 for the pure-noise winner, to show PSR is fooled where DSR is not -->

### Probability of Backtest Overfitting (100 strategies, 8 blocks)

| Data | Days | PBO |
|---|---|---|
| Pure noise (average over 20 seeds) | 252 | 0.582 |
| Pure noise (seed 42) | 252 | 0.914 |
| Edge of 0.2% per day | 2,520 | <!-- TODO --> |

### Findings

- On pure noise, DSR sits around 0.5 and PBO around 0.5: the detector correctly finds no evidence of skill.
- With a large planted edge and 10 years of data, DSR reaches 1.0 and the planted strategy is correctly identified.
- With only one year of data, even a large real edge cannot be confirmed (DSR 0.746), because the luck in one year is comparable to the edge.
- A small, realistic edge (annualised Sharpe of about 0.3) could not be confirmed even with 10 years of data when 1,000 strategies were tried. **The more strategies you test, the larger an edge must be before it stands out from luck.**
- PBO from a single dataset varies considerably between random seeds, because the 70 splits all reuse the same days. Averaging over several datasets gives a more reliable picture.

## Project structure

```
backtest-overfitting-detector/
├── detect.py              # command-line tool: report and verdict
├── detector/
│   ├── stats.py           # Sharpe ratio, skewness, kurtosis
│   ├── psr.py             # PSR, expected maximum Sharpe, DSR
│   ├── cscv.py            # CSCV and PBO
│   ├── synthetic.py       # noise and planted-edge data generators
│   ├── data_files.py      # saving and loading returns CSVs
│   └── plots.py           # logit histogram, in-sample vs out-of-sample scatter
└── experiments/           # scripts used for the results above
```

## Limitations

- **Detects luck, not bugs.** A backtest with look-ahead bias or unrealistic trading costs can still pass.
- **Judges the past, not the future.** A genuine historical edge can disappear.
- **The number of trials must be honest.** DSR is only as good as the $N$ it is given; strategies that were tried and discarded still count.
- **Correlated strategies.** Near-identical variants inflate $N$. The tool supports passing a smaller effective number of trials, but does not yet estimate it automatically.
- **Synthetic data is normally distributed.** Real returns have fatter tails, which DSR accounts for but which the validation above does not yet test.

## Roadmap

- [ ] Run the detector on a real parameter sweep (moving-average strategies on 15 years of SPY data)
- [ ] Unit tests with `pytest`
- [ ] Fat-tailed synthetic data for validation
- [ ] Estimate the effective number of independent trials automatically
- [ ] C++ implementation of the CSCV core, exposed to Python with pybind11, with benchmarks

## References

Bailey, D.H. and López de Prado, M. (2012) 'The Sharpe ratio efficient frontier', *Journal of Risk*, 15(2).

Bailey, D.H. and López de Prado, M. (2014) 'The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality', *Journal of Portfolio Management*, 40(5), pp. 94–107. doi:10.3905/jpm.2014.40.5.094.

Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J. (2017) 'The probability of backtest overfitting', *Journal of Computational Finance*, 20(4). doi:10.21314/JCF.2016.322.