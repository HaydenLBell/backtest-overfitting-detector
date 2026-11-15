import numpy as np
import matplotlib.pyplot as plt


def plot_logits(logits):
    """Histogram of where the in-sample winner ranked out-of-sample."""
    plt.figure()
    plt.hist(logits, bins=20)
    plt.axvline(0, color="red")
    plt.xlabel("Logit (below 0 = winner was below average out-of-sample)")
    plt.ylabel("Number of splits")
    plt.title("Where the in-sample winner ranked out-of-sample")
    plt.show()


def plot_in_vs_out(winners_in_sample, winners_out_sample):
    """Scatter of each split winner's in-sample vs out-of-sample Sharpe."""
    # annualise so the numbers are easier to read
    in_annual = np.array(winners_in_sample) * np.sqrt(252)
    out_annual = np.array(winners_out_sample) * np.sqrt(252)

    plt.figure()
    plt.scatter(in_annual, out_annual)
    plt.axhline(0, color="grey")

    # straight line of best fit, to show the overall trend
    slope, intercept = np.polyfit(in_annual, out_annual, 1)
    xs = np.array([in_annual.min(), in_annual.max()])
    plt.plot(xs, slope * xs + intercept, color="red")

    plt.xlabel("Winner's in-sample Sharpe (annualised)")
    plt.ylabel("Same strategy's out-of-sample Sharpe (annualised)")
    plt.title("Does in-sample performance carry over?")
    plt.show()