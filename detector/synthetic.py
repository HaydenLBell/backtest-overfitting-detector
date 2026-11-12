import numpy as np


# num_days = T (rows)
# num_strategies = N (columns)
def daily_returns(num_days, num_strategies, daily_vol=0.01, seed=None):
    rng = np.random.default_rng(seed)
    returns_matrix = rng.normal(0, daily_vol, size=(num_days, num_strategies))
    return returns_matrix


def matrix_pretty_print(matrix):
    num_days = matrix.shape[0]
    num_strategies = matrix.shape[1]

    # header row: strategy names
    header = "".ljust(8)
    for strategy in range(num_strategies):
        name = "Strat " + str(strategy + 1)
        header = header + name.rjust(10)
    print(header)
    print("-" * len(header))

    # one row per day, returns shown as percentages
    for day in range(num_days):
        row = ("Day " + str(day + 1)).ljust(8)
        for strategy in range(num_strategies):
            percent = round(matrix[day][strategy] * 100, 2)
            cell = str(percent) + "%"
            row = row + cell.rjust(10)
        print(row)


matrix_pretty_print(daily_returns(5, 5, seed=42))
