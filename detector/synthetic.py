import numpy as np


# num_days = T (rows)
# num_strategies = N (columns)
def daily_returns(num_days, num_strategies, daily_vol=0.01, seed=None):
    rng = np.random.default_rng(seed)
    returns_matrix = rng.normal(0, daily_vol, size=(num_days, num_strategies))
    return returns_matrix

def returns_with_edge(num_days, num_strategies, edge, edge_column=0, daily_vol=0.01, seed=None):
    returns = daily_returns(num_days, num_strategies, daily_vol, seed)

    returns[:, edge_column] = returns[:, edge_column] + edge    

    return returns

    

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


###################
###   TESTING   ###
#################### 

#print(daily_returns(5, 5, seed=42))
#matrix_pretty_print(daily_returns(5, 5, seed=42))

#plain = daily_returns(5, 3, seed=1)
#with_edge = returns_with_edge(5, 3, 0.002, seed=1)

#print(plain.mean(axis=0))
#print(with_edge.mean(axis=0))