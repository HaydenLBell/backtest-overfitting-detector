import numpy as np
from detector.synthetic import daily_returns

def save_returns(returns, path):
    np.savetxt(path, returns, delimiter=",")

def load_returns(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


if __name__ == "__main__":
    returns = daily_returns(252, 100, seed=1)
    save_returns(returns, "test_returns.csv")
    loaded = load_returns("test_returns.csv")

    print(loaded.shape)                            # (252, 100)
    print(np.allclose(returns, loaded))            # True
