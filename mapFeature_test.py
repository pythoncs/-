import numpy as np


def mapFeature(X, n):
    map_X = np.zeros((len(X), 1))
    X1 = X[:, 1].reshape(-1, 1)
    X2 = X[:, 2].reshape(-1, 1)
    num = 0
    for i in range(0, n + 1):
        for j in range(0, n - i + 1):
            num += 1
            temp = (X1 ** i) * (X2 ** j)
            map_X = np.hstack((map_X, np.array(temp).reshape(-1, 1)))
    return map_X[:, 1:]



