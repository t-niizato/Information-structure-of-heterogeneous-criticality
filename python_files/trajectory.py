import numpy as np
import itertools

def trajectory_difference(N, history):
    x = history[:, 0:N] - history[0, 0:N]
    y = history[:, N:2*N] - history[0, N:2*N]

    comb = itertools.combinations(np.arange(N), 2)
    count = 0
    data = np.zeros(60)
    for i in comb:
        dx = x[:, i[0]] - x[:, i[1]]
        dy = y[:, i[0]] - y[:, i[1]]
        data[count] = np.sum(np.sqrt(dx * dx + dy * dy))
        count += 1

    return np.std(data)


