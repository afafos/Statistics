import pandas
import numpy as np
from IPython.display import display
from typing import List


def print_characteristics(N: List[int]) -> None:
    methods = [
        lambda n: np.random.normal(0.0, 1.0, n),
        lambda n: np.random.standard_cauchy(n),
        lambda n: np.random.standard_t(3.0, n),
        lambda n: np.random.poisson(10.0, n),
        lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    ]

    names = ['normal', 'cauchy', "student's", 'poisson', 'uniform']
    repeats = 1000

    for i in range(len(methods)):
        for n in N:
            data = np.zeros([2, 5])
            for j in range(repeats):
                sample = methods[i](n)

                sample.sort()
                x = np.mean(sample)
                med_x = np.median(sample)
                z_r = (sample[0] + sample[-1]) / 2.0
                z_q = (sample[int(np.ceil(n / 4.0) - 1)] + sample[int(np.ceil(3.0 * n / 4.0) - 1)]) / 2.0
                r = int(np.round(n / 4.0))
                z_tr = (1.0 / (n - 2 * r)) * sum([sample[i] for i in range(r, n - r)])

                stats = [x, med_x, z_r, z_q, z_tr]
                for k in range(len(stats)):
                    data[0][k] += stats[k]
                    data[1][k] += stats[k] * stats[k]

            data /= repeats
            data[1] -= data[0] ** 2
            df = pandas.DataFrame(data, columns=["x", "med x", "z_R", "z_Q", "z_{tr}"], index=["E(z)", "D(z)"])
            print(f"{names[i]} n = {n}")
            display(df)