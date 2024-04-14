import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class LinearModel:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    def predict(self, x):
        return self.b0 + self.b1 * x


class LSM(LinearModel):
    def fit(self, x, y):
        xy_m = np.mean(x * y)
        x_m = np.mean(x)
        x_2_m = np.mean(x ** 2)
        y_m = np.mean(y)
        self.b1 = (xy_m - x_m * y_m) / (x_2_m - x_m * x_m)
        self.b0 = y_m - x_m * self.b1


class LAD(LinearModel):
    def fit(self, x, y):
        def abs_error(b, *data):
            x, y = data
            y_predict = b[0] + b[1] * x
            return np.linalg.norm(y - y_predict, ord=1)

        self.b0, self.b1 = opt.minimize(abs_error, [0, 1], args=(x, y)).x


def original(x):
    return 2 * x + 2


def chi_table(data):
    mu = np.mean(data)
    sigma = np.std(data)
    print(f'mu={mu}, sigma={sigma}')

    k = int(np.floor(1.72 * len(data) ** (1 / 3)))
    borders = np.linspace(np.floor(np.min(data)), np.ceil(np.max(data)), k - 1)
    borders = np.insert(borders, 0, -np.inf)
    borders = np.append(borders, np.inf)

    table = []
    table.append(['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$',
                  '$np_i$', '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])

    ns = []
    ps = []
    nps = []
    n_sub_nps = []
    ress = []

    for i in range(len(borders) - 1):
        left = borders[i]
        right = borders[i + 1]

        n = ((left < data) & (data <= right)).sum()
        ns.append(n)

        p = stats.norm.cdf(right) - stats.norm.cdf(left)
        ps.append(p)

        np_ = len(data) * p
        nps.append(np_)

        n_sub_np = n - np_
        n_sub_nps.append(n_sub_np)

        res = n_sub_np ** 2 / np_
        ress.append(res)

        table.append([i + 1, f'({round(left, 2)}, {round(right, 2)}]',
                      round(n, 2), round(p, 2), round(np_, 2), round(n_sub_np, 2), round(res, 2)])
    table.append(['$\sum$', '-', sum(ns), sum(ps), round(sum(nps)), round(sum(n_sub_nps)), round(sum(ress), 2)])
    return table


def write_table(path, table):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in table:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def lab6():
    x = np.arange(-1.8, 2.2, 0.2)
    y = original(x) + np.random.standard_normal(len(x))

    lsm = LSM()
    lsm.fit(x, y)
    print(f'LSM: b0 = {lsm.b0}, b1 = {lsm.b1}')

    lad = LAD()
    lad.fit(x, y)
    print(f'LAD: b0 = {lad.b0}, b1 = {lad.b1}')

    fig, ax = plt.subplots()
    ax.scatter(x, y, label='data')
    points = np.linspace(-1.8, 2, 100)
    ax.plot(points, original(points), color='red', label='original')
    ax.plot(points, lsm.predict(points), color='green', label='lsm')
    ax.plot(points, lad.predict(points), color='purple', label='lad')
    ax.legend()
    ax.grid()

    plt.show()

    x = np.arange(-1.8, 2.2, 0.2)
    y = original(x) + np.random.standard_normal(len(x))
    y[0] += 10
    y[19] -= 10

    lsm = LSM()
    lsm.fit(x, y)
    print(f'b0 = {lsm.b0}, b1 = {lsm.b1}')

    lad = LAD()
    lad.fit(x, y)
    print(f'LAD: b0 = {lad.b0}, b1 = {lad.b1}')

    fig, ax = plt.subplots()
    ax.scatter(x, y, label='data')
    points = np.linspace(-1.8, 2, 100)
    ax.plot(points, original(points), color='red', label='original')
    ax.plot(points, lsm.predict(points), color='green', label='lsm')
    ax.plot(points, lad.predict(points), color='purple', label='lad')
    ax.legend()
    ax.grid()

    plt.show()


if __name__ == "__main__":
    lab6()