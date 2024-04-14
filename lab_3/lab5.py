import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def gen_norm(p, size):
    return np.random.multivariate_normal([0, 0], [[1, p], [p, 1]], size).T


def gen_mix(size):
    return 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size).T + \
        0.1 * np.random.multivariate_normal([0, 0], [[10, 9], [9, 10]], size).T


def rq(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    n1 = np.array([x >= med_x and y >= med_y for x, y in zip(x, y)]).sum()
    n2 = np.array([x < med_x and y >= med_y for x, y in zip(x, y)]).sum()
    n3 = np.array([x < med_x and y < med_y for x, y in zip(x, y)]).sum()
    n4 = np.array([x >= med_x and y < med_y for x, y in zip(x, y)]).sum()
    return ((n1 + n3) - (n2 + n4)) / len(x)


def confidence_ellipse(x, y, ax, n_std=3.0):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor='navy')

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def lab5():
    tables = []
    param_signs = ['$E(z)$', '$E(z^2)$', '$D(z)$']
    for size in [20, 60, 100]:
        table = []
        for p in [0, 0.5, 0.9]:
            table.append(['$\\rho = ' + str(p) + '$', '$r$',
                          '$r_Q$', '$r_S$'])
            for param_sign, param_calc_f in zip(param_signs,
                                                [np.mean, lambda vals: np.mean(np.array(vals) ** 2), np.std]):
                row = []
                row.append(param_sign)
                for coef_calc_f in [lambda x, y: stats.pearsonr(x, y)[0], lambda x, y: stats.spearmanr(x, y)[0], rq]:
                    row.append(round(param_calc_f([coef_calc_f(*gen_norm(p, size)) for i in range(1000)]), 3))
                table.append(row)
        tables.append(table)

    mix_table = []
    param_signs = ['$E(z)$', '$E(z^2)$', '$D(z)$']
    for size in [20, 60, 100]:
        mix_table.append(['$n$ = ' + str(size), '$r$', '$r_Q$', '$r_S$'])
        for param_sign, param_calc_f in zip(param_signs, [np.mean, lambda vals: np.mean(np.array(vals) ** 2), np.std]):
            row = []
            row.append(param_sign)
            for coef_calc_f in [lambda x, y: stats.pearsonr(x, y)[0], lambda x, y: stats.spearmanr(x, y)[0], rq]:
                row.append(round(param_calc_f([coef_calc_f(*gen_mix(size)) for i in range(1000)]), 3))
            mix_table.append(row)

    for size, table in zip([20, 60, 100], tables):
        with open("task1_data/" + str(size) + ".tex", "w") as f:
            f.write("\\begin{tabular}{|c|c|c|c|}\n")
            f.write("\\hline\n")
            for row in table:
                f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
                f.write("\\hline\n")
            f.write("\\end{tabular}")

    with open("task1_data/mix.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in mix_table:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")

    for n in [20, 60, 100]:
        fig, ax = plt.subplots(1, 3)
        for i, p in enumerate([0, 0.5, 0.9]):
            x, y = gen_norm(p, n)
            ax[i].scatter(x, y)
            confidence_ellipse(x, y, ax[i])
            ax[i].grid()
            ax[i].set_title(fr'n = {n}, $\rho$ = {p}')

    fig, ax = plt.subplots(1, 3)
    for i, n in enumerate([20, 60, 100]):
        x, y = gen_mix(n)
        ax[i].scatter(x, y)
        confidence_ellipse(x, y, ax[i])
        ax[i].grid()
        ax[i].set_title(f'n = {n}')


if __name__ == "__main__":
    lab5()