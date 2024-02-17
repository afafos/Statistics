import numpy as np
from math import gamma
import matplotlib.pyplot as plt
from typing import List


def print_graphics(N: List[int]) -> None:
    # Normal distribution
    mu_normal = 0.0
    var_normal = 1.0
    sample_normal = [np.random.normal(mu_normal, var_normal, n) for n in N]

    # Cauchy distribution
    mu_cauchy = 0.0
    var_cauchy = 1.0
    sample_cauchy = [mu_cauchy + var_cauchy * np.random.standard_cauchy(n) for n in N]

    # Student's distribution
    t_student = 3.0
    sample_student = [np.random.standard_t(t_student, n) for n in N]

    # Poisson distribution
    mu_poisson = 10.0
    sample_poisson = [np.random.poisson(mu_poisson, n) for n in N]

    # Uniform distribution
    a_uniform = -np.sqrt(3)
    b_uniform = np.sqrt(3)
    sample_uniform = [np.random.uniform(a_uniform, b_uniform, n) for n in N]

    samples = [
        sample_normal,
        [sample_cauchy[i][(np.abs(sample_cauchy[i]) <= 10)] for i in range(len(N))],  # to avoid outliers
        [sample_student[i][(np.abs(sample_student[i]) <= 10)] for i in range(len(N))],  # to avoid outliers
        sample_poisson,
        sample_uniform
    ]

    # Normal distribution
    def calc_normal(x: float):
        return (1.0 / (var_normal * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_normal) ** 2) / (var_normal ** 2))

    # Cauchy distribution
    def calc_cauchy(x: float):
        return (1.0 / np.pi) * (var_cauchy / ((x - mu_cauchy) ** 2 + var_cauchy ** 2))

    # Student's distribution
    def calc_student(x: float):
        return (gamma((t_student + 1.0) / 2.0) / (np.sqrt(t_student * np.pi) * gamma(t_student / 2.0))) * (
                (1.0 + (x ** 2) / t_student) ** (- (t_student + 1.0) / 2.0))

    # Poisson distribution
    def calc_poisson(x: float):
        return (mu_poisson ** x) * np.exp(-mu_poisson) / gamma(x + 1)

    # Uniform distribution
    def calc_uniform(x: float):
        return (1.0 / (b_uniform - a_uniform)) if (x >= a_uniform) and (x <= b_uniform) else 0.0

    densities = [calc_normal, calc_cauchy, calc_student, calc_poisson, calc_uniform]

    names = ['Normal', 'Cauchy', "Student's", 'Poisson', 'Uniform']

    bins = [15, 25, 45]

    for j in range(len(samples)):
        plt.figure(figsize=(15, 5))
        plt.suptitle(f'Cumulative Distribution Function for {names[j]} Distribution')
        for i in range(len(N)):
            plt.subplot(100 + len(N) * 10 + i + 1)
            x_min = min(samples[j][i])
            x_max = max(samples[j][i])
            x = np.linspace(x_min, x_max, 100)
            y = [densities[j](xi) for xi in x]
            plt.hist(samples[j][i], bins=bins[i], density=True, color='white', edgecolor='black')
            plt.plot(x, y, color='black', linewidth=1)
            plt.title(f'n = {N[i]}')
            plt.xlabel('values')
            plt.ylabel('Density')
        plt.show()