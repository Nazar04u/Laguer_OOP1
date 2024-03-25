import numpy as np
import pandas as pd
import unittest
from scipy.integrate import quad
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact


class Laguer:

    def __init__(self, beta=2, sigma=4):
        self._beta = 2
        self._sigma = 4

    def lagger(self, t, n):
        l_0 = np.sqrt(self.sigma) * (np.exp(-self.beta * t / 2))
        l_1 = np.sqrt(self.sigma) * (1 - self.sigma * t) * (np.exp(-self.beta * t / 2))

        if n == 0:
            return l_0
        if n == 1:
            return l_1
        if n >= 2:
            l_next = (2 * 2 - 1 - t * self.sigma) / 2 * l_1 - (2 - 1) / 2 * l_0
            for j in range(3, n + 1):
                l_0 = l_1
                l_1 = l_next
                l_next = (2 * j - 1 - t * self.sigma) / j * l_1 - (j - 1) / j * l_0
            return l_next

    def tabulate_lagger(self, T, n, ):
        t = np.linspace(0, T, 100)
        results = self.lagger(t, n)
        df = pd.DataFrame({'t': t, 'l': results})
        return df.round(5)

    def experiment(self, epsilon=1e-3, N=20):
        t = 0
        while True:
            t += 0.0001
            res = []

            for i in range(N + 1):
                x = abs(self.lagger(t, N))
                if x < epsilon:
                    res.append(x)
                    if i == N:
                        return t, pd.DataFrame({"results": res})
                else:
                    break

    def quad(self, f, a, b, N=10000):

        x = np.linspace(a, b, N)
        s = sum([f(i) for i in x])
        return s * abs(b - a) / N

    def lagger_transformation(self, f, n):
        def integrand(t):
            return f(t) * self.lagger(t, n) * np.exp(-t * (self.sigma - self.beta))

        b = self.experiment(100)[0]

        return quad(integrand, 0, b)

    def tabulate_tranformation(self, f, N):
        t = range(1, N + 1)
        results = [self.lagger_transformation(f, n) for n in t]

        return results

    def reversed_lagger_transformation(self, h_list, t):
        result_sum = 0

        h_list_new = list(filter(lambda x: x != 0, h_list))

        for i in range(len(h_list_new)):
            result_sum += h_list_new[i] * self.lagger(t, i)

        return result_sum

    def plot_lagger(self, T, N):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        for n in range(N + 1):
            lagger_values = self.tabulate_lagger(T, n)
            ax.plot(lagger_values['t'], lagger_values['l'], label=f"n={n}", linewidth=2.0, alpha=0.7)

        ax.set_xlabel("t")
        ax.set_ylabel("l(t)")
        ax.set_title("Lagger polynomials")
        fig.legend(loc='lower center', ncol=5)
        plt.show()

    def plot_transformation(self, f, n):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        transform_values = self.tabulate_tranformation(f, n, )
        ax.bar(range(1, n + 1), transform_values, alpha=0.7, edgecolor='black')

        ax.set_xlabel("n")
        ax.set_ylabel("f_n")
        ax.set_title("Transformation")
        ax.set_xticks(range(1, n + 1))
        fig.tight_layout()
        plt.axhline(0, color='black')
        plt.show()

    def plot_tranformations(self, f, n, t1=0, t2=2 * np.pi):

        transform_values = self.tabulate_tranformation(f, n)
        reversed_transform_values = [self.reversed_lagger_transformation(transform_values, t) for t in
                                     np.linspace(t1, t2, 1000)]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.subplots(2, 1)
        ax[0].bar(range(1, n + 1), transform_values, alpha=0.7, edgecolor='black')

        ax[0].set_xlabel("n")
        ax[0].set_ylabel("f_n")
        ax[0].set_title("Transformation coefs")
        ax[0].set_xticks(range(1, n + 1))
        fig.tight_layout()
        ax[0].axhline(0, color='black')

        ax[1].plot(np.linspace(t1, t2, 1000), reversed_transform_values, alpha=0.7, linewidth=2.0)
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("f(t)")
        ax[1].set_title("Reversed transformation")

        plt.show()

