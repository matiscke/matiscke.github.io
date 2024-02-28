# Inspired by Rodrigo Luger's CV, https://github.com/rodluger/cv

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
lato = fm.FontProperties()

min_year = 2017
max_year = 2023



fig, ax = plt.subplots(figsize=(6, 4))


def plot_cites(ax):
    """Plot citations over time."""

    c = pd.read_csv('metrics-citations.csv', delimiter=', ', engine='python')
    c = c[c['Year'].between(min_year, max_year)]

    cdf = np.cumsum(c.Total)
    # bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.plot(c.Year, cdf, ".", color="C2", ms=3)
    ax.plot(c.Year, cdf, "-", color="C4", lw=3, alpha=0.5)
    plt.setp(
        ax.get_xticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    plt.setp(
        ax.get_yticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(10)
    ax.set_ylabel("citations", fontsize=16)
    ax.set_xlabel("year", fontsize=16)
    # ax.set_xlim(min_year, datetime.now().year + datetime.now().month / 12)
    ax.set_xlim(min_year, max_year + 0.5)


def plot_metrics(ax):
    """Plot h-index and i10-index."""
    m = pd.read_csv('metrics-indices.csv', delimiter=', ', engine='python')
    m = m[m['Year'].between(min_year, max_year)]

    for i, metric in enumerate(['h Index', 'i10 Index']):

        # HACK to increase resolution.
        fac = 1
        xi = np.repeat(m['Year'], fac) + np.tile(np.linspace(0, 1, fac, endpoint=False), len(m['Year']))
        yi = np.interp(xi, m['Year'] + 0.5, m[metric])

        ax.plot(xi, yi, ".", color="C%d" % i, ms=3)
        ax.plot(xi, yi, "-", color="C%d" % i, lw=3, alpha=0.75, label=metric)

    plt.setp(
        ax.get_xticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    plt.setp(
        ax.get_yticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("index", fontsize=16)
    ax.set_xlabel("year", fontsize=16)
    # ax.set_xlim(min_year, datetime.now().year + datetime.now().month / 12)
    ax.set_xlim(min_year, max_year + 0.5)
    ax.legend()



def make_plots():
    fig, axs = plt.subplots(1, 3, figsize=(11, 2))
    fig.subplots_adjust(wspace=0.6)
    # plot_papers(axs[0])
    plot_cites(axs[1])
    plot_metrics(axs[2])
    for axis in axs[4:]:
        axis.axis("off")

    plt.show()
    fig.savefig("metrics.pdf", bbox_inches="tight")


if __name__ == "__main__":
    make_plots()
