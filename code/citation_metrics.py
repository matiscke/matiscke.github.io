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

def plot_papers(ax):
    """Plot publications over time."""
    p = pd.read_csv('metrics-papers.csv', delimiter=', ', engine='python')
    p = p[p['Year'].between(min_year, max_year)]

    cdf = np.cumsum(p.Refereed)
    ax.plot(p.Year, cdf, ".", color="C7", ms=4)
    ax.plot(p.Year, cdf, "-", color="C7", lw=4, alpha=0.7)
    plt.setp(
        ax.get_xticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    plt.setp(
        ax.get_yticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Customizing tick locations to show odd years
    odd_years = [year for year in range(min_year, max_year + 1) if year % 2 != 0]
    ax.set_xticks(odd_years)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(10)
    ax.set_ylabel("refereed\npublications", fontsize=15)
    ax.set_xlabel("year", fontsize=15)
    # ax.set_xlim(min_year, datetime.now().year + datetime.now().month / 12)
    ax.set_xlim(min_year, max_year + 0.5)


def plot_cites(ax):
    """Plot citations over time."""

    c = pd.read_csv('metrics-citations.csv', delimiter=', ', engine='python')
    c = c[c['Year'].between(min_year, max_year)]

    cdf = np.cumsum(c.Total)
    ax.plot(c.Year, cdf, ".", color="C4", ms=4)
    ax.plot(c.Year, cdf, "-", color="C4", lw=4, alpha=0.7)
    plt.setp(
        ax.get_xticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    plt.setp(
        ax.get_yticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Customizing tick locations to show odd years
    odd_years = [year for year in range(min_year, max_year + 1) if year % 2 != 0]
    ax.set_xticks(odd_years)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(10)
    ax.set_ylabel("citations", fontsize=15)
    ax.set_xlabel("year", fontsize=15)
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

        ax.plot(xi, yi, ".", color="C%d" % i, ms=4)
        ax.plot(xi, yi, "-", color="C%d" % i, lw=4, alpha=0.75, label=metric)

    plt.setp(
        ax.get_xticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.7
    )
    plt.setp(
        ax.get_yticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.7
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Customizing tick locations to show odd years
    odd_years = [year for year in range(min_year, max_year + 1) if year % 2 != 0]
    ax.set_xticks(odd_years)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("index", fontsize=15)
    ax.set_xlabel("year", fontsize=15)
    # ax.set_xlim(min_year, datetime.now().year + datetime.now().month / 12)
    ax.set_xlim(min_year, max_year + 0.5)
    ax.legend()


def make_plots():
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 3, figsize=(11, 2))
    fig.subplots_adjust(wspace=0.8)
    plot_papers(axs[0])
    plot_cites(axs[1])
    plot_metrics(axs[2])
    for axis in axs:
        axis.spines[['right', 'top']].set_visible(False)
    # fig.suptitle("Metrics", fontsize=16, x=0.1, y=1.1)
    fig.text(0.5, -0.5, 'h Index: the largest number h such that h publications have at least h citations.', ha='left', fontsize=9, color='gray', fontproperties=lato)
    fig.text(0.5, -0.6, 'i10 Index: number of publications with at least 10 citations.', ha='left', fontsize=9, color='gray', fontproperties=lato)
    plt.show()
    fig.savefig("../img/metrics.svg", bbox_inches="tight")


if __name__ == "__main__":
    make_plots()
