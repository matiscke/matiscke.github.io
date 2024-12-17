# Inspired by Rodrigo Luger's CV, https://github.com/rodluger/cv

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RationalQuadratic, DotProduct, Matern

from scipy.interpolate import interp1d

lato = fm.FontProperties()

min_year = 2016
max_year = 2024
predict_until = 2026  # Extend predictions until ?


def plot_with_gp_predictions(ax, x, y, color, cumulative=True):
    """Plot with GP predictions and curved confidence intervals."""
    # Actual data up to max_year
    actual_years = x[x <= max_year]
    actual_values = y[x <= max_year]

    # Fit Gaussian Process model using data up to max_year
    # kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
    # kernel = RationalQuadratic(length_scale=1.0, alpha=1.0, alpha_bounds=(1e-4, 1e4), length_scale_bounds=(1e-4, 1e4))
    # kernel = C(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
    # kernel = 1.* C(1., (0.01, 100.0)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)
    # kernel = 1.*  (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0))) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)
    kernel = 1.* C(1., (0.01, 100.0))**2 * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)


    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    gp.fit(actual_years.reshape(-1, 1), actual_values)

    # Predict for the future starting from max_year + 1
    future_years = np.arange(max_year + 1, predict_until + 1, step=1.).reshape(-1, 1)
    y_pred, sigma = gp.predict(future_years, return_std=True)

    # Ensure predictions never decrease
    y_pred = np.maximum.accumulate(y_pred)

    if cumulative:
        # Plot the actual data points and cumulative sum
        y_cumsum = np.cumsum(actual_values)
    else:
        y_cumsum = actual_values
    ax.plot(actual_years, y_cumsum, ".", color=color, ms=4)

    if cumulative:
        # Compute cumulative sums for predictions starting from the last actual data point
        y_pred_cumsum = np.cumsum(y_pred) + y_cumsum[-1]
        lower_confidence = np.cumsum(y_pred - 1.96 * sigma) + y_cumsum[-1]
        upper_confidence = np.cumsum(y_pred + 3. * sigma) + y_cumsum[-1]
    else:
        y_pred_cumsum = y_pred
        lower_confidence = y_pred - 1.96 * sigma
        upper_confidence = y_pred + 3. * sigma

    # Plot the predictions
    # ax.plot(future_years, y_pred_cumsum, ":", color=color, lw=2, alpha=0.5)
    # ax.plot(future_years[:2], y_pred_cumsum[:2], ":", color=color, lw=4, alpha=0.5)
    ax.fill_between(future_years.ravel(),
                    lower_confidence,
                    upper_confidence,
                    color=color, alpha=0.2, edgecolor="none")

    # extend current and plot connection actual-> predicted
    ax.plot([max_year, max_year + 0.5], [y_cumsum[-1], y_cumsum[-1] + (y_pred_cumsum[0] - y_cumsum[-1])/1.7], # make this look more realistic until I figure out a better GP kernel
            "-", color=color, lw=4, alpha=0.5)
    # ax.plot([max_year, max_year + 1], [y_cumsum[-1], y_pred_cumsum[0]], ":", color=color, lw=2, alpha=0.5)
    ax.fill_between([max_year, max_year + 1],
                    [y_cumsum[-1], lower_confidence[0]],
                    [y_cumsum[-1], upper_confidence[0]],
                    color=color, alpha=0.2, edgecolor="none")

    plt.setp(
        ax.get_xticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    plt.setp(
        ax.get_yticklabels(), rotation=30, fontsize=10, fontproperties=lato, alpha=0.75
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    odd_years = [year for year in range(min_year, predict_until + 1) if year % 2 != 0]
    ax.set_xticks(odd_years)

    # for tick in ax.get_xticklabels() + ax.get_yticklabels():
    #     tick.set_fontsize(10)
    # ax.set_ylabel(y_label, fontsize=15)
    # ax.set_xlabel("year", fontsize=15)


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


    plot_with_gp_predictions(ax, p.Year.values, p.Refereed.values, "C7")



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

    plot_with_gp_predictions(ax, c.Year.values, c.Total.values, "C4")



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

        plot_with_gp_predictions(ax, xi.values, yi, "C%d" % i, cumulative=False)

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
    # ax.legend()



def make_plots():
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 3, figsize=(11, 2))
    fig.subplots_adjust(wspace=0.8)
    plot_papers(axs[0])
    plot_cites(axs[1])
    plot_metrics(axs[2])
    for ax in axs:
        ax.spines[['right', 'top']].set_visible(False)
        # ax.set_xlim(min_year, datetime.now().year + datetime.now().month / 12)

        odd_years = [year for year in range(min_year, max_year + 2) if year % 2 != 0]
        ax.set_xticks(odd_years)
        ax.set_xlim(right=predict_until - 0.6)
        ax.set_ylim(top=0.8 * ax.get_ylim()[1])

    fig.text(0.5, -0.5, 'h Index: the largest number h such that h publications have at least h citations.', ha='left', fontsize=9, color='gray', fontproperties=lato)
    fig.text(0.5, -0.6, 'i10 Index: number of publications with at least 10 citations.', ha='left', fontsize=9, color='gray', fontproperties=lato)
    fig.savefig("../img/metrics.svg", bbox_inches="tight")


if __name__ == "__main__":
    make_plots()
