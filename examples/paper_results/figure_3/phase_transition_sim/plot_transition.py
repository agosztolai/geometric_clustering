import numpy as np
import networkx as nx
from pathlib import Path
import scipy as sc
import pylab as plt
from tqdm import tqdm
import matplotlib
import scipy.interpolate as sci
import networkx as nx
import pickle
import itertools

matplotlib.use("Agg")
marker = itertools.cycle(("p", "+", ".", "o", "*", "s", "v"))
color = itertools.cycle(("C0", "C1", "C2", "C3", "C4", "C5", "C6"))


def plot_phase_transition_old(r, osmean, osstd, ax2, th=0.2):

    f = sci.interp1d(np.arange(len(times)), times)
    ostime = np.log10(f(osmean[:, 0]))
    mixtime = np.log10(f(osmean[:, 1]))

    ostimestd = ostime - np.log10(f(osmean[:, 0] - osstd[:, 0]))
    mixtimestd = mixtime - np.log10(f(osmean[:, 1] - osstd[:, 1]))

    mean = mixtime - ostime
    std = np.sqrt((ostimestd / ostime) ** 2 + (mixtimestd / mixtime) ** 2)

    ax2.plot(r, mean, "-o", marker=next(marker), markersize=7)
    ax2.fill_between(r, mean + std, mean - std, alpha=0.35)

    ax2.axvline((c_ - np.sqrt(c_)) / (c_ + np.sqrt(c_)), c="C0", lw=3, ls="--")

    gplus = sci.interp1d(mean + std, r)
    g = sci.interp1d(mean, r)
    gminus = sci.interp1d(mean - std, r)

    return [gminus(th), g(th), gplus(th)]


def plot_phase_transition(r, kappamin_avg, kappamin_std, ax2, c_, c):

    mean = np.array(kappamin_avg)
    std = np.array(kappamin_std)

    from scipy.ndimage.filters import gaussian_filter
    smooth_mean = gaussian_filter(mean, 1.2)

    #ax2.plot(r, mean, "-o", marker=next(marker), markersize=7, c=c, label=r"$\bar k={}$".format(c_))
    ax2.plot(r, mean, "-+", c=c, label=r"$\bar k={}$".format(c_))
    #ax2.plot(r, smooth_mean, "-", c=c, label=r"$\bar k={}$".format(c_))
    ax2.fill_between(r, mean + std, mean - std, alpha=0.35)
    ax2.axvline((c_ - np.sqrt(c_)) / (c_ + np.sqrt(c_)), lw=1, ls="--", c=c)


def compute_intersection(r, kappamin_avg, kappamin_std, th=0.1, last_ER=4):

    mean = np.array(kappamin_avg)

    #from scipy.ndimage.filters import gaussian_filter
    #mean = gaussian_filter(mean, 1.0)
    std = np.array(kappamin_std)
    shift = 0 #np.mean(mean[:last_ER])
    mean -= shift
    #th *= np.mean(std[:last_ER])

    gplus = sci.interp1d(mean + std, r, fill_value="extrapolate")
    g = sci.interp1d(mean, r, fill_value="extrapolate")
    gminus = sci.interp1d(mean - std, r, fill_value="extrapolate")
    return gminus(th), g(th), gplus(th), th + shift


if __name__ == "__main__":
    cases = 20
    c = [5, 8, 10, 15, 20, 25, 30]  # average degree
    #c = [8, 10, 15, 20, 25, 30]  # average degree
    #n = 5000
    th = 0.035
    last_ER = 4  # last values considered as ER to average over
    times = [0.1]
    folder = Path("data")

    fig, ax2 = plt.subplots(figsize=(5, 3))
    ax2.set_xlabel(r"Edge density ratio, $r$")
    ax2.set_ylabel(
        r"Timescale separation, $\langle \overline{\tau_{os}} \rangle - \tau^\mathsf{mix}_{ij}$"
    )

    rs = pickle.load(open(folder / "rs.pkl", "rb"))
    kappamin_avgs = pickle.load(open(folder / "kappamin_avgs.pkl", "rb"))
    kappamin_stds = pickle.load(open(folder / "kappamin_stds.pkl", "rb"))
    intersections = []
    for i, (r, kappamin_avg, kappamin_std, c_) in enumerate(
        zip(rs, kappamin_avgs, kappamin_stds, c)
    ):
        _c = next(color)
        plot_phase_transition(r, kappamin_avg, kappamin_std, ax2, c_, _c)
        intersection = compute_intersection(
            r, kappamin_avg, kappamin_std, th=th, last_ER=last_ER
        )
        #plt.axhline(intersection[-1], c=_c)
        intersections.append(intersection)

    ax2.set_xlabel(r"Edge density ratio, $r$")
    ax2.set_ylabel(r"Sensitivity index, ")
    ax2.axhline(th, c="k")
    ax2.set_xlim([0.1, 1])
    ax2.set_ylim([0.002, 10])
    # ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.legend(loc='upper right')
    plt.savefig("pt_curvature.svg", bbox_inches="tight")

    plt.figure(figsize=(5, 2))
    k = np.linspace(0.1, 40, 1000)
    rcrit = np.clip((k - np.sqrt(k)) / (k + np.sqrt(k)), 0, 1)

    plt.plot(k, rcrit)
    intersections = np.array(intersections)
    # plt.errorbar(c, np.array(rc2)[:, 1], yerr=np.array(rc2)[:, [0, 2]].T / 2, fmt="o")
    plt.plot(c, intersections[:, 1], "o")  # , fmt="o")
    plt.fill_between(c, intersections[:, 0], intersections[:, 2], alpha=0.35)

    plt.xlabel(r"Mean degree, $\overline{k}$")
    plt.ylabel(r"Critical edge density ratio, $r_{\overline{k}}^*$")
    plt.gca().set_ylim(0, 1)
    plt.gca().set_xlim(0, 40)
    plt.savefig("critical_density.svg", bbox_inches='tight')
