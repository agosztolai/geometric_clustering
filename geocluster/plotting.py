"""plotting functions"""
import itertools
import os
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cmx
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats


def plot_edge_curvatures(
    times, kappas, edge_color=None, ylog=False, filename="curvature", ext=".svg"
):
    """plot edge curvature"""

    fig = plt.figure(constrained_layout=True, figsize=(7, 6))
    gs = fig.add_gridspec(
        ncols=2, nrows=3, width_ratios=[3, 1], height_ratios=[2.5, 1, 1]
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.get_xaxis().set_visible(False)
    if ylog:
        ax1.set_yscale("symlog")
        ax1.set_ylabel(r"Edge curvature, $\log\kappa_{ij}$")
    else:
        ax1.set_ylabel(r"Edge curvature, $\kappa_{ij}$")
    ax1.set_xlim([np.log10(times[0]), np.log10(times[-1])])
    ax1.set_ylim([np.min(kappas), 1.1])

    ax2 = fig.add_subplot(gs[1, 0])
    #    ax2.tick_params(axis="x", which="both", left=False, top=False, labelleft=False)
    ax2.set_ylim([-0.1, 1])
    ax2.set_ylabel("Density of \n zero-crossings")

    ax3 = fig.add_subplot(gs[2, 0])
    #    kappas[kappas>0] = 0
    var = np.sum(np.abs(np.diff(kappas, axis=0)), axis=1)
    ax3.plot(np.log10(times[1:]), var)
    ax3.set_ylabel("Variance of curvature")
    ax3.set_xlabel(r"Diffusion time, $\log(\tau)$")

    gs.update(wspace=0.00)
    gs.update(hspace=0)

    for i, kappa in enumerate(kappas.T):
        if edge_color is not None:
            #            color = cmx.tab10(int(edge_color[i] / np.max(edge_color) * 10))
            normalized = (edge_color[i] - np.min(edge_color)) / (
                np.max(edge_color) - np.min(edge_color)
            )
            color = cmx.inferno(normalized)
        elif edge_color is None:
            if all(kappa > 0):
                color = "C0"
            else:
                color = "C1"

        ax1.plot(np.log10(times), kappa, c=color, lw=0.5)

    ax1.axhline(1, ls="--", c="k")
    ax1.axhline(0, ls="--", c="k")

    # find zero crossings
    roots = []
    for j in range(kappas.shape[1]):
        f = InterpolatedUnivariateSpline(np.log10(times), kappas[:, j], k=3)
        _roots = f.roots()
        if len(_roots) > 0:
            roots.append(_roots)

    #    shift_origin = 0.4
    #    shift = int(shift_origin*len(times))
    #    kappas = kappas[:, shift:-2]
    #    t_mins =  times[shift + np.array([ np.argmin(abs(kappas[i])) for i in range(kappas.shape[0]) ])]

    roots = np.array(roots).flatten()

    # plot Gaussian kde
    if len(roots) > 0:
        Tind = np.linspace(np.log10(times[0]), np.log10(times[-2]), 100)

        pdf = stats.gaussian_kde(roots)
        ax2.plot(Tind, pdf(Tind), color="navy", linestyle="-")
        ax2.scatter(roots, np.zeros_like(roots), marker="x", color="k", alpha=0.1)

    plt.savefig(filename + ext)

    return fig


def plot_graph_snapshots(
    graph,
    times,
    kappas,
    folder="images",
    filename="image",
    node_size=5,
    edge_width=2,
    node_labels=False,
    disable=False,
    ext=".svg",
    figsize=(5, 4),
):
    """plot the curvature on the graph for each time"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for i, kappa in tqdm(enumerate(kappas), total=len(kappas), disable=disable):
        plot_graph(
            graph,
            kappa,
            node_size=node_size,
            node_labels=node_labels,
            edge_width=edge_width,
            figsize=figsize,
            node_colors="curvature",
        )
        plt.title(r"$log_{10}(t)=$" + str(np.around(np.log10(times[i]), 2)))

        plt.savefig(folder + "/" + filename + "_%03d" % i + ext, bbox_inches="tight")
        plt.close()


def plot_graph(
    graph,
    kappa=None,
    node_size=20,
    edge_width=1,
    node_labels=False,
    node_colors=None,
    color_map=0,
    figsize=(10, 7),
    label="Edge curvature",
):
    """plot the curvature on the graph"""

    if "pos" in graph.nodes[1]:
        pos = list(nx.get_node_attributes(graph, "pos").values())
    else:
        pos = nx.spring_layout(graph)

    if len(pos[0]) > 2:
        pos = np.asarray(pos)[:, [0, 2]]

    plt.figure(figsize=figsize)
    if kappa is not None:
        kappa_min = abs(min(np.min(kappa), 0))
        kappa_max = max(np.max(kappa), 0)
        kappa_0 = kappa_min / (1.0 + kappa_min)
        cdict = {
            "red": [(0.0, 0.0, 0.0), (kappa_0, 0.1, 0.1), (1.0, 1.0, 1.0)],
            "green": [(0.0, 0.1, 0.1), (kappa_0, 0.1, 0.1), (1.0, 0.0, 0.0)],
            "blue": [(0.0, 1.0, 1.0), (kappa_0, 0.1, 0.1), (1.0, 0.0, 0.0)],
            "alpha": [(0.0, 0.8, 0.8), (kappa_0, 0.02, 0.02), (1.0, 0.8, 0.8)],
        }

        if color_map == 0:
            edge_cmap = col.LinearSegmentedColormap("my_colormap", cdict)
            edge_vmin = -kappa_min
            edge_vmax = 1.0
        elif color_map == 1:
            edge_cmap = cmx.inferno
            edge_vmin = -kappa_min
            edge_vmax = 1.1 * kappa_max  # to avoid the not so visible bright yellow
    else:
        kappa = np.ones(len(graph.edges()))
        edge_vmin = 1.0
        edge_vmax = 1.0
        edge_cmap = plt.get_cmap("viridis")

    if node_colors is None:
        node_colors = "k"
        cmap = None
        vmin = None
        vmax = None
    elif node_colors == "curvature":
        incidence_matrix = nx.incidence_matrix(graph, weight="weight").toarray()
        inverse_degrees = np.diag([1.0 / graph.degree[i] for i in graph.nodes])
        node_colors = inverse_degrees.dot(incidence_matrix.dot(kappa))

        cmap = edge_cmap
        vmin = -kappa_min
        vmax = 1.0
    else:
        cmap = plt.get_cmap("tab20")
        vmin = None
        vmax = None

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_size=node_size,
        node_color=node_colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=edge_width,
        edge_color=kappa,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_cmap=edge_cmap,
    )

    edges = plt.cm.ScalarMappable(
        norm=plt.cm.colors.Normalize(edge_vmin, edge_vmax), cmap=edge_cmap
    )

    plt.colorbar(edges, label=label)

    if node_labels:
        labels_gt = {}
        for i in graph:
            labels_gt[i] = str(i) + " " + str(graph.nodes[i]["old_label"])
        nx.draw_networkx_labels(graph, pos=pos, labels=labels_gt)

    plt.axis("off")


def plot_scales(graph, edge_scales, figsize=(10, 5)):
    """plot scales on edges, from curvatures"""
    plt.figure()
    plt.hist(np.log10(edge_scales), bins=40)
    plt.savefig("hist_scales.png")

    pos = list(nx.get_node_attributes(graph, "pos").values())
    cmap = plt.get_cmap("plasma")

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos=pos, node_size=0)
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edge_color=np.log10(edge_scales),
        width=2,
        edge_cmap=cmap,
        alpha=0.5,
    )

    edges = plt.cm.ScalarMappable(
        norm=plt.cm.colors.Normalize(
            np.log10(min(edge_scales)), np.log10(max(edge_scales))
        ),
        cmap=cmap,
    )

    plt.colorbar(edges, label="Edge scale")

    plt.savefig("graph_scales.png")


def plot_coarse_grain(
    graphs,
    edge_color=None,
    folder="coarse_grain",
    filename="image",
    ext=".png",
    node_size=5,
    edge_width=2,
    node_labels=False,
    disable=False,
    figsize=(5, 4),
):
    """plot coarse grained graphs"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for i, graph in enumerate(graphs):
        plot_graph(
            graph,
            edge_color,
            node_size=node_size,
            node_labels=node_labels,
            edge_width=edge_width,
            figsize=figsize,
        )

        plt.savefig(folder + "/" + filename + "_%03d" % i + ext, bbox_inches="tight")
