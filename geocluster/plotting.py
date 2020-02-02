"""plotting functions"""
import os
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col


def plot_edge_curvatures(
        times, kappas, ylog=False, filename="edge_curvature", ext=".svg"
):
    """plot edge curvature"""

    fig = plt.figure()
    ax = plt.gca()

    for kappa in kappas.T:
        if all(kappa > 0):
            color = "C0"
        else:
            color = "C1"
        plt.plot(np.log10(times), kappa, c=color, lw=0.5)

    ax.axhline(1, ls="--", c="k")
    ax.axhline(0, ls="--", c="k")

    if ylog:
        ax.set_yscale("symlog")
        ax.set_ylabel("log(edge curvature)")
    ax.set_ylabel("edge curvature")

    ax.set_ylim([np.min(kappas), 1.1])
    ax.set_xlim([np.log10(times[0]), np.log10(times[-1])])

    if filename is not None:
        plt.savefig(filename + ext)

    return fig, ax


def plot_graph_snapshots(
        graph,
        times,
        kappas,
        folder="images",
        filename="image",
        node_size=20,
        edge_width=2,
        node_labels=False,
        ext=".svg",
        figsize=(5, 4),
):
    """plot the curvature on the graph for each time"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for i, kappa in tqdm(enumerate(kappas), total=len(kappas)):
        plot_graph(
            graph,
            kappa,
            node_size=node_size,
            node_labels=node_labels,
            edge_width=edge_width,
            figsize=figsize,
        )
        plt.title(r"$log_{10}(t)=$" + str(np.around(np.log10(times[i]), 2)))
        plt.savefig(folder + "/" + filename + "_%03d" % i + ext, bbox_inches="tight")
        plt.close()


def plot_graph(
        graph,
        kappa,
        node_size=20,
        edge_width=1,
        node_labels=False,
        node_colors=None,
        figsize=(10, 7),
):
    """plot the curvature on the graph for a given time t"""

    if "pos" in graph.nodes[1]:
        pos = list(nx.get_node_attributes(graph, "pos").values())
    else:
        pos = nx.spring_layout(graph)

    if len(pos[0]) > 2:
        pos = np.asarray(pos)[:, [0, 2]]

    plt.figure(figsize=figsize)

    kappa_min = abs(min(np.min(kappa), 0))
    kappa_0 = kappa_min / (1.0 + kappa_min)
    cdict = {
        "red": [(0.0, 0.1, 0.1), (kappa_0, 0.1, 0.1), (1.0, 1.0, 1.0)],
        "green": [(0.0, 0.1, 0.1), (kappa_0, 0.1, 0.1), (1.0, 0.1, 0.1)],
        "blue": [(0.0, 1.0, 1.0), (kappa_0, 0.1, 0.1), (1.0, 0.1, 0.1)],
        "alpha": [(0.0, 0.8, 0.8), (kappa_0, 0.05, 0.05), (1.0, 0.8, 0.8)],
    }

    edge_cmap = col.LinearSegmentedColormap("my_colormap", cdict)
    edge_vmin = -kappa_min
    edge_vmax = 1.0

    if node_colors is None:
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

    plt.colorbar(edges, label="Edge curvature")

    if node_labels:
        labels_gt = {}
        for i in graph:
            labels_gt[i] = str(i) + " " + str(graph.nodes[i]["old_label"])
        nx.draw_networkx_labels(graph, pos=pos, labels=labels_gt)

    plt.axis("off")


def plot_scales(graph, edge_scales):
    """plot scales on edges, from curvatures"""
    plt.figure()
    plt.hist(np.log10(edge_scales), bins=40)
    plt.savefig("hist_scales.png")

    pos = list(nx.get_node_attributes(graph, "pos").values())
    cmap = plt.get_cmap("plasma")

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=pos, node_size=1)
    nx.draw_networkx_edges(
        graph, pos=pos, edge_color=np.log10(edge_scales), width=2, edge_cmap=cmap, alpha=0.5
    )

    edges = plt.cm.ScalarMappable(
        norm=plt.cm.colors.Normalize(
            np.log10(min(edge_scales)), np.log10(max(edge_scales))
        ),
        cmap=cmap,
    )

    plt.colorbar(edges, label="Edge scale")

    plt.savefig("graph_scales.png")
