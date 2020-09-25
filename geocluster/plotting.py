"""plotting functions"""
import os
from pathlib import Path

import matplotlib
import matplotlib.colors as col
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
from tqdm import tqdm


def _savefig(fig, folder, filename, ext):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename).with_suffix(ext), bbox_inches="tight")


def plot_edge_curvatures(
    times,
    kappas,
    ylog=False,
    folder="figures",
    filename="curvature",
    ext=".svg",
    ax=None,
    figsize=(5, 4),
):
    """plot edge curvature"""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None

    for kappa in kappas.T:
        if all(kappa > 0):
            color = "C0"
        else:
            color = "C1"
        ax.plot(np.log10(times), kappa, c=color, lw=0.5)
        

    if ylog:
        ax.set_xscale("symlog")
    ax.axhline(0, ls="--", c="k")
    ax.axis([np.log10(times[0]), np.log10(times[-1]), np.min(kappas), 1])
    ax.set_xlabel(r'$log_{10}(t)$')
    ax.set_ylabel('Edge curvatures')

    _savefig(fig, folder, filename, ext=ext)
    return fig, ax


def plot_edge_curvature_variance(
    times,
    kappas,
    folder="figures",
    filename="curvature_variance",
    ext=".svg",
    ax=None,
    figsize=(5, 4),
):
    """Plot the variance of the curvature across edges."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None

    ax.plot(np.log10(times), np.std(kappas, axis=1))
    ax.set_xlabel(r'$log_{10}$(t)')
    ax.set_ylabel('Edge curvature variance')
    ax.set_xlim([np.log10(times[0]), np.log10(times[-1])])

    _savefig(fig, folder, filename, ext=ext)
    return fig, ax


def plot_graph_snapshots(
    graph,
    times,
    kappas,
    folder="images",
    filename="image",
    node_size=5,
    edge_width=2,
    disable=False,
    ext=".svg",
    figsize=(5, 4),
):
    """plot the curvature on the graph for each time"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    matplotlib.use("Agg")
    for i, kappa in tqdm(enumerate(kappas), total=len(kappas), disable=disable):
        plt.figure(figsize=figsize)
        plot_graph(
            graph, edge_color=kappa, node_size=node_size, edge_width=edge_width,
        )
        plt.title(r"$log_{10}(t)=$" + str(np.around(np.log10(times[i]), 2)))

        plt.savefig(folder + "/" + filename + "_%03d" % i + ext, bbox_inches="tight")
        plt.close()


def _get_colormap(edge_color, colormap="standard"):
    """Get custom colormaps"""
    if colormap == "adaptive":
        edge_color_min = np.min(edge_color)  # abs(min(np.min(edge_color), 0))
        edge_color_max = np.max(edge_color)  # max(np.max(edge_color), 0)
        edge_color_0 = -edge_color_min / (edge_color_max - edge_color_min)

        cdict_with_neg = {
            "red": [(0.0, 0.0, 0.0), (edge_color_0, 0.1, 0.1), (1.0, 1.0, 1.0)],
            "green": [(0.0, 0.1, 0.1), (edge_color_0, 0.1, 0.1), (1.0, 0.0, 0.0)],
            "blue": [(0.0, 1.0, 1.0), (edge_color_0, 0.1, 0.1), (1.0, 0.0, 0.0)],
            "alpha": [(0.0, 0.8, 0.8), (edge_color_0, 0.2, 0.2), (1.0, 0.8, 0.8)],
        }

        cdict_no_neg = {
            "red": [(0, 0.1, 0.1), (1.0, 1.0, 1.0)],
            "green": [(0, 0.1, 0.1), (1.0, 0.0, 0.0)],
            "blue": [(0, 0.1, 0.1), (1.0, 0.0, 0.0)],
            "alpha": [(0, 0.2, 0.2), (1.0, 0.8, 0.8)],
        }
        if edge_color_0 < 0:
            return col.LinearSegmentedColormap("my_colormap", cdict_no_neg)
        return col.LinearSegmentedColormap("my_colormap", cdict_with_neg)

    return plt.cm.coolwarm


def plot_graph(
    graph,
    edge_color=None,
    edge_width=1,
    node_colors=None,
    node_size=20,
    colormap="standard",
    show_colorbar=True,
    vmin=None,
    vmax=None,
):
    """plot the curvature on the graph"""
    pos = list(nx.get_node_attributes(graph, "pos").values())
    
    if pos == []:
        pos = nx.spring_layout(graph)

    if edge_color is not None:
        cmap = _get_colormap(edge_color, colormap=colormap)

        if vmin is None:
            vmin = np.min(edge_color)
        if vmax is None:
            vmax = np.max(edge_color)
    else:
        cmap, vmin, vmax = None, None, None
        
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_size=node_size,
        node_color=node_colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
    )

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=edge_width,
        edge_color=edge_color,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        alpha=0.5,
    )

    if show_colorbar:
        norm = plt.cm.colors.Normalize(vmin, vmax)
        edges = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(edges)

    plt.axis("off")
    
    
def plot_communities(
    graph,
    kappas,
    all_results,
    folder="communities",
    edge_color="0.5",
    edge_width=2,
    figsize=(15, 10),
    ext =".png",
):
    
    from pygenstability.plotting import plot_single_community
    
    """now plot the community structures at each time in a folder"""
    if not os.path.isdir(folder):
        os.mkdir(folder)

    mpl_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    
    pos = list(nx.get_node_attributes(graph, "pos").values())
    if pos == []:
        pos = nx.spring_layout(graph)
        for i in graph:
            graph.nodes[i]['pos'] = pos[i]
    
    for time_id in tqdm(range(len(all_results["times"]))):
        plt.figure(figsize=figsize)
        plot_single_community(
            graph, all_results, time_id, edge_color="1", edge_width=3, node_size=10
        )
        plot_graph(
            graph, edge_color=kappas[time_id], node_size=0, edge_width=edge_width,
        )
        plt.savefig(
            os.path.join(folder, "time_" + str(time_id) + ext), bbox_inches="tight"
        )
        plt.close()
    matplotlib.use(mpl_backend)    


# unused/deprecated functions below


def plot_scales_distribution(
    times, edge_scales, method="hist", filename="hist_scales.png", figsize=(10, 5),
):
    """plot scales on edges with histogram, or gaussian kernel, or both"""
    plt.figure(figsize=figsize)

    if method in ("hist", "both"):
        plt.hist(np.log10(edge_scales), bins=40, density=True)

    if method in ("gaussian", "both"):
        pdf = stats.gaussian_kde(np.log10(edge_scales))
        plt.plot(np.log10(times), pdf(np.log10(times)), color="navy", linestyle="-")
        plt.scatter(
            np.log10(edge_scales),
            np.zeros_like(edge_scales),
            marker="x",
            color="k",
            alpha=0.1,
        )

    plt.xlabel("Zero crossings / edge scales")
    plt.gca().set_xlim(np.log10(times[0]), np.log10(times[-1]))
    plt.savefig(filename)


def plot_scales_graph(graph, edge_scales, filename="graph_scales.png", figsize=(10, 5)):
    """plot scales on edges, from curvatures"""
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

    plt.savefig(filename)


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

    for i, graph in tqdm(enumerate(graphs), total=len(graphs), disable=disable):
        plt.figure(figsize=figsize)

        plot_graph(
            graphs[0],
            "k",
            node_colors="k",
            node_size=node_size,
            node_labels=node_labels,
            edge_width=edge_width,
            figsize=figsize,
            show_colorbar=False,
        )
        total_weight = sum([graph.nodes[u]["weight"] for u in graph])
        node_size_weight = (
            node_size
            * np.array([graph.nodes[u]["weight"] for u in graph])
            / total_weight
            * 100
        )
        plot_graph(
            graph,
            "C1",
            node_colors="C1",
            node_size=node_size_weight,
            node_labels=node_labels,
            edge_width=edge_width,
            figsize=figsize,
            show_colorbar=False,
        )

        plt.savefig(folder + "/" + filename + "_%03d" % i + ext, bbox_inches="tight")
        plt.close()


def plot_embeddings(embeddings, folder="embedding", filename="image", ext=".png"):
    """plot the embedding results on scatter plot"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for i, embedding in enumerate(embeddings):
        plt.figure()
        plt.scatter(embedding[0], embedding[1])
        plt.savefig(folder + "/" + filename + "_%03d" % i + ext, bbox_inches="tight")
