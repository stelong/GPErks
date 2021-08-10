from itertools import cycle

import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from GPErks.constants import HEIGHT, WIDTH
from GPErks.utils.plotting import get_col, interp_col


def boxplot(ST, S1, S2, index_i, index_ij, ylabel):
    df_ST = pd.DataFrame(data=ST, columns=index_i)
    df_S1 = pd.DataFrame(data=S1, columns=index_i)
    df_S2 = pd.DataFrame(
        data=S2,
        columns=["(" + elem[0] + ", " + elem[1] + ")" for elem in index_ij],
    )

    plt.style.use("seaborn")
    gs = grsp.GridSpec(2, 2)
    fig = plt.figure(figsize=(2 * WIDTH, 2 * HEIGHT / 2))

    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(ax=ax0, data=df_S1)
    ax0.set_ylim(0, 1)
    ax0.set_title("First-order effect", fontweight="bold", fontsize=12)
    ax0.set_xticklabels(
        ax0.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    ax1 = fig.add_subplot(gs[0, 1])
    sns.boxplot(ax=ax1, data=df_ST)
    ax1.set_ylim(0, 1)
    ax1.set_title("Total effect", fontweight="bold", fontsize=12)
    ax1.set_xticklabels(
        ax1.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    ax2 = fig.add_subplot(gs[1, :])
    sns.boxplot(ax=ax2, data=df_S2)
    ax2.set_ylim(0, 1)
    ax2.set_title("Second-order effect", fontweight="bold", fontsize=12)
    ax2.set_xticklabels(
        ax2.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    fig.tight_layout()
    # plt.savefig(savepath + ylabel + "_box.pdf", bbox_inches="tight", dpi=1000)
    plt.show()


def donut(ST, S1, index_i, ylabel):
    ST_mean = np.mean(ST, axis=0)
    S1_mean = np.mean(S1, axis=0)

    sum_s1 = S1_mean.sum()
    sum_st = ST_mean.sum()
    ho = sum_st - sum_s1
    x_si = np.array(list(S1_mean) + [ho])
    x_sti = ST_mean

    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDTH, 2 * HEIGHT / 4))

    c = "blue"
    colors = interp_col(get_col(c), len(index_i))
    colors += [interp_col(get_col("gray"), 6)[2]]

    wedges, _ = axes[0].pie(
        x_si,
        radius=1,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[0].set_title("S1", fontsize=12, fontweight="bold")

    axes[1].pie(
        x_sti,
        radius=1,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[1].set_title("ST", fontsize=12, fontweight="bold")

    plt.figlegend(
        wedges, index_i + ["higher-order int."], ncol=5, loc="lower center"
    )
    # plt.savefig(
    #     savepath + ylabel + "_donut.pdf", bbox_inches="tight", dpi=1000
    # )
    plt.show()


def fancy_donut(ST, S1, S2, index_i, ylabel):
    ST = np.median(ST, axis=0)
    S1 = np.median(S1, axis=0)
    S2 = np.median(S2, axis=0)

    D = len(ST)
    V = np.zeros((D, D), dtype=float)

    for j in range(D - 1):
        i1 = int(j * (D - 1) - np.sum(np.array(range(j))))
        i2 = int((j + 1) * (D - 1) - np.sum(np.array(range(j + 1))))
        V[j, j + 1 :] = S2[i1:i2]

    for m in range(D):
        for n in range(D):
            if m > n:
                V[n, m] = 0.5 * V[n, m]
                V[m, n] = V[n, m]

    S = np.hstack(
        (
            S1.reshape(-1, 1),
            np.array([np.sum(V[ind, :]) for ind in range(D)]).reshape(-1, 1),
        )
    )
    rem = np.sum(ST) - (np.sum(S[:, 0]) + np.sum(S[:, 1]))
    S = np.vstack((S, np.array([rem, 0]).reshape(1, -1)))

    fig, axis = plt.subplots(1, 1, figsize=(4 * WIDTH / 3, HEIGHT / 2))

    colors = [
        "red",
        "purple",
        "amber",
        "light_green",
        "blue",
        "pink",
        "teal",
        "brown",
    ]

    inner_colors = []
    for c, _ in zip(cycle(colors), index_i):
        cis = interp_col(get_col(c), 6)
        inner_colors.append(cis[3])
        inner_colors.append(cis[2])
    inner_colors += 2 * [interp_col(get_col("gray"), 6)[2]]

    outer_colors = [get_col(c)[1] for c in colors]
    outer_colors += [interp_col(get_col("gray"), 6)[2]]

    wedges, _ = axis.pie(
        S.sum(axis=1),
        radius=1,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )
    axis.pie(
        S.flatten(),
        radius=1 - 0.3,
        colors=inner_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )

    axis.set(aspect="equal")
    axis.set_title(ylabel, fontsize=12, fontweight="bold")

    plt.figlegend(
        wedges,
        index_i + ["higher-order\ninteractions"],
        ncol=5,
        loc="lower center",
    )
    plt.show()


def heatmap(ST, S1, index_i, ylabel):
    ST_mean = np.mean(ST, axis=0).reshape(1, -1)
    S1_mean = np.mean(S1, axis=0).reshape(1, -1)

    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDTH, 2 * HEIGHT / 8))

    df = pd.DataFrame(data=S1_mean, index=[ylabel], columns=index_i)
    h1 = sns.heatmap(
        df,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidth=0.1,
        cbar_kws={"shrink": 0.8},
        ax=axes[0],
    )
    axes[0].set_title("S1", fontsize=12, fontweight="bold")
    axes[0].tick_params(left=False, bottom=False)
    h1.set_xticklabels(h1.get_xticklabels(), rotation=45, va="top")
    h1.set_yticklabels(h1.get_yticklabels(), rotation=0, ha="right")

    df = pd.DataFrame(data=ST_mean, index=[ylabel], columns=index_i)
    ht = sns.heatmap(
        df,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidth=0.1,
        cbar_kws={"shrink": 0.8},
        ax=axes[1],
    )
    axes[1].set_title("ST", fontsize=12, fontweight="bold")
    axes[1].tick_params(left=False, bottom=False)
    ht.set_xticklabels(ht.get_xticklabels(), rotation=45, va="top")
    ht.set_yticklabels(ht.get_yticklabels(), rotation=0, ha="right")

    # plt.savefig(savepath + ylabel + "_heat.pdf", bbox_inches="tight", dpi=1000)
    plt.show()


def network(ST, S1, S2, index_i, index_ij, ylabel):
    def angle(p, c):
        [dx, dy] = p - c

        if dx == 0:
            if dy > 0:
                return 0.5 * np.pi
            else:
                return 1.5 * np.pi
        elif dx > 0:
            if dy >= 0:
                return np.arctan(dy / dx)
            else:
                return 2.0 * np.pi + np.arctan(dy / dx)
        elif dx < 0:
            return np.pi + np.arctan(dy / dx)

    ST_mean = np.mean(ST, axis=0)
    S1_mean = np.mean(S1, axis=0)
    S2_mean = np.mean(S2, axis=0)

    maximum = np.max([ST_mean.max(), S1_mean.max(), S2_mean.max()])
    ST_mean /= maximum
    S1_mean /= maximum
    S2_mean /= maximum

    min_size = 0
    max_size = 200
    foreground_node_size = [
        min_size + (max_size - min_size) * k for k in list(S1_mean)
    ]
    backgroud_node_size = [
        min_size + (max_size - min_size) * k for k in list(ST_mean)
    ]
    edge_width = [
        np.sqrt((min_size + (max_size - min_size) * k) / np.pi)
        for k in list(S2_mean)
    ]

    Sources = list(list(zip(*index_ij))[0])
    Targets = list(list(zip(*index_ij))[1])
    Weights = list(S2_mean)

    G = nx.Graph()
    for s, t, w in zip(Sources, Targets, Weights):
        G.add_edges_from([(s, t)], w=w)

    Pos = nx.circular_layout(G)

    c = "blue"
    colors = interp_col(get_col(c), 5)
    fig, axis = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))

    nx.draw_networkx_nodes(
        G,
        Pos,
        node_size=backgroud_node_size,
        node_color=len(index_i) * [colors[4]],
        ax=axis,
    )
    nx.draw_networkx_nodes(
        G,
        Pos,
        node_size=foreground_node_size,
        node_color=len(index_i) * [colors[0]],
        ax=axis,
    )
    nx.draw_networkx_edges(
        G,
        Pos,
        width=edge_width,
        edge_color=len(index_ij) * [colors[2]],
        alpha=0.8,
        ax=axis,
    )

    center = [0.0, 0.0]
    radius = 1.0

    names = nx.draw_networkx_labels(
        G, Pos, font_size=12, font_family="DejaVu Sans", ax=axis
    )
    for node, text in names.items():
        position = (
            1.2 * radius * np.cos(angle(Pos[node], center)),
            1.2 * radius * np.sin(angle(Pos[node], center)),
        )
        text.set_position(position)
        text.set_clip_on(False)

    axis.axis("equal")
    axis.set_axis_off()
    fig.tight_layout()
    # plt.savefig(
    #     savepath + ylabel + "_network.pdf", bbox_inches="tight", dpi=1000
    # )
    plt.show()
