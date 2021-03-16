import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_col(color_name=None):
    """Material Design color palettes (only '100' and '900' variants).
    Help: call with no arguments to see the list of available colors, these are also returned into a list
    Kwarg:
            - color_name: string representing the color's name
    Output:
            - color: list of two elements
                    [0] = lightest color '100'-variant (RGB-triplet in [0, 1])
                    [1] = darkest color '900'-variant (RGB-triplet in [0, 1])
    """
    colors = {
        "red": [[255, 205, 210], [183, 28, 28]],
        "pink": [[248, 187, 208], [136, 14, 79]],
        "purple": [[225, 190, 231], [74, 20, 140]],
        "deep_purple": [[209, 196, 233], [49, 27, 146]],
        "indigo": [[197, 202, 233], [26, 35, 126]],
        "blue": [[187, 222, 251], [13, 71, 161]],
        "light_blue": [[179, 229, 252], [1, 87, 155]],
        "cyan": [[178, 235, 242], [0, 96, 100]],
        "teal": [[178, 223, 219], [0, 77, 64]],
        "green": [[200, 230, 201], [27, 94, 32]],
        "light_green": [[220, 237, 200], [51, 105, 30]],
        "lime": [[240, 244, 195], [130, 119, 23]],
        "yellow": [[255, 249, 196], [245, 127, 23]],
        "amber": [[255, 236, 179], [255, 111, 0]],
        "orange": [[255, 224, 178], [230, 81, 0]],
        "deep_orange": [[255, 204, 188], [191, 54, 12]],
        "brown": [[215, 204, 200], [62, 39, 35]],
        "gray": [[245, 245, 245], [33, 33, 33]],
        "blue_gray": [[207, 216, 220], [38, 50, 56]],
    }
    if not color_name:
        print("\n=== Colors available are:")
        for key, _ in colors.items():
            print("- " + key)
        return list(colors.keys())
    else:
        color = [
            [colors[color_name][i][j] / 255 for j in range(3)]
            for i in range(2)
        ]
        return color


def interp_col(color, n):
    """Linearly interpolate a color.
    Args:
            - color: list with two elements:
                    color[0] = lightest color variant (get_col('color_name')[0])
                    color[1] = darkest color variant (get_col('color_name')[1]).
            - n: number of desired output colors (n >= 2).
    Output:
            - lsc: list of n linearly scaled colors.
    """
    c = [
        np.interp(list(range(1, n + 1)), [1, n], [color[0][i], color[1][i]])
        for i in range(3)
    ]
    lsc = [[c[0][i], c[1][i], c[2][i]] for i in range(n)]
    return lsc


def check_distr(A, thre):
    mean = []
    std = []
    for a in A.T:
        mean.append(a.mean())
        std.append(a.std())
    l = []
    for i, (m, s) in enumerate(zip(mean, std)):
        if m < thre or m - 3 * s < 0:
            l.append(i)
    return l


def correct(A, thre):
    l = check_distr(A, thre)
    A[:, l] = np.zeros((A.shape[0], len(l)), dtype=float)
    return A


def gsa_box(path, thre, index_i, index_ij, ylab, savefig=False):
    ST = np.loadtxt(path + "STi.txt", dtype=float)
    S1 = np.loadtxt(path + "Si.txt", dtype=float)
    S2 = np.loadtxt(path + "Sij.txt", dtype=float)

    ST = correct(ST, thre)
    S1 = correct(S1, thre)
    S2 = correct(S2, thre)

    df_ST = pd.DataFrame(data=ST, columns=index_i)
    df_S1 = pd.DataFrame(data=S1, columns=index_i)
    df_S2 = pd.DataFrame(data=S2, columns=index_ij)

    plt.style.use("seaborn")
    gs = grsp.GridSpec(2, 2)
    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(2 * width, 2 * height / 2))

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
    if savefig:
        plt.savefig(path + ylab + "_box.pdf", bbox_inches="tight", dpi=1000)
    else:
        plt.show()
    return


def gsa_donut(path, thre, index_i, ylab, savefig=False):
    ST = np.loadtxt(path + "STi.txt", dtype=float)
    S1 = np.loadtxt(path + "Si.txt", dtype=float)

    ST = correct(ST, thre)
    S1 = correct(S1, thre)

    S1_mean = np.mean(S1, axis=0)
    ST_mean = np.mean(ST, axis=0)

    sum_s1 = S1_mean.sum()
    sum_st = ST_mean.sum()
    ho = sum_st - sum_s1
    x_si = np.array(list(S1_mean) + [ho])
    x_sti = ST_mean

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

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
    axes[0].set_title("S1", fontsize=12)

    axes[1].pie(
        x_sti,
        radius=1,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[1].set_title("ST", fontsize=12)

    plt.figlegend(
        wedges, index_i + ["higher-order int."], ncol=5, loc="lower center"
    )

    if savefig:
        plt.savefig(path + ylab + "_donut.pdf", bbox_inches="tight", dpi=1000)
    else:
        plt.show()
    return


def plot_dataset(Xdata, Ydata, xlabels, ylabels, color="C0"):
    """Plot Y high-dimensional dataset by pairwise plotting its features against each X dataset's feature.
    Args:
            - Xdata: n*m1 matrix
            - Ydata: n*m2 matrix
            - xlabels: list of m1 strings representing the name of X dataset's features
            - ylabels: list of m2 strings representing the name of Y dataset's features.
    """
    height = 9.36111
    width = 5.91667
    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    out_dim = Ydata.shape[1]
    fig, axes = plt.subplots(
        nrows=out_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(2 * width, 2 * height / 3),
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(
            Xdata[:, i % in_dim], Ydata[:, i // in_dim], fc=color, ec=color
        )
        inf = min(Xdata[:, i % in_dim])
        sup = max(Xdata[:, i % in_dim])
        mean = 0.5 * (inf + sup)
        delta = sup - mean
        if i // in_dim == out_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim])
            axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
        if i % in_dim == 0:
            axis.set_ylabel(ylabels[i // in_dim])
    plt.suptitle("Sample dimension = {} points".format(sample_dim))
    plt.show()
    return
