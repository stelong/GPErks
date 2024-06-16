import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def barplot(axis, df, colors):
    sns.barplot(
        data=df, x="Parameter", y="Value", hue="Index", errorbar="sd", palette=colors, ax=axis
    )
    axis.set_ylim(0, 1)


def boxplot(axis, df, colors):
    sns.boxplot(
        data=df, x="Parameter", y="Value", hue="Index", palette=colors, ax=axis
    )
    axis.set_ylim(0, 1)


def donut(axis, df, colors):
    df_ = pd.pivot_table(
        df, values="Value", index="Index", columns="Parameter", aggfunc="median"
    )

    n_colors = len(df_.columns.values)
    cmap = plt.get_cmap(colors, n_colors)
    outer_colors = [cmap(i) for i in range(n_colors)]
    in_c = [c[:3] + (0.7,) for c in outer_colors]
    inner_colors = [
        item for pair in zip(in_c, len(in_c) * [(0.0, 0.0, 0.0, 0.15)]) for item in pair
    ]

    st = df_.loc["ST"].to_numpy()
    s1 = df_.loc["S1"].to_numpy()
    s1ho = [item for pair in zip(s1, st - s1) for item in pair]

    wo, _ = axis.pie(
        st,
        radius=1,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )
    wi, _ = axis.pie(
        s1ho,
        radius=1 - 0.3,
        colors=inner_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )

    axis.set(aspect="equal")
    axis.legend(
        wo + [wi[1]],
        list(df_.columns.values) + ["Higher-order\ninteractions"],
        ncol=1,
        loc="center right",
        bbox_to_anchor=(1, 0, 0.5, 1),
        title="Parameter",
        title_fontsize=11,
    )


def heatmap(axis, df, colors):
    df_ = pd.pivot_table(
        df, values="Value", index="Index", columns="Parameter", aggfunc="median"
    )
    sns.heatmap(
        data=df_,
        vmin=0.0,
        vmax=1.0,
        cmap=colors,
        annot=True,
        fmt=".2f",
        linewidth=0.1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axis,
    )
