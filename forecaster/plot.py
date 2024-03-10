import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"


def create_scatter_plot_matrix(
    pdf: pd.DataFrame, hue: str, diag_sharey: bool = False, **kwargs
) -> None:
    grid = sns.PairGrid(pdf, hue=hue, diag_sharey=diag_sharey, **kwargs)
    grid.map_upper(sns.scatterplot, s=15)
    grid.map_lower(sns.kdeplot)
    grid.map_diag(sns.kdeplot, lw=2)
    grid.add_legend()


def create_bar_plot(
    pdf: pd.DataFrame, category: str, target: str, threshold: int, column: str = ""
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(column.capitalize())
    axes[0].set_title("Count plot")
    order = pdf[category].value_counts().sort_values(ascending=False).index
    subplot = sns.countplot(ax=axes[0], y=category, data=pdf, orient="v", order=order)
    subplot.axvline(threshold, color="grey", linestyle="--")
    axes[0].text(threshold, 3, f"{threshold}", rotation=90, verticalalignment="center")

    axes[1].set_title(f"Bar plot - based on {target}")
    sns.barplot(ax=axes[1], y=category, x=target, data=pdf, order=order)
