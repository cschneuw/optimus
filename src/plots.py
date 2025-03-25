import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def annotate(data, **kws):
    r, p = np.corrcoef(data['y_test'], data['y_pred'])[0, 1], 0
    ax = plt.gca()
    ax.text(.05, .8, f'r = {r:.2f}', transform=ax.transAxes)

    
# Function to clean up cortical y-tick labels
def clean_ytick_label(label):
    label = label.replace("_epsilon", "").replace("_Cortical", "")
    label = label.replace("LH", "Left").replace("RH", "Right")
    label = label.replace("LH", "Left").replace("RH", "Right")
    label = label.replace("LH", "Left").replace("RH", "Right")
    label = label.replace("_", " ")  # Remove remaining underscores
    if len(label.split(" || ")) > 1: 
        label = label.split(" || ")[1]
    
    return label


def plot_pred_correlation(
    y_pred, y_test, idx_test, data, sample_vars, predict_vars, ax=None, figure_savename=None
):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().numpy()
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.numpy()

    df_y_test = (
        pd.DataFrame(y_test, columns=predict_vars)
        .set_index(idx_test)
        .melt(ignore_index=False)
        .reset_index(drop=False)
    )
    df_y_pred = (
        pd.DataFrame(y_pred, columns=predict_vars)
        .set_index(idx_test)
        .melt(ignore_index=False)
        .reset_index(drop=False)
    )

    df_y = (
        df_y_test.merge(
            df_y_pred.drop(columns=["index", "variable"]),
            left_index=True,
            right_index=True,
        )
        .rename(columns={"value_x": "y_test", "value_y": "y_pred"})
        .merge(data[sample_vars].reset_index(drop=False), on="index")
    )

    g = sns.lmplot(
        data=df_y,
        x="y_test",
        y="y_pred",
        col="variable",
        col_wrap=4,
        x_ci="sd",
        facet_kws=dict(sharex=False, sharey=False),
    )

    g.map_dataframe(annotate)
    if figure_savename is not None:
        g.savefig(figure_savename)
    return g


def plot_cv_train_history(cv_history, plt_traj=False, figure_savename=None):
    df = pd.json_normalize(cv_history)
    df = df.explode(column=list(cv_history.keys()))

    df["train_pearson"] = df["train_pearson"].apply(np.mean)
    df["test_pearson"] = df["test_pearson"].apply(np.mean)

    df["train_r2"] = df["train_r2"].apply(np.mean)
    df["test_r2"] = df["test_r2"].apply(np.mean)

    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "fold"})
    df["epoch"] = [*range(0, df.fold.value_counts()[0])] * (df.fold.max() + 1)
    df = df.melt(
        value_vars=[
            "train_loss",
            "test_loss",
            "train_pearson",
            "test_pearson",
            "train_r2",
            "test_r2",
        ],
        id_vars=["epoch", "fold"],
    )

    df[["set", "metric"]] = df["variable"].str.split("_", 1, expand=True)

    if plt_traj:
        g = sns.relplot(
            data=df,
            x="epoch",
            y="value",
            style="fold",
            hue="set",
            col="metric",
            kind="line",
            facet_kws=dict(sharex=False, sharey=False),
        )
    else:
        g = sns.relplot(
            data=df,
            x="epoch",
            y="value",
            hue="set",
            col="metric",
            kind="line",
            facet_kws=dict(sharex=False, sharey=False),
        )
    if figure_savename is not None:
        g.savefig(figure_savename)

    return g


def plot_correlation_bars(
    df_corr, predict_vars, subtract_mean=False, font_size=None, height=6, figure_savename=None,
):
    if subtract_mean:
        df_temp = df_corr
        df_corr = df_corr.sub(
            df_corr[df_corr["Model"] == df_corr["Model"].iloc[0]].mean()
        )
        df_corr["Model"] = df_temp["Model"]

    df_corr_melt = df_corr.melt(value_vars=predict_vars, id_vars="Model")
    df_corr_melt["Group"] = df_corr.melt(value_vars=predict_vars, id_vars="Model")[
        "Model"
    ].apply(lambda s: re.sub(r"_\d+", "", s))
    ticks = np.arange(0, df_corr_melt.Model.drop_duplicates().shape[0], 1)[
        ~df_corr_melt.drop_duplicates("Model", keep="first").Group.duplicated()
    ]
    labels = df_corr_melt.Model[~df_corr_melt.Group.duplicated()].tolist()

    g = sns.catplot(
        data=df_corr_melt,
        y="Model",
        x="value",
        hue="Group",
        col="variable",
        kind="bar",
        height=height,
        aspect=0.4,
        sharex=False,
        legend=False,
        dodge=False,
        errwidth=1,
    )

    g.set(yticks=ticks)
    if font_size is None:
        g.set_yticklabels(labels)
    else:
        g.set_yticklabels(labels, fontdict={"fontsize": font_size})
    g.fig.tight_layout()

    if figure_savename is not None:
        plt.savefig(figure_savename, bbox_inches = "tight")
    return g

def plot_random_histplots(imputed_X_lst, imputed_name_lst, df_X0 = None, hue_col=None, n_plots = 10, figsize = (20, 30), palette=None): 

    if df_X0 is None: 
        df_X0 = imputed_X_lst[0]

    fig, axes = plt.subplots(nrows=n_plots, ncols=len(imputed_X_lst), figsize=figsize)

    for i in range(len(imputed_X_lst)):
        for j in range(n_plots): 
            plot_X = imputed_X_lst[i]
            if imputed_name_lst[i] != "Original": 
                plot_X = plot_X[df_X0.isna()]
            sns.histplot(plot_X, x=df_X0.columns.values[j], hue=hue_col, ax=axes[j][i], palette = palette)
            axes[j][i].set_xlabel('')

    for ax, col in zip(axes[0], imputed_name_lst):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], df_X0.columns.values):
        ax.set_ylabel(row, rotation=90, size='large')
