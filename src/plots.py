import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import pearsonr

from nilearn.image import new_img_like
from nilearn import plotting
import nibabel as nib


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


def plot_top_features(feature_importances_list, feature_names, top_n=20, hue=None, palette="viridis"):
    for idx, importances in enumerate(feature_importances_list):
        # Create a DataFrame to pair features with their importances
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Sort by importance and select top N features
        top_features = feature_df.nlargest(top_n, "Importance")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_features,
            x="Importance",
            y="Feature",
            hue=hue,
            palette=palette
        )
        plt.title(f"Top {top_n} Features for Array {idx + 1}")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

# Step 3: Plotting helper function
def plot_importance(importances, title, top_n=20, hue=None, palette="viridis"):
    """Plot feature importances."""
    importances = importances.head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="Importance", y="Feature", data=importances, hue=hue, palette=palette, orient="h"
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Mean Permutation Importance (Pearson Correlation)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

# Custom scoring function for Pearson correlation (handles multi-target)
def pearson_scorer(estimator, X, y):
    """Compute Pearson correlation for multi-target predictions."""
    predictions = estimator.predict(X)
    correlations = []
    for i in range(y.shape[1]):  # Iterate over targets
        correlations.append(pearsonr(predictions[:, i], y.iloc[:, i])[0])
    return np.array(correlations)


def plot_brain_feature_importances(label='MMSE', head=20, method=None, savefilename=None):
    atlas_path = '../figures/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'  # Path to the atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Load region label map from data/AD/Schaefer2018_200Parcels_7Networks_order_info.txt, each line is feature_name,region_label
    region_label_map = pd.read_csv('../figures/Schaefer2018_200Parcels_7Networks_order_info.txt')
    region_label_map = region_label_map.set_index('feature_name')['region_label']

    # Load feature importance
    df = pd.read_csv(f'../figures/processed/adni_feature_importances_{label}.csv')
    df['abs_importance'] = df['importance'].abs()  # Use absolute importance for ranking
    df_sorted = df.sort_values(by='abs_importance', ascending=False).head(head)

    # Map feature importance to brain regions
    # Assuming the atlas regions are labeled as integers (LH and RH have different IDs)
    importance_map = np.zeros_like(atlas_data)

    for _, row in df_sorted.iterrows():
        feature_name = row['feature']
        importance_value = row['importance']

        # Extract the region index
        region_label = region_label_map.get(feature_name, None)

        # Map the importance value to the atlas region
        importance_map[atlas_data == region_label] = importance_value

    # Create a new NIfTI image with the importance map
    importance_img = new_img_like(atlas_img, importance_map)

    # Plot the brain map using nilearn's plotting function
    plotting.plot_glass_brain(
        importance_img,
        colorbar=True,
        cmap="vlag",
        vmin=-6.5, vmax=6.5,
        display_mode='lyrz',
    )

    if savefilename is not None:
        plt.savefig(f'plots/adni_brain_fi_{label}.svg')
    # Show the plot
    plotting.show()