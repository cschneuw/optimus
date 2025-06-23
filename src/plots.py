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


# ---------------------------------------------------------------
# Imputation distribution plots
# ---------------------------------------------------------------


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


def plot_ecdf_grid(ecdf_data_dict, features_of_interest, imputer_names, savefolder=None):

    sns.set_style("white")
    sns.set_context("talk")

    selected_features = [f for f in features_of_interest if f in ecdf_data_dict]
    num_features = len(selected_features)
    ncols = 3
    nrows = (num_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()

    for i, feature_name in enumerate(selected_features):
        ecdf_data = ecdf_data_dict[feature_name]

        sns.lineplot(
            data=ecdf_data,
            x="Value",
            y="ECDF",
            hue="Method",
            ax=axes[i],
            palette=sns.color_palette("Set1"),
            linewidth=3,
            alpha=0.8
        )

        axes[i].set_title(feature_name)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('ECDF')
        axes[i].get_legend().remove()
        sns.despine()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [imputer_names.get(l, l) for l in labels]

    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(labels), 4),
        title="Imputation Method",
        frameon=False,
        fontsize=16,
        title_fontsize=18
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.85])

    if savefolder is not None:
        plt.savefig(f"{savefolder}_ecdf_plot.png", format="png", bbox_inches='tight')

    plt.show()


def plot_metric_heatmap(
    metric_df,
    metric_name,
    imputer_names,
    features_of_interest=None,
    annotations_df=None,
    savefolder=None,
    cmap="vlag",
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    figsize=(14, 6),
    dpi=300,
    font_scale=1.2,
    annot_format_func=None,
    keys_list=None,
    full_palette=None
):
    """
    Plots a sorted, grouped heatmap with optional annotations and color grouping by modality.
    """

    sns.set_context("talk", font_scale=font_scale)
    sns.set_style("white")

    # Subset and sort features
    if features_of_interest is not None:
        metric_df = metric_df.loc[features_of_interest]
        if annotations_df is not None:
            annotations_df = annotations_df.loc[features_of_interest]

    # Sort columns (features) by mean metric across imputers
    feature_means = metric_df.mean(axis=0)
    sorted_features = feature_means.sort_values().index.tolist()
    metric_df = metric_df[sorted_features]
    if annotations_df is not None:
        annotations_df = annotations_df[sorted_features]

    # Create modality group colors
    if keys_list is not None and full_palette is not None:
        cat_palette = {}
        new_key_list = []
        for i, mod in enumerate(["MRIth", "RNA", "CSF", "DNA", "ADNI_cog"]):
            is_modal = [k == mod for k in keys_list]
            new_key_list.extend(np.array(keys_list)[is_modal])
            cat_palette[mod] = list(full_palette.values())[i]

        # Make sure row_colors aligns with sorted feature list
        feature_to_modality = dict(zip(keys_list, keys_list))  # keys_list is aligned with metric_df.columns
        col_colors = pd.Series(sorted_features).map(lambda x: cat_palette.get(feature_to_modality.get(x, ""), "grey"))
    else:
        col_colors = None
        cat_palette = {}

    # Format annotations
    if annotations_df is not None and annot_format_func is not None:
        annotations_df = annotations_df.applymap(annot_format_func)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.heatmap(
        metric_df.transpose(),
        annot=annotations_df.transpose() if annotations_df is not None else None,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="lightgrey",
        square=False,
        cbar_kws={'label': metric_name, 'ticks': cbar_ticks} if cbar_ticks else {'label': metric_name},
        ax=ax
    )

    # Clean tick labels
    ax.set_yticklabels([
    imputer_names.get(label.get_text(), label.get_text()).replace("_", " ")
    for label in ax.get_yticklabels()
], rotation=0, fontsize=12)
    
    ax.set_xticklabels(
        [label.get_text().replace("_", " ") for label in ax.get_xticklabels()],
        rotation=45, ha='right', fontsize=11
    )

    ax.set_ylabel("Imputation Method", fontsize=13)
    ax.set_xlabel("Feature", fontsize=13)

    # Add legend for group colors (feature modality)
    if col_colors is not None:
        modality_legend = [Patch(facecolor=color, label=label) for label, color in cat_palette.items()]
        ax.legend(handles=modality_legend, title="Feature Type", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    # Save if requested
    if savefolder is not None:
        fname = f"{savefolder}_{metric_name.lower().replace(' ', '_')}_heatmap.png"
        plt.savefig(fname, bbox_inches="tight", format="png")

    plt.show()


# ---------------------------------------------------------------
# Feature importance bar plots
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# Brain region importance using Schaefer's atlas
# ---------------------------------------------------------------


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