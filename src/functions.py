import json
import re
import logging as log

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch import is_tensor
from scipy.stats import pearsonr


def mean_abs(x):
    return np.mean(np.absolute(x), 0)


def max_abs(x):
    return np.max(np.absolute(x), 0)


def center(X):
    return X - np.mean(X, axis=0)


def decorrelate(X):
    newX = center(X)
    cov = X.T.dot(X) / float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    return decorrelated


def whiten(X):
    newX = center(X)
    cov = X.T.dot(X) / float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    # Rescale the decorrelated data
    whitened = decorrelated / np.sqrt(eigVals + 1e-5)
    return whitened


def load_csv(file="../../ADNI_corticalThichness.csv", cortical_var="Schaefer_200_7"):
    data = pd.read_csv(file)
    # string to list of float
    data[cortical_var] = data[cortical_var].apply(lambda s: json.loads(s))

    return data


def prepare(data, predict_vars, cortical_var="Schaefer_200_7", demographic_vars=[]):
    y = data[predict_vars].values
    X = pd.DataFrame(data[cortical_var].to_list()).values
    # if demographic variables are to be included
    if len(demographic_vars) != 0:
        X = np.concatenate([X, data[demographic_vars]], axis=1)
    # TODO : Add an option to include diagnosis as a input variable encoded either as a score, or as one hot vectors.
    idx = np.arange(X.shape[0])

    return X, y, idx


def annotate(data, **kws):
    r, p = pearsonr(data["y_pred"], data["y_test"])
    ax = plt.gca()
    ax.text(0.05, 0.8, "r={:.2f}, p={:.2g}".format(r, p), transform=ax.transAxes)


def cv_corr_to_dataframe(model_corr, predict_vars, model_name=""):
    corr_dict = {}
    for i, key in enumerate(predict_vars):
        corr_dict[key] = [fold[i] for fold in model_corr]
    df_cv = pd.DataFrame(corr_dict)
    df_cv["Model"] = model_name
    return df_cv


def csv_to_list(filename):
    lst = pd.read_csv(filename, header=None)
    return lst[0].values.tolist()


def set_zero(
    X, featname_list, per_network=True, per_roi=False, per_custom=None, string_idx=1, retention=True,
):
    # TODO: The mean values were computed on all datapoints, might be of use to only compute it over the training/test sets only.
    # Note that since we normalized the data at the beginning, the average values will be around zero (or very small), so that we could have just set them to zero instead.

    if per_network and not per_roi:
        # 7 networks
        subset_namelist = list(set([ft.split("_")[string_idx] for ft in featname_list]))

    elif per_roi and not per_network:
        # 47 regions
        pattern = r"_\d+"
        subset_namelist = list(set([re.sub(pattern, "", ft) for ft in featname_list]))

    elif not per_network and not per_network and per_custom is None:
        # global average for all features except the one of interest
        subset_namelist = featname_list

    elif not per_network and not per_network and per_custom is not None:
        subset_namelist = per_custom

    # Implement retention and elimination of features

    for ft_keep in subset_namelist:
        if is_tensor(X):
            X_subset = X.clone().detach()
        else:
            X_subset = X.copy()

        if not per_network and not per_network and per_custom is None: 
            keepmap = np.array(list(map(lambda x: ft_keep == x, featname_list)))
        elif not per_network and not per_network and per_custom is not None: 

            cluster_keep = ft_keep  # cluster name since input is a dict
            ft_keep = per_custom[cluster_keep]
            _, _, b_indexes = np.intersect1d(ft_keep, featname_list, return_indices=True)
            keepmap = np.zeros(len(featname_list), dtype=bool)
            keepmap[b_indexes] = True
        else: 
            keepmap = np.array(list(map(lambda x: ft_keep in x, featname_list)))
            
        # to use a boolean mask on a numpy.ndarray, it is imperative that the mask itself is a numpy.ndarray
        log.info(f"FEATURES OF INTEREST : {ft_keep}")
        
        if retention:
            X_subset[:, ~keepmap] = 0
        else : 
            X_subset[:, keepmap] = 0

        yield X_subset, ft_keep


def cluster_from_dendrograms(dendro): 
    cluster_dict = {}

    for cluster in list(set(dendro['leaves_color_list'])): 
        bool_cluster = [c==cluster for c in dendro['leaves_color_list']]
        lst_clusters = dendro['ivl']

        cluster_dict[cluster] = np.array(lst_clusters)[np.array(bool_cluster)]

    return cluster_dict


def complexity_torch(model):
    total_parameters=0
    for params in list(model.parameters()):
        nn=1
        for s in list(params.size()):
            nn = nn*s
        total_parameters += nn
    return total_parameters


def complexity_scikit_lr(model):
    total_parameters = 0
    total_parameters += model.coef_.shape[0]*model.coef_.shape[1]
    total_parameters += len(model.intercept_)
    return total_parameters


def complexity_scikit_mlp(model):
    total_parameters = 0
    for coefs, inters in zip(model.coefs_,model.intercepts_) :
        total_parameters += coefs.shape[0]*coefs.shape[1]
        total_parameters += len(inters)
    return total_parameters
    
    
def complexity_xgboost(model):
    trees = model.get_dump()
    total_leaves = 0
    for tree in trees:
        num_leaves = tree.count("leaf")
        total_leaves += num_leaves
    return total_leaves


def multiply_weights(weight_list): 
    result = np.array([])

    for w in reversed(weight_list):  
        if len(result) == 0:
            result = w
        else: 
            result = result.dot(w)     

    return result

def compute_tree_importance(model):

    df_xgboost_W = pd.DataFrame.from_dict(model.get_score(importance_type='weight'), orient='index').rename(columns={0:'xgboost_weight'})
    df_xgboost_logW = (df_xgboost_W.apply(np.log10)-df_xgboost_W.apply(np.log10).min())/(df_xgboost_W.apply(np.log10).max()-df_xgboost_W.apply(np.log10).min())
    df_xgboost_W = (df_xgboost_W-df_xgboost_W.min())/(df_xgboost_W.max()-df_xgboost_W.min())
    df_xgboost_logW = df_xgboost_logW.rename(columns={'xgboost_weight': 'xgboost_log(weight)'})

    df_xgboost_G = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient='index').rename(columns={0:'xgboost_gain'})
    df_xgboost_logG = (df_xgboost_G.apply(np.log10)-df_xgboost_G.apply(np.log10).min())/(df_xgboost_G.apply(np.log10).max()-df_xgboost_G.apply(np.log10).min())
    df_xgboost_G = (df_xgboost_G-df_xgboost_G.min())/(df_xgboost_G.max()-df_xgboost_G.min())
    df_xgboost_logG = df_xgboost_logG.rename(columns={'xgboost_gain': 'xgboost_log(gain)'})

    df_xgboost = pd.concat([df_xgboost_W.T, df_xgboost_G.T, df_xgboost_logW.T, df_xgboost_logG.T])

    return df_xgboost

def rank_dataframe(df):
    # Initialize a DataFrame to store the ranks
    ranks = pd.DataFrame(index=df.index, columns=df.columns)

    # Loop through each row of the DataFrame
    for index, row in df.iterrows():
        # Sort the row data in descending order and get the indices
        sorted_indices = row.argsort()[::-1]

        # Assign ranks based on the sorted indices
        ranks.loc[index, sorted_indices] = range(1, len(sorted_indices) + 1)

    # Add the ranks as new columns to the DataFrame
    ranks.columns = ['Rank_' + col for col in df.columns]
    
    return ranks


def missing_results_dict_to_dataframe(dict_output, target_names=[]):

    df_results = pd.json_normalize(dict_output)

    for col in df_results.columns[df_results.columns.str.startswith("results")]:
        # Create new columns for each subcategory
        for i, subcategory in enumerate(target_names):
            new_col_name = f"{col}.{subcategory}"
            df_results[new_col_name] = df_results[col].apply(lambda x: x[i] if x is not None else None)

            print(new_col_name)

        # Optionally, drop the original list column
        df_results.drop(columns=col, inplace=True)

    # Show the expanded DataFrame
    return df_results


def generate_metric_table(
    results_list,
    targets,
    metric_name,
    source="Adjusted",
    float_format="%.3f",
    csv_filename=None,
    sort_order="ascending"
):
    """
    Create a LaTeX and CSV table for a single metric across targets, models, and imputers,
    including mean ± std for performance, imputation time, and fitting time.

    Parameters
    ----------
    results_list : list of dict
        List of experiment results.
    targets : list of str
        Target names (e.g., ['ADNI_MEM', 'ADNI_EF', 'ADNI_VS', 'ADNI_LAN']).
    metric_name : str
        Metric to extract (e.g., 'mae_score').
    source : str
        'Adjusted' or 'Original'.
    float_format : str
        Format for floats (e.g., '%.3f').
    csv_filename : str or None
        If provided, saves the table to CSV.
    sort_order : str
        'ascending' or 'descending' for sorting by mean.

    Returns
    -------
    df : pd.DataFrame
        Final formatted DataFrame.
    latex_table : str
        LaTeX-formatted table string.
    """
    rows = []
    version_key = "results_adj" if source.lower() == "adjusted" else "results_org"

    for res in results_list:
        result_block = res.get(version_key)
        if result_block is None:
            continue

        metric_values = result_block.get(metric_name)
        if metric_values is None:
            continue

        if len(metric_values) != len(targets):
            continue

        ordinal_imputer = res["params"].get("ordinal_imputer")
        continuous_imputer = res["params"].get("continuous_imputer")
        model = res["params"].get("model")

        values = np.array(metric_values, dtype=np.float64)
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Time metrics
        imp_times = np.array(res.get("imputation_time", []), dtype=np.float64)
        fit_times = np.array(res.get("fitting_time", []), dtype=np.float64)

        row = {
            "Ordinal Imputer": ordinal_imputer,
            "Continuous Imputer": continuous_imputer,
            "Model": model,
            "Mean": mean_val,
            "Mean ± SD": f"{mean_val:.3f} ± {std_val:.3f}",
            "Imputation Time": f"{imp_times.mean():.2f}" if imp_times.size > 0 else "N/A",
            "Fitting Time": f"{fit_times.mean():.2f}" if fit_times.size > 0 else "N/A"
        }

        row.update({target: val for target, val in zip(targets, values)})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns for display
    display_cols = (
        ["Ordinal Imputer", "Continuous Imputer", "Model"] +
        targets +
        ["Mean ± SD", "Imputation Time", "Fitting Time"]
    )
    df = df.sort_values(by="Mean", ascending=(sort_order == "ascending"))
    df = df[display_cols]

    df.drop_duplicates(subset=["Ordinal Imputer", "Continuous Imputer", "Model"] +
        targets +
        ["Mean ± SD",], inplace=True)

    # Save CSV if requested
    if csv_filename:
        df.to_csv(csv_filename, index=False)

    # Generate LaTeX table
    latex_table = df.to_latex(
        index=False,
        escape=False,
        float_format=float_format,
        caption=f"{metric_name.replace('_', ' ').upper()} across targets with timing info",
        label=f"tab:{metric_name}",
        longtable=False
    )

    return df, latex_table
