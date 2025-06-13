import time

import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pytorch_tabnet.tab_model import TabNetRegressor

from src.dataset import DemographicAdjustmentTransformer

def correlation(y1, y2):
    """Compute pearson's correlation between two inputs of the same shape with the computed correlation and estimated p-value for non-correlation testing.

    Args:
        y1 (torch.tensor or numpy.array): First matrix.
        y2 (torch.tensor or numpy.array): Second matrix.

    Returns:
        list: List of pearon's correlation coefficients and their estimated p-value
    """
    if torch.is_tensor(y1):
        y1 = y1.detach().numpy()
    elif isinstance(y1, pd.DataFrame):
        y1 = y1.values
    if torch.is_tensor(y2):
        y2 = y2.detach().numpy()
    elif isinstance(y1, pd.DataFrame):
        y2 = y2.values

    elif len(y1.shape) > 1 or len(y2.shape) > 1:
        corrlist = [pearsonr(y1[:, i], y2[:, i])[0] for i in range(y1.shape[1])]
    else : 
        corrlist = pearsonr(y1, y2)[0]

    return np.array(corrlist)



def loglikelihood(y1, y2):
    n = y1.shape[0] # num_obs
    k = y1.shape[1] # num_features
    residuals = y1 - y2

    ll = -(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(residuals.dot(residuals) / n)

    return ll


def AIC_BIC(y1, y2):
    ll = loglikelihood(y1, y2)
    n = y1.shape[0]# num_obs
    k = y1.shape[1]# num_features + 1

    AIC = (-2 * ll) + (2 * k)
    BIC = (-2 * ll) + (k * np.log(n))

    return AIC, BIC


def compute_all_metrics(y_test, y_pred): 
    # Metrics computed in adjusted space
    mse_score = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    mae_score = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")
    explained_variance = explained_variance_score(y_test, y_pred, multioutput="raw_values")

    try: 
        corr = correlation(y_pred, y_test)
    except: 
        print("Problem when computing correlation!")
        if type(y_test) == type(y_pred): 
            print("Type of inputs is different.")
            corr = None

    return mse_score, mae_score, r2, explained_variance, corr


def train_imputer_model(
    df_X_train, df_X_test, df_y_train, df_y_test,
    c_train, c_test,
    ordinal_model, name_ordinal_imputer, 
    continuous_model, name_continuous_imputer, 
    model, name_model, 
    imputer_model=None, name_imputer=None, 
    separate_imputers=True,
    ordinal_features = ['APOE_epsilon2', 'APOE_epsilon3', 'APOE_epsilon4']
): 
    # Define which columns are ordinal and which are continuous
    continuous_features = [col for col in df_X_train.columns if col not in ordinal_features]

    # Check if a general imputer model (like MissForest or MICEForest) is provided
    if imputer_model is not None and name_imputer is not None and not separate_imputers:
        # If `imputer_model` can handle both categorical and continuous data types
        print(f"Using general imputer model: {name_imputer}")

        # Ensure that ordinal columns are marked as categorical
        df_X_train = df_X_train.copy()
        df_X_test = df_X_test.copy()
        
        for col in ordinal_features:
            df_X_train[col] = df_X_train[col].astype("category")
            df_X_test[col] = df_X_test[col].astype("category")

        # Create a pipeline with the general imputer
        pipeline = Pipeline(steps=[
            (name_imputer, imputer_model)
        ])

        # Fit and transform the entire dataset with the general imputer
        pipeline.fit(df_X_train)
        X_train_imputed = pipeline.transform(df_X_train)
        X_test_imputed = pipeline.transform(df_X_test)

        # Convert transformed output back to DataFrame with original column names
        df_X_train_imputed = pd.DataFrame(X_train_imputed, columns=df_X_train.columns)
        df_X_test_imputed = pd.DataFrame(X_test_imputed, columns=df_X_test.columns)

    else:
        # Separate imputers for ordinal and continuous data
        print("Using separate imputers for ordinal and continuous data.")

        df_X_train = df_X_train.copy()
        df_X_test = df_X_test.copy()

        # Continuous Imputation Transformer (Example: SimpleImputer)
        continuous_imputer = Pipeline([
            (name_continuous_imputer, continuous_model),
        ])

        # Ordinal Imputation Transformer (Example: KNN Imputer)
        ordinal_imputer = Pipeline([
            (name_ordinal_imputer, ordinal_model)
        ])

        # Create a ColumnTransformer to apply the appropriate imputer to each type of variable
        preprocessor = ColumnTransformer(
            transformers=[
                ('ordinal', ordinal_imputer, ordinal_features),
                ('continuous', continuous_imputer, continuous_features)
            ],
            remainder='passthrough'
        )

        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])

         # Separate imputers for ordinal and continuous data
        # Fit and transform, then convert back to DataFrame with original column names

        if df_X_train.isna().any().any():
                
            start = time.time()
            pipeline.fit(df_X_train)
            end = time.time()

            impute_model_time = end - start

            X_train_imputed = pipeline.transform(df_X_train)
            df_X_train_imputed = df_X_train.copy()
            df_X_train_imputed[ordinal_features+continuous_features] = X_train_imputed
        else :
            print("No NaN in train data -> Keep as it is. ")
            df_X_train_imputed = df_X_train
            
            impute_model_time = None

        # Transform the test set
        if df_X_test.isna().any().any(): 
            X_test_imputed = pipeline.transform(df_X_test)
            df_X_test_imputed = df_X_test.copy()
            df_X_test_imputed[ordinal_features+continuous_features] = X_test_imputed
        else : 
            print("No NaN in test data -> Keep as it is. ")
            df_X_test_imputed = df_X_test

    # Demographics adjustment for y
    demographic_adjustment_y = DemographicAdjustmentTransformer()
    y_train_adjusted = demographic_adjustment_y.fit_transform(df_y_train, c_train)
    y_test_adjusted = demographic_adjustment_y.transform(df_y_test, c_test)

    # Demographics adjustment for X
    demographic_adjustment_X = DemographicAdjustmentTransformer(categorical_columns=ordinal_features)
    X_train_adjusted = demographic_adjustment_X.fit_transform(df_X_train_imputed, c_train)
    X_test_adjusted = demographic_adjustment_X.transform(df_X_test_imputed, c_test)

    # Standardize only continuous features
    scaler = StandardScaler()

    X_train_adjusted[continuous_features] = scaler.fit_transform(X_train_adjusted[continuous_features])
    X_test_adjusted[continuous_features] = scaler.transform(X_test_adjusted[continuous_features])

    # Perform prediction and save variables
    start = time.time()

    if isinstance(model, TabNetRegressor): 
        X_train_adjusted = X_train_adjusted.values
        y_train_adjusted = y_train_adjusted.values

        X_test_adjusted = X_test_adjusted.values
        #y_test_adjusted = y_test_adjusted.values
    
    model.fit(X_train_adjusted, y_train_adjusted) 
    end = time.time()

    predict_model_time = end - start

    y_pred_adjusted = model.predict(X_test_adjusted)

    if isinstance(y_pred_adjusted, pd.DataFrame) and all(
    isinstance(col, str) and col.endswith("_prediction") for col in y_pred_adjusted.columns
    ):
        y_pred_adjusted.columns = [col.replace("_prediction", "") for col in y_pred_adjusted.columns]
    else:
        y_pred_adjusted = pd.DataFrame(y_pred_adjusted, columns=y_test_adjusted.columns)

    # Metrics computed in original space
    y_pred = demographic_adjustment_y.inverse_transform(y_pred_adjusted, c_test)

    params = {
        "ordinal_imputer": name_ordinal_imputer, 
        "continuous_imputer": name_continuous_imputer, 
        "model": name_model, "train_shape" : X_train_adjusted.shape, 
        "test_shape": X_test_adjusted.shape
    }
    
    if df_X_test.shape[0] != 1: 

        # Metrics computed in adjusted space
        mse_score_adj, mae_score_ajd, r2_adj, explained_variance_adj, corr_adj = compute_all_metrics(y_test_adjusted.values, y_pred_adjusted)

        results_adj = {
            "mse_score": mse_score_adj, 
            "mae_score":mae_score_ajd, 
            "r2":r2_adj, 
            "explained_variance":explained_variance_adj, 
            "corr":corr_adj, 
        }

        mse_score, mae_score, r2, explained_variance, corr = compute_all_metrics(df_y_test.values, y_pred)

        results_org = {
            "mse_score": mse_score, 
            "mae_score": mae_score, 
            "r2": r2, 
            "explained_variance": explained_variance, 
            "corr": corr, 
        }

    else : 
        print("Saving predictions in dict!")
        results_adj = {
            "y_pred": y_pred_adjusted.values, 
            "y_test": y_test_adjusted.values,
        }

        results_org = {
            "y_pred": y_pred.values, 
            "y_test": df_y_test.values,
        }


    dict_results = {
        "params": params, 
        "imputation_time": impute_model_time,
        "fitting_time": predict_model_time, 
        "results_adj": results_adj, 
        "results_org": results_org
        }

    return dict_results

