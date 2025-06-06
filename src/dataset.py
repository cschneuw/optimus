import os
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------ Load datasets ------------------------------ #

def load_cognition():
    targets = ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS']
    df_uwnpsychsum = pd.read_csv("../data/adnimerge/uwnpsychsum.csv").replace({"sc":"bl","Female":0, "Male":1}).drop(columns=["ADNI_EF"]).rename(columns={"ADNI_EF2" : "ADNI_EF"})
    
    return df_uwnpsychsum, targets

def load_brain_imaging():
    df_all = pd.read_csv("../../../dataset/ADNI/ALL_3.csv").replace({"sc":"bl","Female":0, "Male":1})

    return df_all

def load_transcriptomics():
    df_counts = pd.read_csv("../data/microarray/gene_counts.csv", index_col=0)
    df_samples = pd.read_csv("../data/microarray/samples.csv", index_col=0)
    df_genes = pd.read_csv("../data/microarray/genes.csv", index_col=0)
    df_dge = pd.read_csv("../data/microarray/dge_results_limma.csv", index_col=0)

    df_counts = df_counts.transpose()
    df_counts = df_counts.loc[:, df_counts.columns.isin(df_dge.Symbol)]

    df_counts = df_counts.merge(df_samples[["SubjectID", "Visit"]], left_index=True, right_index=True)
    df_counts = df_counts.reset_index(drop=True).drop_duplicates()
    df_counts = df_counts.rename(columns={"Visit":"VISCODE"})

    return df_counts, df_dge

def load_CSF():
    df_adni_roche_elecsys = pd.read_csv("../data/ida/UPENNBIOMK_ROCHE_ELECSYS_02Jun2025.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_adni_roche_elecsys = df_adni_roche_elecsys.rename(columns={"VISCODE2": "VISCODE", "PHASE": "ORIGPROT", "ABETA42": "ABETA"})

    df_upennbiomk12_2020 = pd.read_csv("../data/adnimerge/upennbiomk12_2020.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_upennbiomk_master = pd.read_csv("../data/adnimerge/upennbiomk_master.csv").replace({"sc": "bl", "Female": 0, "Male": 1})

    df_upennbiomk12_2020["BATCH"] = "UPENNBIOMK12_2020"
    df_upennbiomk_master = pd.concat([df_upennbiomk_master, df_upennbiomk12_2020])

    df_upenn_all = pd.concat([
        df_upennbiomk_master,
        df_adni_roche_elecsys
    ], axis=0)

    df_upenn_all.sort_values(by=["RID", "VISCODE"], inplace=True)
    df_upenn_all.drop_duplicates(subset=["RID", "VISCODE"], keep="last", inplace=True)

        # Define which biomarker maps to which column
    censor_column_map = {
        "Abeta42": "ABETA",
        "Tau": "TAU",
        "PTau": "PTAU",
        "Ptau": "PTAU",  # handle case inconsistency
    }

    def parse_censor_comment(comment, censored_column_map):
        """
        Parse a censor comment and return a dictionary mapping biomarker columns
        to imputed values (based on censoring thresholds).
        """
        if pd.isnull(comment):
            return {}

        result = {}
        tokens = comment.split(',')
        for token in tokens:
            token = token.strip()

            # Match expressions like "Abeta42>1700", "Tau<80"
            match = re.match(r"(Abeta42|Tau|PTau|Ptau)\s*([<>])\s*(\d+)", token)
            if match:
                biomarker, op, value = match.groups()
                column = censor_column_map.get(biomarker)
                value = float(value)
                if op == '>':
                    result[column] = value  # or value + epsilon
                elif op == '<':
                    result[column] = value  # or value - epsilon
            # Optionally handle known non-numeric entries (e.g. "Sample Hemolyzed")
            # else: you can log or ignore
        return result

    # Assuming your df has columns: COMMENT, ABETA, TAU, PTAU
    for idx, row in df_upenn_all.iterrows():
        if pd.isnull(row["COMMENT"]):
            continue
        updates = parse_censor_comment(row["COMMENT"], censored_column_map=censor_column_map)
        for col, val in updates.items():
            if pd.isnull(row[col]):
                df_upenn_all.at[idx, col] = val

    return df_upenn_all


def load_apoe():
    df_apoe_adnigo2 =  pd.read_csv("../data/adnimerge/apoego2.csv")
    df_apoe_adni3 = pd.read_csv("../data/adnimerge/apoe3.csv")

    df_apoe = pd.concat([df_apoe_adnigo2, df_apoe_adni3])
    df_apoe = df_apoe[["ORIGPROT", "RID", "SITEID", "APGEN1", "APGEN2"]]

    df_genotype = pd.concat([df_apoe, df_apoe[["APGEN1", "APGEN2"]].apply(lambda s: s.value_counts(), axis=1).fillna(0).add_prefix('APOE_epsilon')], axis=1).drop(columns=["SITEID", "APGEN1", "APGEN2"])

    return df_genotype


def load_pickle_data_palettes(drop_pet=True):
    # Define file paths
    file_paths = {
        "df_X": "df_X_original.pickle",
        "df_y": "df_y_original.pickle",
        "df_all": "df_all_original.pickle",
        "df_FinalCombination": "df_all.pickle",
        "dict_select": "select_features.pickle",
        "miss_mask": "filter_miss_mask.pickle"
    }

    # Load all pickle files
    data = {key: pd.read_pickle(f"../pickle/{path}") for key, path in file_paths.items()}
    
    # Convert miss_mask to list
    data["miss_mask"] = data["miss_mask"].tolist()

    # Extract selected features
    feature_keys = ["RNA", "CSF", "DNA", "MRIth", "MRIvol", "PET"]
    select_features = [data["dict_select"][key] for key in feature_keys]
    select_features_df = pd.DataFrame(select_features).T
    select_features_df.columns = feature_keys
    data["df_select_features"] = select_features_df
    
    # Define colormaps
    full_palette = {
        "orange": "#ff4b41", "yellow": "#ffaa41", "blue": "#75d8ff", 
        "cyan": "#d7d341", "purple": "#e59edd", "green": "#70d941"
    }
    
    data["colormaps"] = {
        "full_palette": full_palette,
        "gender_palette": {"0": full_palette["green"], "1": full_palette["purple"]},
        "dx_palette": {"CN": "#75d8ff", "MCI": "#ffcc92", "AD": "#ff4b41"}
    }

    if drop_pet: 
        data["df_X"] = data["df_X"][[col for col in data["df_X"].columns if not re.search(r'PET$|GM', col)]]

    return data

# ------------------------------ Modify dataset ------------------------------ #

def csv_to_list(filename):
    lst = pd.read_csv(filename, header=None)
    return lst[0].values.tolist()


def read_featurenames(remove_medial_wall=True):
    r_featurename="../../../dataset/ADNI/rh_mapping_names.csv"
    l_featurename="../../../dataset/ADNI/lh_mapping_names.csv"
    
    r_features = csv_to_list(r_featurename)
    l_features = csv_to_list(l_featurename)

    all_features = [None]*(len(l_features)+len(r_features))

    if remove_medial_wall: 
        all_features[::2] = l_features
        all_features[1::2] = r_features
        all_features = all_features[2:]

    all_features = list(map(lambda s : s.replace("7Networks_", ""), all_features))

    return all_features
    

def fill_age_gender(df, df_demog):

    if df.AGE.isna().any() or df.PTGENDER.isna().any(): 
        
        # Note that df_demog must contain the following columns: RID, PTGENDER, PTDOB
        df_demog["PTDOB"] = pd.to_datetime(df_demog["PTDOB"], format="%Y")
        merged_df = df.merge(df_demog[["RID", "PTGENDER", "PTDOB"]], on='RID', how='left').drop_duplicates(subset=["SubjectID", "VISCODE"])

        if df.AGE.isna().any(): 
            
            # Fill NaN values in 'AGE' with the difference between 'Scan date' and 'PTDOB' divided by one year
            merged_df['AGE'].fillna(merged_df.fillna((pd.to_datetime(merged_df['Scan Date']) - merged_df['PTDOB']).dt.days // 365), inplace=True)

        elif df.PTGENDER.isna().any(): 

            # Fill NaN values in 'PTGENDER' with values from 'PTGENDER_y' (which is 'PTGENDER' from df_demog)
            merged_df['PTGENDER_x'].fillna(merged_df['PTGENDER_y'], inplace=True)
            
            if (merged_df.PTGENDER_x == merged_df.PTGENDER_y).all():
                merged_df = merged_df.drop(columns="PTGENDER_y").rename(columns={"PTGENDER_x": "PTGENDER"})

        return merged_df
        
    else : 
            return df

def correct_age_viscode_increment(df): 

    df_test = df.copy()
    # Group the dataframe by RID
    grouped = df_test.groupby('RID')

    # Iterate over RID groups
    for rid, group in grouped:
        # Group the current RID group by AGE
        age_groups = group.groupby('AGE')
        
        # Iterate over AGE groups
        for age, age_group in age_groups:
            # Check if there are multiple occurrences of the same age
            if len(age_group) > 1:
                # Iterate over rows in the AGE group
                for index, row in age_group.iterrows():
                    # Check if the current row is not the first row in the AGE group
                    if index != age_group.index[0]:
                        # Increment the age based on the Months column
                        df_test.loc[index, 'AGE'] += row['Months'] / 12

    return df_test

def merge_inputs_to_targets(input_df_list=[], target_df=None, df_demog=None, targetnames=[], inputnames=[]):

    output_list = []

    for i, df_input in enumerate(input_df_list) : 

        df_input_merged = df_input.merge(target_df, on=["RID", "VISCODE"], how="left", suffixes=('', '_drop'))
        df_input_merged = df_input_merged.drop(df_input_merged.filter(regex='_drop$').columns, axis=1)

        df_input_merged = fill_age_gender(df_input_merged, df_demog)
        
        if len(inputnames)==len(input_df_list):
            df_input_merged = df_input_merged.dropna(subset=targetnames+inputnames+["AGE", "PTGENDER"]).drop_duplicates(subset=["SubjectID", "VISCODE"])

        output_list.append(df_input_merged)

    return  tuple(output_list)

# ------------------------------ Preprocessing ------------------------------ #

# Custom Transformer for Demographic Correction

class DemographicAdjustmentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None):
        """
        Initialize the transformer with optional categorical column names.

        Args:
            categorical_columns (list, optional): List of categorical column names that should not be transformed.
        """
        self.model_ = None
        self.coef_ = None
        self.intercept_ = None
        self.categorical_columns = categorical_columns  # Allow setting categorical columns during initialization
        self.non_categorical_columns_ = None  # This will be set in fit if not already provided

    def fit(self, X, demographic_data, categorical_columns=None):
        """
        Fit a linear regression model on demographic variables for each non-categorical feature in X.
        
        Args:
            X (pd.DataFrame): DataFrame of features (including both categorical and continuous variables).
            demographic_data (pd.DataFrame): DataFrame containing demographic variables to be regressed out.
            categorical_columns (list, optional): List of categorical column names to exclude from transformation.
        """
        # Allow setting categorical columns at the time of fitting
        if categorical_columns is not None:
            if categorical_columns != self.categorical_columns : 
                raise ValueError("Inconsistent categorical columns!")
        
        if self.categorical_columns is None:
            # Automatically detect categorical columns if none are provided
            self.categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Set non-categorical columns for regression
        self.non_categorical_columns_ = X.columns.difference(self.categorical_columns)
        
        # Fit the model only on non-categorical columns
        X_non_categorical = X[self.non_categorical_columns_]
        self.model_ = LinearRegression()
        self.model_.fit(demographic_data, X_non_categorical)

        # Store coefficients and intercept for transformation
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_

        return self

    def transform(self, X, demographic_data):
        """
        Remove effects of demographic variables from each non-categorical feature in X by using residuals.
        
        Args:
            X (pd.DataFrame): DataFrame of features (including both categorical and continuous variables).
            demographic_data (pd.DataFrame): DataFrame containing demographic variables to be regressed out.
        
        Returns:
            pd.DataFrame: Transformed DataFrame with residuals for non-categorical columns and original values for categorical columns.
        """
        if self.model_ is None : 
            raise ValueError("This CustomModel instance is not fitted yet. "
                                      "Call 'fit' with appropriate arguments before using this method.")

        # Predict demographic influence on non-categorical columns
        X_non_categorical = X[self.non_categorical_columns_]
        X_non_categorical = X_non_categorical.astype("float32")
        preds = self.model_.predict(demographic_data)
        
        # Calculate residuals for non-categorical columns
        residuals = X_non_categorical - preds

        # Combine residuals with categorical columns and ensure original column order
        X_transformed = X.copy()
        X_transformed[self.non_categorical_columns_] = residuals

        # Check that the categorical columns were not inadvertently altered
        if X_transformed[self.non_categorical_columns_].compare(X[self.non_categorical_columns_]).empty:
            raise ValueError("Ordinal values wrongly modified!")
        
        return X_transformed

    def fit_transform(self, X, demographic_data, categorical_columns=None):
        """Fit and transform in a single step for convenience."""
        self.fit(X, demographic_data, categorical_columns)
        return self.transform(X, demographic_data)
    
    def inverse_transform(self, residuals, demographic_data):
        """
        Apply the reverse transformation to add back the demographic effects to the non-categorical columns.
        
        Args:
            residuals (pd.DataFrame): DataFrame with residuals for non-categorical columns.
            demographic_data (pd.DataFrame): DataFrame containing demographic variables to re-apply the demographic influence.
        
        Returns:
            pd.DataFrame: DataFrame with demographic effects added back to non-categorical columns and original values for categorical columns.
        """
        if self.model_ is None:
            raise ValueError("The model has not been fitted yet. Please call `fit` before using `inverse_transform`.")
        
        # if not isinstance(residuals, pd.DataFrame):
        #     residuals = pd.DataFrame(residuals, columns=self.non_categorical_columns_)
        
        X_original = residuals.copy()
        
        # Predict demographic effects for non-categorical columns
        preds = self.model_.predict(demographic_data)

        # Add the demographic effect back to the residuals for non-categorical columns
        X_original[self.non_categorical_columns_] = residuals[self.non_categorical_columns_] + preds

        if X_original[self.non_categorical_columns_].compare(residuals[self.non_categorical_columns_]).empty:
            raise ValueError("Ordinal values wrongly modified!")
    
        return X_original


class Preprocessing:
    """Superclass with static methods to preprocess features derived from MRI data."""
    def __init__(self, data = None):
        if data is not None : 
            self.standardize_mean = data.mean()
            self.standardize_std = data.std()
        else: 
            self.standardize_mean = None
            self.standardize_std = None

        self.correct_coef_ = None
        self.correct_intercept_ = None

    def correct_demographics(self, data, c):
        """Correct the MRI features for covariates by fitting a linear regression to the MRI data and keeping residuals.

        Args:
            data (numpy.array): MRi data to correct.
            c (numpy.array): Covariates to correct for.
            
        Returns:
            _type_: _description_
        """
        res = np.zeros_like(data)
        pred = np.zeros_like(data)

        if self.correct_coef_ is None or self.correct_intercept_ is None:

            print("Correct demographics : fit linear regression parameters to input data.")

            regr = LinearRegression()
            regr.fit(c, data)

            self.correct_coef_ = regr.coef_
            self.correct_intercept_ = regr.intercept_
        else : 
            print("Correct demographics : linear regression already fitted, transform only.")

        regr = LinearRegression(fit_intercept=True)
        regr.coef_ = self.correct_coef_
        regr.intercept_ = self.correct_intercept_

        pred = regr.p
    def standardize(self, data):
        """Center and divide data by feature-wise standard deviation.

        Args:
            data (torch.tensor): Data to standardize.

        Returns:
            torch.tensor: Standardized data
        """
        # PyTorch implementation of standard scaler
        if self.standardize_mean is None or self.standardize_std is None:

            print("Standardizer not fitted yet : new fit to data. ") 

            self.standardize_mean = data.mean(0)
            self.standardize_std = data.std(0)

        data -= self.standardize_mean
        data /= self.standardize_std

        return data

    @staticmethod
    def remove_outliers(X, y, X_threshold=None, y_threshold=None):
        """Remove outliers from input and target data.

        Args:
            X (np.array): Input data.
            y (np.array): Target data.
            X_threshold (tuple, optional): Minimum and maximum threhold boundaries to apply to input data. Defaults to None.
            y_threshold (tuple, optional): Minimum and maximum threhold boundaries to apply to target data. Defaults to None.

        Returns:
            np.array: Input and target wihtout the outliers and a boolean array of size of the original number of samples with samples that were flagged as outliers.
        """
        X_is_out = np.full((X.shape[0],), False)
        y_is_out = np.full((y.shape[0],), False)

        # thesholds = (min, max)
        if X_threshold is not None:
            X_is_out = np.logical_or(
                (X < X_threshold[0]).any(axis=1), (X > X_threshold[1]).any(axis=1)
            )

        if y_threshold is not None:
            y_is_out = np.logical_or(
                (y < y_threshold[0]).any(axis=1), (y > y_threshold[1]).any(axis=1)
            )

        is_outlier = np.logical_or(X_is_out, y_is_out)

        X = X[~is_outlier]
        y = y[~is_outlier]

        return X, y, is_outlier
    


    def standardize(self, data):
        """Center and divide data by feature-wise standard deviation.

        Args:
            data (torch.tensor): Data to standardize.

        Returns:
            torch.tensor: Standardized data
        """
        # PyTorch implementation of standard scaler
        if self.standardize_mean is None or self.standardize_std is None:

            print("Standardizer not fitted yet : new fit to data. ") 

            self.standardize_mean = data.mean(0)
            self.standardize_std = data.std(0)

        data -= self.standardize_mean
        data /= self.standardize_std

        return data

    @staticmethod
    def remove_outliers(X, y, X_threshold=None, y_threshold=None):
        """Remove outliers from input and target data.

        Args:
            X (np.array): Input data.
            y (np.array): Target data.
            X_threshold (tuple, optional): Minimum and maximum threhold boundaries to apply to input data. Defaults to None.
            y_threshold (tuple, optional): Minimum and maximum threhold boundaries to apply to target data. Defaults to None.

        Returns:
            np.array: Input and target wihtout the outliers and a boolean array of size of the original number of samples with samples that were flagged as outliers.
        """
        X_is_out = np.full((X.shape[0],), False)
        y_is_out = np.full((y.shape[0],), False)

        # thesholds = (min, max)
        if X_threshold is not None:
            X_is_out = np.logical_or(
                (X < X_threshold[0]).any(axis=1), (X > X_threshold[1]).any(axis=1)
            )

        if y_threshold is not None:
            y_is_out = np.logical_or(
                (y < y_threshold[0]).any(axis=1), (y > y_threshold[1]).any(axis=1)
            )

        is_outlier = np.logical_or(X_is_out, y_is_out)

        X = X[~is_outlier]
        y = y[~is_outlier]

        return X, y, is_outlier