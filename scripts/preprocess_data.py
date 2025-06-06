import sys
import re 
import ast

sys.path.insert(0, '..')
sys.path.insert(0, './src')

from src.train import *
from src.functions import *
from src.plots import *
from src.dataset import *
from src.multixgboost import *

import warnings

import missingno as msno

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

df_cog_merge, df_demog, agg_targets, sep_targets = load_cognition()
df_all, df_MRI, df_PET = load_brain_imaging()
df_CSF = load_CSF()
df_counts, df_dge = load_transcriptomics()
df_genotype = load_apoe()

df_inventory = pd.read_csv("../../../dataset/ADNI/neuropsychological/inventory.csv").replace({"sc":"bl","Female":0, "Male":1})

df_counts = pd.read_csv("../../../dataset/ADNI/gene_expression_microarray/filtered_counts.csv", index_col=0)
df_samples = pd.read_csv("../../../dataset/ADNI/gene_expression_microarray/filtered_samples.csv", index_col=0)
df_genes = pd.read_csv("../../../dataset/ADNI/gene_expression_microarray/filtered_genes.csv", index_col=0)
df_dge = pd.read_csv("../../../dataset/ADNI/gene_expression_microarray/dge_gene_selection.csv", index_col=0)

df_counts = df_counts.transpose()
df_counts = df_counts.loc[:, df_counts.columns.isin(df_dge.Symbol)]
df_counts = df_counts.loc[:, ~df_counts.columns.str.contains("ENSG")]

df_counts = df_counts.merge(df_samples[["SubjectID", "Visit"]], left_index=True, right_index=True)
df_counts = df_counts.reset_index(drop=True).drop_duplicates()
df_counts = df_counts.rename(columns={"Visit":"VISCODE"})

df_dge = df_dge[~df_dge.Symbol.str.contains("ENSG")]

print(f'Number of patients in imaging dataset dataset : {df_all[["SubjectID", "RID", "DX"]].drop_duplicates().dropna().shape[0]}')
print(f'Number of patients in imaging dataset dataset : {len(df_counts.SubjectID.unique())}')

df_MRI.rename(columns={"Research Group":"DX"}, inplace=True)
df_PET.rename(columns={"Research Group":"DX"}, inplace=True)

print(f"Cogntitive : {df_cog_merge.shape} , Demographics : {df_demog.shape}")
print(f"Cortical thickness : {df_all.shape} , Gray matter volume : {df_MRI.shape} , PET activity : {df_PET.shape}")
print(f"Cerebrospinal fluid : {df_CSF.shape}")
print(f"Transcriptomics counts : {df_counts.shape}, Differential gene expression : {df_dge.shape}")

df_1 = df_cog_merge.merge(df_demog[["RID", "VISCODE"]], how="outer", left_on=["RID", "VISCODE"], right_on=["RID", "VISCODE"], suffixes=('_cog', '_dem'))
df_2 = df_1.merge(df_all, how="outer",left_on=["RID", "VISCODE"], right_on=["RID", "VISCODE"], suffixes=('', '_mri')).merge(df_MRI, how="outer",left_on=["RID", "VISCODE"], right_on=["RID", "VISCODE"], suffixes=('', '_gm')).merge(df_PET, how="outer",left_on=["RID", "VISCODE"], right_on=["RID", "VISCODE"], suffixes=('', '_pet'))
df_3 = df_2.merge(df_CSF, how="outer",left_on=["RID", "VISCODE"], right_on=["RID", "VISCODE"], suffixes=('', '_csf'))#.merge(df_counts, how="outer",left_on=["RID", "VISCODE"], right_on=["RID"], suffixes=('', '_dge'))

df_3 = df_3.rename(columns={"APOE4":"APOE_epsilon4"}).merge(df_genotype, how="outer",left_on=["RID"], right_on=["RID"], suffixes=('', '_gene')) 
df_counts = df_counts.merge(df_inventory.rename(columns={"PTID":"SubjectID"})[["SubjectID","RID"]], how="left").drop_duplicates()
df_3 = df_3.merge(df_counts, how="outer",left_on=["RID"], right_on=["RID"], suffixes=('', '_dge'))

df_4 = df_3.drop_duplicates().set_index(["RID", "VISCODE"])

all_cols_to_merge = df_4.filter(regex='_mri|_gm|_pet|_dge|_csf|_gene').columns.tolist()
all_cols_to_merge.sort()
all_cols_to_merge = [col for col in all_cols_to_merge if not re.search(r'.*VISCODE.*', col)]

# Have a look at the missing data in the merged dataset

check_cols = sorted(list(set(list(map(lambda r: '_'.join(r.split('_')[:-1]), all_cols_to_merge)))) + all_cols_to_merge)
msno.matrix(df_4[check_cols])

df_4_ = df_4.copy() # just in case

for col in all_cols_to_merge: 

    update_colname = re.sub('_mri|_gm|_pet|_dge|_csf|_gene', '', col) 

    print(f"{col} \t ----> \t {update_colname}")

    df_4.loc[:, update_colname].update(df_4_.loc[:, col])
    df_4.drop(columns=col, inplace=True)

def f(s): 
    if type(s) == str: 
        return s.replace("<", "").replace(">", "")
    else : 
        return s

for col in all_cols_to_merge: 
    
    update_colname = re.sub('_mri|_gm|_pet|_dge|_csf|_gene', '', col) 

    df_4_nona = df_4_[df_4_[col].notna() & df_4_[update_colname].notna()][[col, update_colname]]

    df_4_nona[col] = df_4_nona[col].apply(f)
    df_4_nona[col] = df_4_nona[col].apply(f)
    df_4_nona[update_colname] = df_4_nona[update_colname].apply(f)
    df_4_nona[update_colname] = df_4_nona[update_colname].apply(f)

    df_4_nona[col] = df_4_nona[col].astype(float, errors="ignore").astype(str, errors="ignore")
    df_4_nona[update_colname] = df_4_nona[update_colname].astype(float, errors="ignore").astype(str, errors="ignore")

    if (df_4_nona[col] != df_4_nona[update_colname]).any(): 
        print(f"Non equal values in columns : {col} | {update_colname}")
        print(df_4_nona[df_4_nona[col] != df_4_nona[update_colname]])

# Check that the merged columns that have common elements are indeed the same otherwise print them and maybe raise an error.


important_features = agg_targets+sep_targets+["Schaefer_200_7", "MRI_Vol", "PET_Vol", "ABETA", "TAU", "PTAU"]


df_5 = df_4.dropna(subset=df_dge.Symbol.tolist()+["Schaefer_200_7", "MRI_Vol", "PET_Vol", "ABETA", "TAU", "PTAU"], how="all")

df_5["DGE"] = df_5[df_dge.Symbol.tolist()].values.tolist()

def replace_nan_list(lst):
    return np.nan if all(pd.isna(x) for x in lst) else lst

# Apply the custom function to the 'DGE' column
df_5['DGE'] = df_5['DGE'].apply(replace_nan_list)

duplicated_columns = df_5.columns[df_5.apply(lambda x: any(x.duplicated()))]

# Print columns with duplicated values
print("Columns with duplicated values:", duplicated_columns)

df_5 = df_5.loc[~df_5.astype(str).duplicated()]

# ### Check for missing values

df_missing_vis = df_5.copy()
df_missing_vis["CSF"] = df_missing_vis[["ABETA", "TAU", "PTAU"]].values.tolist()
df_missing_vis["RNA"] = df_missing_vis[df_dge.Symbol.tolist()].values.tolist()
df_missing_vis["CSF"] = df_missing_vis["CSF"].apply(replace_nan_list)
df_missing_vis["RNA"] = df_missing_vis["RNA"].apply(replace_nan_list)

df_missing_vis = df_missing_vis.drop(columns=df_dge.Symbol.tolist()+["ABETA", "TAU", "PTAU"])

msno.matrix(df_missing_vis[agg_targets+sep_targets+["Schaefer_200_7", "MRI_Vol", "PET_Vol", "CSF", "RNA"]])

print(df_missing_vis.PET_Vol.notna().sum())

features_dge = df_dge.Symbol.tolist()

for string in features_dge:
    if ("|" in string):
        new_strings = string.split(' || ')
        for n in new_strings: 
            print(n)
    else: 
        print(string)

features_csf = ["ABETA", "TAU", "PTAU"]

# ## Create final dataframe

important_features = set(important_features+df_dge.Symbol.tolist())

df_FinalCombination =  df_5.copy()
print(f"Shape : {df_FinalCombination.shape}")
print(f"Remove rows where any target is NaN")
df_FinalCombination = df_FinalCombination.dropna(subset=["ADNI_MEM", "ADNI_EF", "ADNI_VS", "ADNI_LAN"], how="any").reset_index(drop=False)
print(f"Shape : {df_FinalCombination.shape}")
print(f"Remove rows where all imaging columns are NaN")
df_FinalCombination = df_FinalCombination.dropna(subset=["Schaefer_200_7", "MRI_Vol", "PET_Vol"], how="all")
print(f"Shape : {df_FinalCombination.shape}")

df_FinalCombination[["Schaefer_200_7", "MRI_Vol", "PET_Vol"]].isna().sum()/len(df_FinalCombination)

df_5_test = df_5.dropna(subset=["ADNI_MEM", "ADNI_EF", "ADNI_VS", "ADNI_LAN"], how="any")

df_FinalCombination.PTAU.notna().sum()

mask_na_apoe = df_FinalCombination[["RID", "APOE_epsilon2", "APOE_epsilon3", "APOE_epsilon4"]].isna().any(axis=1)

# Update APOE_epsilon2 and APOE_epsilon3 to 0 if APOE_epsilon4 == 2 AND APOE_epsilon2 or APOE_epsilon3 are NaN
df_FinalCombination.loc[(df_FinalCombination['APOE_epsilon4'] == 2) & (df_FinalCombination['APOE_epsilon2'].isna()), 'APOE_epsilon2'] = 0
df_FinalCombination.loc[(df_FinalCombination['APOE_epsilon4'] == 2) & (df_FinalCombination['APOE_epsilon3'].isna()), 'APOE_epsilon3'] = 0

# Group by 'RID' and check for inconsistencies across APOE_epsilon columns
for rid, group in df_FinalCombination.groupby('RID'):
    epsilon2_values = group['APOE_epsilon2'].dropna().unique()
    epsilon3_values = group['APOE_epsilon3'].dropna().unique()
    epsilon4_values = group['APOE_epsilon4'].dropna().unique()

    # Check for inconsistencies: if there are more than one unique non-NaN value
    if len(epsilon2_values) > 1 or len(epsilon3_values) > 1 or len(epsilon4_values) > 1:
        raise ValueError(f"Inconsistent APOE values for RID {rid}: "
                         f"APOE_epsilon2 = {epsilon2_values}, "
                         f"APOE_epsilon3 = {epsilon3_values}, "
                         f"APOE_epsilon4 = {epsilon4_values}")

print(list(set(df_FinalCombination.RID[mask_na_apoe]) & set(df_FinalCombination.RID[~mask_na_apoe])))

msno.matrix(df_FinalCombination.loc[:,~df_FinalCombination.columns.duplicated()].copy())

print(df_FinalCombination["AGE"].isna().sum())
df_FinalCombination = df_FinalCombination.dropna(subset=["AGE"])
print(df_FinalCombination["AGE"].isna().sum())
df_FinalCombination = df_FinalCombination.loc[:,~df_FinalCombination.columns.duplicated()].copy()


msno.matrix(df_FinalCombination)


df_FinalCombination[["RID", "VISCODE"]]


select_features = df_dge.Symbol.tolist()+["ADNI_MEM", "ADNI_VS", "ADNI_LAN", "ADNI_EF"] +["ABETA", "TAU", "PTAU", "APOE_epsilon2", "APOE_epsilon3", "APOE_epsilon4"]+["Schaefer_200_7", "MRI_Vol", "PET_Vol"]


df_FinalCombination[["RID", "VISCODE"]].duplicated().sum()


df_adnimerge = pd.read_csv("../../../dataset/ADNI/adnimerge/ADNIMERGE_02Nov2023.csv").replace({"sc":"bl","Female":0, "Male":1})
df_adnimerge = df_adnimerge[["RID", "VISCODE", "DX"]].replace({"Dementia":"AD"})

df_DX = pd.concat([df_all[["RID", "VISCODE", "DX"]], df_MRI[["RID", "VISCODE", "DX"]], df_PET[["RID", "VISCODE", "DX"]], df_adnimerge[["RID", "VISCODE", "DX"]]]).replace({"EMCI":"MCI", "LMCI":"MCI", "SMC":"CN"})
df_DX["FROM"] = ["ALL"]*df_all.shape[0] + ["MRI"]*df_MRI.shape[0] + ["PET"]*df_PET.shape[0] + ["MERGE"]*df_adnimerge.shape[0]

df_DX = df_DX.drop_duplicates(subset=["RID", "VISCODE", "DX"], keep="last").dropna().reset_index(drop=True).sort_values(["RID", "VISCODE"])
df_DX = df_DX.drop_duplicates(subset=["RID", "VISCODE"], keep="last")


df_FinalCombination.DX = df_FinalCombination.DX.replace({"EMCI": "MCI", "LMCI": "MCI", "SMC": "CN"})
df_FinalCombination = df_FinalCombination.dropna(subset="DX")

def has_list(x):
    return any(isinstance(i, list) for i in x)

mask = df_FinalCombination.apply(has_list)
df_FinalCombination = df_FinalCombination.drop_duplicates(subset=["RID", "VISCODE", "ORIGPROT", "COLPROT", "USERDATE", "EXAMDATE", "ADNI_MEM", "ADNI_LAN", "ADNI_VS", "ADNI_EF", "DX"])


# Define a function to convert the values to months
def convert_to_months(value):
    if value.startswith('m'):
        return int(value[1:])
    else:
        return 0

# Apply the function to the 'period' column
df_FinalCombination["Months"] = df_FinalCombination["VISCODE"].apply(lambda x: convert_to_months(x))


df_FinalCombination = fill_age_gender(df_FinalCombination, df_demog)
df_FinalCombination = correct_age_viscode_increment(df_FinalCombination)


df_FinalCombination.PTGENDER.value_counts()

df_FinalCombination.to_csv("../../../dataset/ADNI/combined_dataset.csv", index=False)

df_FinalCombination[["PTAU", "TAU", "ABETA"]] = df_FinalCombination[["PTAU", "TAU", "ABETA"]].apply(lambda x : x.str.replace(">|<","", regex=True)).apply(pd.to_numeric)

df_predict = df_FinalCombination.loc[:, ["RID", "VISCODE"]+select_features]


def dataframe_to_matrices(data, unpack_input_names, unpack_suffix, unpack_feature_names, keep_input_names, target_names):

    df = data.copy()
    
    df_y = df[target_names]

    df_list = [df[keep_input_names]]

    for (in_name, in_suff) in zip(unpack_input_names, unpack_suffix): 


        print(in_name)

        if in_suff == "_Cortical" : 
            X_temp = np.array([ast.literal_eval(x) if isinstance(x, str) else np.full((202, ), np.nan)for x in df[in_name].to_list()])
            X_temp = X_temp[:, 2:]
        else :   
            X_temp = np.array([ast.literal_eval(x)  if isinstance(x, str) else np.full((200, ), np.nan)  for x in df[in_name].to_list()])

        print(f"Data shape : {df.shape}")
        print(f"Number of non-NaN values : {df[in_name].notna().sum()}")

        new_names = [feat+in_suff for feat in unpack_feature_names]
        df_X_temp = pd.DataFrame(X_temp, columns=new_names)

        df_list.append(df_X_temp.set_index(df_list[0].index))

    df_X = pd.concat(df_list, axis=1)

    df_X = df_X.loc[:,~df_X.columns.duplicated()].copy()

    return df_X, df_y, df_list


df_X, df_y, df_list =  dataframe_to_matrices(df_predict, unpack_input_names=["Schaefer_200_7", "MRI_Vol", "PET_Vol"], unpack_suffix=["_Cortical", "_GM", "_PET"], unpack_feature_names=read_featurenames(), keep_input_names=df_dge.Symbol.tolist()+["ABETA", "TAU", "PTAU", "APOE_epsilon2", "APOE_epsilon3", "APOE_epsilon4"], target_names=["ADNI_MEM", "ADNI_EF", "ADNI_LAN", "ADNI_VS"])

select_MRIthickness = df_X.columns[df_X.columns.str.endswith("Cortical")]
select_MRIvolume = df_X.columns[df_X.columns.str.endswith("GM")]
select_PET = df_X.columns[df_X.columns.str.endswith("PET")]
select_RNA = df_dge.Symbol.tolist()

sep_targets = ["ADNI_MEM", "ADNI_EF", "ADNI_VS", "ADNI_LAN"]


df_dge.Symbol.str.contains("ENSG").sum()


del df_dge

select_CSF = ["ABETA", "TAU", "PTAU"]
select_gene = ["APOE_epsilon2", "APOE_epsilon3", "APOE_epsilon4"]
select_features = [select_RNA, select_CSF, select_gene, select_MRIthickness, select_MRIvolume, select_PET]

dict_select = {'ADNI_cog': ['ADNI_MEM', 'ADNI_EF', 'ADNI_VS', 'ADNI_LAN'], 'RNA': select_RNA, 'CSF': select_CSF, 'DNA': select_gene, 'MRIth': select_MRIthickness, 'MRIvol': select_MRIvolume, 'PET': select_PET}

df_y[df_y.isna().any(axis=1)] = df_y.ffill()[df_y.isna().any(axis=1)]
print((df_FinalCombination.index == df_y.index).all())
print((df_FinalCombination.index == df_X.index).all())


df_X[select_RNA].isna().all(axis=1).sum()/df_X.shape[0]


df_X[select_gene].isna().all(axis=1).sum()/df_X.shape[0]


df_X[select_CSF].dropna()


df_X


df_FinalCombination[["RID", "VISCODE", "DX", "AGE", "PTGENDER", "PTEDUCAT"]].isna().sum()

# # Save data as pickle


df_X.iloc[:, -200:]


import pickle 

with open('../pickle/df_X_original.pickle', 'wb') as handle:
    pickle.dump(df_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../pickle/df_y_original.pickle', 'wb') as handle:
    pickle.dump(df_y[["ADNI_MEM", "ADNI_EF", "ADNI_VS", "ADNI_LAN"]], handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../pickle/df_all_original.pickle', 'wb') as handle:
    pickle.dump(df_FinalCombination, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../pickle/select_features.pickle', 'wb') as handle:
    pickle.dump(dict_select, handle, protocol=pickle.HIGHEST_PROTOCOL)


df_X = pd.read_pickle('../pickle/df_X_original.pickle')
df_y =  pd.read_pickle('../pickle/df_y_original.pickle')
df_all = pd.read_pickle('../pickle/df_all_original.pickle')
df_FinalCombination = pd.read_pickle('../pickle/df_all_original.pickle')


sorted(df_all.columns.tolist())

# # Save scaled values 


df_X_std = df_X.copy()
categorical_cols = ~df_X_std.columns.str.startswith("APOE")


df_X_std = df_X.copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_X_std)
df_X_std.loc[:, categorical_cols] = pd.DataFrame(scaled_features[:, categorical_cols], index=df_X.index, columns=df_X.columns[categorical_cols])


filter_miss_mask = df_all.isna()


with open('../pickle/df_all_original.pickle', 'rb') as handle:
    df_all = pickle.load(handle)

with open('../pickle/df_X.pickle', 'wb') as handle:
    pickle.dump(df_X_std, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../pickle/df_y.pickle', 'wb') as handle:
    pickle.dump(df_y[["ADNI_MEM", "ADNI_EF", "ADNI_VS", "ADNI_LAN"]], handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../pickle/df_all.pickle', 'wb') as handle:
    pickle.dump(df_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../pickle/filter_miss_mask.pickle', 'wb') as handle:
    pickle.dump(filter_miss_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Verify new data


df_all_update = pd.read_csv("../../../dataset/ADNI/updated_adni_features/ALL_3.csv").replace({"sc":"bl","Female":0, "Male":1})


df_ad_cn = pd.read_csv("../../../dataset/ADNI/updated_adni_features/data/AD_CN/CSF.csv", names=["ABETA", "PTAU", "TAU"]).replace({"sc":"bl","Female":0, "Male":1})
df_ad_mci = pd.read_csv("../../../dataset/ADNI/updated_adni_features/data/AD_MCI/CSF.csv", names=["ABETA", "PTAU", "TAU"]).replace({"sc":"bl","Female":0, "Male":1})
df_cn_mci = pd.read_csv("../../../dataset/ADNI/updated_adni_features/data/CN_MCI/CSF.csv", names=["ABETA", "PTAU", "TAU"]).replace({"sc":"bl","Female":0, "Male":1})


df_ad_cn_label = pd.read_csv("../../../dataset/ADNI/updated_adni_features/data/AD_CN/AD_CN_label.csv").replace({"sc":"bl","Female":0, "Male":1})


df_all[df_all["SubjectID"] == "002_S_0295"][["RID", "SubjectID", "PTGENDER","AGE", "VISCODE", "VISCODE"]]


df_all_update[df_all_update["SubjectID"] == "002_S_0295"][["RID", "SubjectID", "PTGENDER","AGE", "VISCODE", "VISCODE"]]


df_all_update[df_all_update["SubjectID"] == "002_S_0295"][["RID", "SubjectID", "PTGENDER","AGE", "VISCODE", "VISCODE"]].duplicated()


df_all_update[df_all_update["SubjectID"] == "002_S_0295"].duplicated()


df_all_update.PTAU.isna().sum() == df_all.PTAU.isna().sum()


