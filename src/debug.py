import pandas as pd

def check_upenn_debug_overlap():
    """
    Run overlap diagnostics between UPENN master CSF dataset and historical UPENN CSF batches.
    """
    common_columns = ['RID', 'VISCODE']

    # Load master and historical batch datasets
    df_upennbiomk_master = pd.read_csv("../../../dataset/ADNI/adnimerge/upennbiomk_master.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_upennbiomk12_2020 = pd.read_csv("../../../dataset/ADNI/adnimerge/upennbiomk12_2020.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_upennbiomk = pd.read_csv("../../../dataset/ADNI/adnimerge/upennbiomk.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_upennbiomk3 = pd.read_csv("../../../dataset/ADNI/adnimerge/upennbiomk3.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_upennbiomk6 = pd.read_csv("../../../dataset/ADNI/adnimerge/upennbiomk6.csv").replace({"sc": "bl", "Female": 0, "Male": 1})
    df_upennbiomk8 = pd.read_csv("../../../dataset/ADNI/adnimerge/upennbiomk8.csv").replace({"sc": "bl", "Female": 0, "Male": 1})

    df_list = [
        ("UPENNBIOMK", df_upennbiomk),
        ("UPENNBIOMK3", df_upennbiomk3),
        ("UPENNBIOMK6", df_upennbiomk6),
        ("UPENNBIOMK8", df_upennbiomk8),
        ("UPENNBIOMK12_2020", df_upennbiomk12_2020),
    ]

    rows_master = set(tuple(row) for row in df_upennbiomk_master[common_columns].dropna().to_numpy())

    for name, df in df_list:
        rows_current = set(tuple(row) for row in df[common_columns].dropna().to_numpy())
        overlapping_rows = rows_master & rows_current
        overlap_count = len(overlapping_rows)

        print(f"Dataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Overlapping with master ({common_columns}): {overlap_count}")
        print(f"Unique rows not in master: {len(rows_current - rows_master)}")


def check_row_overlap(df1, df2, common_columns, df1_label="df1", df2_label="df2", rename_columns=None):
    """
    Compare rows between two dataframes using a list of columns.

    Parameters:
    - df1, df2: pandas DataFrames to compare.
    - common_columns: list of column names to base the comparison on.
    - df1_label, df2_label: optional names to label outputs.
    - rename_columns: optional dict to rename columns in df1 (e.g., {"Visit": "VISCODE"}).

    Prints overlap statistics and sample differing rows.
    """
    if rename_columns:
        df1 = df1.rename(columns=rename_columns)
    
    # Convert selected columns to sets of tuples, drop rows with NaNs
    rows_df1 = set(tuple(row) for row in df1[common_columns].dropna().to_numpy())
    rows_df2 = set(tuple(row) for row in df2[common_columns].dropna().to_numpy())

    overlapping_rows = rows_df1 & rows_df2
    only_in_df1 = rows_df1 - rows_df2
    only_in_df2 = rows_df2 - rows_df1

    print(f"\n--- Row Overlap Report ---")
    print(f"Using columns: {common_columns}")
    print(f"Number of overlapping rows: {len(overlapping_rows)}")
    print(f"Shape of {df1_label}: {df1.shape}")
    print(f"Shape of {df2_label}: {df2.shape}")
    print(f"Rows only in {df1_label}: {len(only_in_df1)}")
    print(f"Rows only in {df2_label}: {len(only_in_df2)}")

    # Show a few example differences
    print(f"\nSample rows only in {df1_label}:")
    for row in list(only_in_df1)[:5]:
        print(row)

    print(f"\nSample rows only in {df2_label}:")
    for row in list(only_in_df2)[:5]:
        print(row)