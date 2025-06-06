import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer model on OpenML dataset")

    parser.add_argument("--dset_id", type=int, required=True,
                        help="OpenML dataset ID to download and train on")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "regression"],
                        help="Task type: 'classification' or 'regression'")
    parser.add_argument("--attentiontype", type=str, required=True,
                        help="Type of attention mechanism to use (e.g., 'softmax', 'linear', 'nystrom')")
    
    # Optional arguments (add more as needed)
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n[INFO] Starting training with the following configuration:")
    print(f" - Dataset ID: {args.dset_id}")
    print(f" - Task: {args.task}")
    print(f" - Attention Type: {args.attentiontype}")
    print(f" - CV Folds: {args.cv_folds}")
    print(f" - Seed: {args.random_seed}\n")

    # 1. Load dataset from OpenML
    # dataset = openml.datasets.get_dataset(args.dset_id)
    # df, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')

    # 2. Preprocessing (placeholder)
    # e.g., encoding categorical variables, normalization, etc.

    # 3. Set up cross-validation (placeholder)
    # from sklearn.model_selection import StratifiedKFold or KFold

    # 4. Train model with specified attention type (placeholder)
    # model = YourTransformerModel(attention_type=args.attentiontype, ...)
    # for fold in range(args.cv_folds):
    #     Train, validate, evaluate...

    # 5. Report results (placeholder)
    # print("Final CV performance: ...")

    print("[INFO] Training pipeline completed.\n")


if __name__ == "__main__":
    main()
