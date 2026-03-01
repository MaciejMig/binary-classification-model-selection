import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

RANDOM_STATE = 67


def load_data(data_dir: Path):
    X_train = pd.read_csv(data_dir / "artifical_train_data.csv")
    y_train = pd.read_csv(data_dir / "artifical_train_labels.csv")["Class"]
    X_test = pd.read_csv(data_dir / "artifical_test_data.csv")
    return X_train, y_train, X_test


def build_model():
    # Best configuration from the report (GridSearchCV, scoring=balanced_accuracy)
    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    parser = argparse.ArgumentParser(description="Train model and generate submission file.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to folder with CSV files.")
    parser.add_argument("--out", type=str, required=True, help="Output path for prediction .txt file.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test = load_data(data_dir)
    model = build_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    idx1 = int(np.where(model.classes_ == 1)[0][0])  # robust: find class '1'
    p1 = proba[:, idx1]

    with out_path.open("w", encoding="utf-8") as f:
        for v in p1:
            f.write(f"{float(v):.10f}\n")

    print(f"Saved: {out_path} (n={len(p1)})")


if __name__ == "__main__":
    main()
