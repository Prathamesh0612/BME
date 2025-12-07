# train_models.py

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].values.astype("float32")

    y_ratio = df["target_ratio"].values.astype("float32")
    y_parts = df["target_parts"].values.astype("float32")
    y_pattern = df["target_pattern"].values.astype("int32")
    y_bases = df["target_base_set"].values.astype("int32")

    return X, y_ratio, y_parts, y_pattern, y_bases


def train_all(csv_path: str, out_dir: str = "models"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    X, y_ratio, y_parts, y_pattern, y_bases = load_dataset(csv_path)

    X_train, X_test, y_ratio_tr, y_ratio_te = train_test_split(
        X, y_ratio, test_size=0.2, random_state=42
    )
    _, _, y_parts_tr, y_parts_te = train_test_split(
        X, y_parts, test_size=0.2, random_state=42
    )
    _, _, y_pat_tr, y_pat_te = train_test_split(
        X, y_pattern, test_size=0.2, random_state=42
    )
    _, _, y_base_tr, y_base_te = train_test_split(
        X, y_bases, test_size=0.2, random_state=42
    )

    # ---- Corruption ratio model ----
    print("Training model_ratio...")
    m_ratio = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        n_jobs=-1,
        random_state=42,
    )
    m_ratio.fit(X_train, y_ratio_tr)
    y_pred = m_ratio.predict(X_test)
    print("ratio R2:", r2_score(y_ratio_te, y_pred))
    print("ratio MAE:", mean_absolute_error(y_ratio_te, y_pred))
    joblib.dump(m_ratio, f"{out_dir}/model_ratio_rf.joblib")

    # ---- Partition count model ----
    print("Training model_parts...")
    m_parts = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        n_jobs=-1,
        random_state=42,
    )
    m_parts.fit(X_train, y_parts_tr)
    y_pred_p = m_parts.predict(X_test)
    print("parts R2:", r2_score(y_parts_te, y_pred_p))
    print("parts MAE:", mean_absolute_error(y_parts_te, y_pred_p))
    joblib.dump(m_parts, f"{out_dir}/model_parts_rf.joblib")

    # ---- Pattern classifier ----
    print("Training model_pattern...")
    m_pattern = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        n_jobs=-1,
        random_state=42,
    )
    m_pattern.fit(X_train, y_pat_tr)
    y_pred_pat = m_pattern.predict(X_test)
    print("pattern acc:", accuracy_score(y_pat_te, y_pred_pat))
    joblib.dump(m_pattern, f"{out_dir}/model_pattern_rf.joblib")

    # ---- Base set classifier ----
    print("Training model_bases...")
    m_bases = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        n_jobs=-1,
        random_state=42,
    )
    m_bases.fit(X_train, y_base_tr)
    y_pred_bs = m_bases.predict(X_test)
    print("bases acc:", accuracy_score(y_base_te, y_pred_bs))
    joblib.dump(m_bases, f"{out_dir}/model_bases_rf.joblib")

    print("All models saved in", out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Training dataset CSV (from features_and_labels.py)")
    ap.add_argument("--out", default="models", help="Output models directory")
    args = ap.parse_args()

    train_all(args.csv, args.out)
