# train_nec_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====== CONFIG ======
CSV_PATH = "nec_dataset.csv"          # <-- yahan apna csv file naam
MODEL_KERAS_PATH = "nec_model_keras.h5"
MODEL_TFJS_DIR = "nec_tfjs_model"     # is folder me model.json aayega

# 26 features (same order as browser side)
FEATURE_COLUMNS = [f"f_{i:02d}" for i in range(26)]
# label columns
CORR_COL = "target_ratio"
PART_COL = "target_parts"
STRAT_COLS = ["target_strategy_header", "target_strategy_structure", "target_strategy_random"]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Features
    FEATURE_COLUMNS = [f"f_{i:02d}" for i in range(26)]
    X = df[FEATURE_COLUMNS].astype("float32").values

    # Labels
    y_corr = df["target_ratio"].astype("float32").values
    y_part = df["target_parts"].astype("int32").values

    # Convert target_pattern → 3 strategy weights
    # pattern: 0 = header, 1 = structure, 2 = random
    pattern = df["target_pattern"].astype(int).values

    strat = np.zeros((len(df), 3), dtype="float32")
    for i, p in enumerate(pattern):
        strat[i, p] = 1.0  # one-hot encode

    return X, y_corr, y_part, strat

def build_model(input_dim, num_part_classes):
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)

    # --- Branch 1: corruption ratio ---
    # output in 0..1, later map -> 0.001..0.01 in JS
    corr_raw = layers.Dense(1, activation="sigmoid", name="corr_raw")(x)

    # --- Branch 2: partition class ---
    # We treat partition_count as classification over unique values
    part_logits = layers.Dense(num_part_classes, activation="softmax", name="part_cls")(x)

    # --- Branch 3: strategy weights (3) ---
    strat_logits = layers.Dense(3, activation="softmax", name="strat")(x)

    model = keras.Model(inputs=inputs, outputs=[corr_raw, part_logits, strat_logits])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "corr_raw": "mse",
            "part_cls": "sparse_categorical_crossentropy",
            "strat": "mse",
        },
        loss_weights={
            "corr_raw": 1.0,
            "part_cls": 1.0,
            "strat": 0.5,  # thoda kam weight
        },
        metrics={
            "corr_raw": ["mse"],
            "part_cls": ["accuracy"],
            "strat": ["mse"],
        },
    )

    return model


def main():
    print("Loading dataset...")
    X, y_corr, y_part, y_strat = load_dataset(CSV_PATH)

    # ---- Normalize features 0..1 range for stability ----
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # partition ko classes me map karo
    unique_parts = np.sort(np.unique(y_part))
    print("Unique partition counts:", unique_parts)

    part_to_class = {p: i for i, p in enumerate(unique_parts)}
    class_to_part = {i: p for p, i in part_to_class.items()}

    y_part_cls = np.array([part_to_class[p] for p in y_part], dtype="int32")

    # Dataset split
    X_tr, X_val, yc_tr, yc_val, yp_tr, yp_val, ys_tr, ys_val = train_test_split(
        X_scaled, y_corr, y_part_cls, y_strat, test_size=0.2, random_state=42
    )

    print("Building model...")
    model = build_model(input_dim=X.shape[1], num_part_classes=len(unique_parts))
    model.summary()

    # Training
    history = model.fit(
        X_tr,
        {
            "corr_raw": yc_tr,
            "part_cls": yp_tr,
            "strat": ys_tr,
        },
        validation_data=(
            X_val,
            {
                "corr_raw": yc_val,
                "part_cls": yp_val,
                "strat": ys_val,
            },
        ),
        epochs=40,
        batch_size=64,
        verbose=1,
    )

    # Save Keras model
    print(f"Saving Keras model to {MODEL_KERAS_PATH} ...")
    model.save(MODEL_KERAS_PATH)

    # Also save partition mapping & scaler for reference (optional, for doc/debug)
    import json
    with open("partition_mapping.json", "w") as f:
        json.dump(class_to_part, f, indent=2)

    # Save scaler
    import joblib
    joblib.dump(scaler, "feature_scaler.pkl")

    print("Training done. Next: convert to TFJS with tensorflowjs_converter:")
    print(f"  tensorflowjs_converter --input_format keras {MODEL_KERAS_PATH} {MODEL_TFJS_DIR}")


if __name__ == "__main__":
    main()
