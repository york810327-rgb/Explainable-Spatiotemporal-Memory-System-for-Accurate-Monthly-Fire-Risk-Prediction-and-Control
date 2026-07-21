import pandas as pd


# =========================
# 1. Read data
# =========================

df_clean = pd.read_parquet("./data/df_clean_local.parquet") 
df_feature = pd.read_parquet("./data/lead1_F1_wgrid50_100.parquet")


print("clean data:")
print(df_clean.shape)

print("feature data:")
print(df_feature.shape)



# =========================
# 2. Check merge keys
# =========================

key_cols = [
    "cell_id",
    "date"
]


print("\nChecking duplicate keys...")


print(
    "clean duplicated:",
    df_clean.duplicated(key_cols).sum()
)


print(
    "feature duplicated:",
    df_feature.duplicated(key_cols).sum()
)


# =========================
# 3. Merge tables
# =========================

print("\nMerging data...")


df = df_clean.merge(
    df_feature.drop(
        columns=["y", "BIOME"],
        errors="ignore"
    ),
    on=["cell_id", "date"],
    how="left",
    validate="one_to_one",
)


print("Merged data:")
print(df.shape)

print("\nMissing ratio after merge:")

print(
    df.isna().mean()
    .sort_values(ascending=False)
    .head(10)
)


# =========================
# 4. Date processing
# =========================

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year


print("\nYear distribution:")
print(
    df["year"].value_counts()
    .sort_index()
)


print("\nSplit distribution:")
print(
    df["split_role"].value_counts()
)


# =========================
# 5. Build train and validation sets
# =========================

train_df = df[df["split_role"] == "train"].copy()
val_df = df[df["split_role"] == "val"].copy()

print("\nTrain shape:")
print(train_df.shape)

print("Validation shape:")
print(val_df.shape)

print("\nTrain year range:")
print(train_df["year"].min(), train_df["year"].max())

print("Validation year range:")
print(val_df["year"].min(), val_df["year"].max())

print("\nTrain positive samples:")
print(int(train_df["y"].sum()))

print("Validation positive samples:")
print(int(val_df["y"].sum()))

print("\nTrain positive rate:")
print(train_df["y"].mean())

print("Validation positive rate:")
print(val_df["y"].mean())


# =========================
# 6. Exclude non-predictor columns
# =========================

non_feature_cols = {
    "cell_id",
    "date",
    "y",
    "year",
    "month",
    "protocol",
    "fold_id",
    "split_role",
    "BIOME",
    "STRATUM",
    "grid10km_id",
    "grid50km_id",
    "grid100km_id",
    "pi",
    "d_weight",
    "pi_clip",
    "d_weight_clip",
    "weight",
    "valid_core",
}

feature_cols = [
    col
    for col in df.columns
    if col not in non_feature_cols
    and pd.api.types.is_numeric_dtype(df[col])
]

print("\nNumber of candidate features:")
print(len(feature_cols))

print("\nCandidate features:")
for col in feature_cols:
    print(col)

if (
    "fire_tslf" in feature_cols
    and "months_since_last_fire" in feature_cols
):
    feature_cols.remove("fire_tslf")

constant_cols = [
    col
    for col in feature_cols
    if train_df[col].nunique(dropna=False) <= 1
]

feature_cols = [
    col
    for col in feature_cols
    if col not in constant_cols
]

print("\nConstant columns removed:")
print(constant_cols)

print("\nFinal number of features:")
print(len(feature_cols))

X_train = train_df[feature_cols]
y_train = train_df["y"].astype(int)

X_val = val_df[feature_cols]
y_val = val_df["y"].astype(int)

print("\nX_train shape:")
print(X_train.shape)

print("X_val shape:")
print(X_val.shape)


# =========================
# 7. Build Logistic-L2 model
# =========================

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight=None,
                random_state=42,
            ),
        ),
    ]
)


print("\nTraining Logistic-L2 model...")

model.fit(
    X_train,
    y_train
)

print("Training finished.")


# =========================
# 8. Predict 2022 validation set
# =========================

val_prob = model.predict_proba(
    X_val
)[:, 1]


print("\nPrediction finished.")

print("Minimum predicted probability:")
print(val_prob.min())

print("Maximum predicted probability:")
print(val_prob.max())


# =========================
# 9. Evaluate validation performance
# =========================

val_auprc = average_precision_score(
    y_val,
    val_prob
)

val_roc_auc = roc_auc_score(
    y_val,
    val_prob
)

val_brier = brier_score_loss(
    y_val,
    val_prob
)

val_positive_rate = y_val.mean()


print("\n=========================")
print("2022 Validation Results")
print("=========================")

print(
    f"Positive rate: {val_positive_rate:.6f}"
)

print(
    f"AUPRC: {val_auprc:.6f}"
)

print(
    f"ROC-AUC: {val_roc_auc:.6f}"
)

print(
    f"Brier score: {val_brier:.6f}"
)


# =========================
# 10. Save validation predictions
# =========================

from pathlib import Path

output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)

val_predictions = val_df[
    [
        "cell_id",
        "date",
        "y",
        "BIOME",
    ]
].copy()

val_predictions["p_logistic_l2"] = val_prob

val_predictions.to_parquet(
    output_dir / "val2022_logistic_l2_predictions.parquet",
    index=False,
)

print("\nValidation predictions saved.")


# =========================
# 11. Save metrics
# =========================

metrics_df = pd.DataFrame(
    [
        {
            "model": "Logistic-L2",
            "protocol": "P1_time_holdout",
            "train_years": "2015-2021",
            "validation_year": 2022,
            "n_train": len(train_df),
            "n_validation": len(val_df),
            "n_features": len(feature_cols),
            "validation_positive_rate": val_positive_rate,
            "validation_auprc": val_auprc,
            "validation_roc_auc": val_roc_auc,
            "validation_brier": val_brier,
            "C": 1.0,
            "class_weight": "none",
            "sample_weight": "none",
        }
    ]
)

metrics_df.to_csv(
    output_dir / "logistic_l2_metrics.csv",
    index=False,
)

print("Metrics saved.")


# =========================
# 12. Save feature list and model
# =========================

import joblib

pd.DataFrame(
    {
        "feature": feature_cols
    }
).to_csv(
    output_dir / "logistic_l2_features.csv",
    index=False,
)

joblib.dump(
    model,
    output_dir / "logistic_l2_model.joblib",
)

print("Feature list and model saved.")