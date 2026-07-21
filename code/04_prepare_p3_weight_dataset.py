import pandas as pd
from pathlib import Path


# =========================
# 1. Read data
# =========================

input_path = (
    "./data/lead1_F1_wgrid50_100.parquet"
)


df = pd.read_parquet(
    input_path
)


print("===================")
print("Original data")
print("===================")

print(df.shape)



# =========================
# 2. Define protected columns
# =========================


keep_cols = [

    # sample id
    "cell_id",
    "date",

    # label
    "y",

    # ecosystem
    "BIOME",
    "STRATUM",

    # spatial split
    "grid10km_id",
    "grid50km_id",
    "grid100km_id",

    # weights
    "pi",
    "d_weight",
    "weight",

    # time 
    "year",
    "month",

]



# =========================
# 3. Remove leakage columns
# =========================

drop_cols = [

    # duplicate target
    "fire_tslf",

]


df = df.drop(
    columns=[
        c for c in drop_cols
        if c in df.columns
    ]
)



# =========================
# 4. Find feature columns
# =========================


protected = set(
    keep_cols
)


feature_cols = [

    c for c in df.columns

    if c not in protected

    and c != "valid_core"

    and c != "pi_clip"

    and c != "d_weight_clip"

]


print("===================")
print("Feature number")
print("===================")

print(
    len(feature_cols)
)



# =========================
# 5. Build final dataset
# =========================


final_cols = (
    keep_cols
    +
    feature_cols
)


p3_df = df[
    final_cols
].copy()



# =========================
# 6. Check
# =========================


print("===================")
print("Final dataset")
print("===================")


print(
    p3_df.shape
)



print("\nMissing ratio top 10:")

print(
    p3_df.isna()
    .mean()
    .sort_values(
        ascending=False
    )
    .head(10)
)



print("\nSpatial columns:")

for c in [
    "grid10km_id",
    "grid50km_id",
    "grid100km_id"
]:

    print(
        c,
        c in p3_df.columns
    )



# =========================
# 7. Save
# =========================


output = Path(
    "./data/p3_weight_dataset.parquet"
)


p3_df.to_parquet(
    output,
    index=False
)


print("\nSaved:")
print(output)