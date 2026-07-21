import pandas as pd
from pathlib import Path


# =========================
# 1. Read data
# =========================

data_path = "./data/lead1_F1_wgrid50_100.parquet"

df = pd.read_parquet(data_path)


print("Original shape:")
print(df.shape)



# =========================
# 2. Remove non-feature columns
# =========================

non_feature_cols = [

    # ID
    "cell_id",
    "date",

    # label
    "y",

    # time
    "year",
    "month",

    # weights / QC
    "pi",
    "d_weight",
    "pi_clip",
    "d_weight_clip",
    "weight",
    "valid_core",

    # analysis only
    "BIOME",
    "STRATUM",

    # spatial block ID
    "grid10km_id",
    "grid50km_id",
    "grid100km_id",

]


feature_cols = [
    c for c in df.columns
    if c not in non_feature_cols
]


# remove duplicate fire feature
if "fire_tslf" in feature_cols:
    feature_cols.remove("fire_tslf")


print("\nFinal feature number:")
print(len(feature_cols))



# =========================
# 3. Define physical groups
# =========================


# Fire history
fire_features = [

    "y_lag1",
    "y_lag12",
    "fire_count_12m",
    "months_since_last_fire",

]


# Drought / Climate
drought_features = [

    "PET_sum_mon_lag1",
    "P_sum_mon_lag1",

    "RH_mean_mon_lag1",
    "RH_min_mon_lag1",

    "SM1_mean_mon_lag1",
    "SM2_mean_mon_lag1",

    "TP_sum_mon_lag1",

    "Tmax_mon_lag1",
    "Tmean_mon_lag1",

    "VPD_max_mon_lag1",
    "VPD_mean_mon_lag1",

    "WD_R_mon_lag1",
    "WD_u_mon_lag1",
    "WD_v_mon_lag1",

    "WS_max_mon_lag1",
    "WS_mean_mon_lag1",
    "WS_strong_frac_lag1",

    "P_sum_mon_lag1_roll3m_sum",
    "PET_sum_mon_lag1_roll3m_sum",
    "VPD_mean_mon_lag1_roll3m_mean",

]


# Vegetation
vegetation_features = [

    "EVI_mean_mon_lag1",

    "LST_day_mon_lag1",

    "LST_night_mon_lag1",

    "NDVI_mean_mon_lag1",

    "NDVI_mean_mon_lag1_roll3m_mean",

]



# =========================
# 4. Check missing columns
# =========================

groups = {
    "fire": fire_features,
    "drought": drought_features,
    "vegetation": vegetation_features
}


for name, cols in groups.items():

    missing = [
        c for c in cols
        if c not in feature_cols
    ]

    if missing:
        print(
            f"\nWARNING {name} missing:"
        )
        print(missing)



# =========================
# 5. Generate ablation sets
# =========================


output_dir = Path("./data")

output_dir.mkdir(
    exist_ok=True
)


# Full model

pd.DataFrame(
    {
        "feature": feature_cols
    }
).to_csv(
    output_dir / "full_features.csv",
    index=False
)



# Without fire history

no_fire = [
    c for c in feature_cols
    if c not in fire_features
]


pd.DataFrame(
    {
        "feature": no_fire
    }
).to_csv(
    output_dir / "no_fire_features.csv",
    index=False
)



# Without drought

no_drought = [
    c for c in feature_cols
    if c not in drought_features
]


pd.DataFrame(
    {
        "feature": no_drought
    }
).to_csv(
    output_dir / "no_drought_features.csv",
    index=False
)



# Without vegetation

no_vegetation = [
    c for c in feature_cols
    if c not in vegetation_features
]


pd.DataFrame(
    {
        "feature": no_vegetation
    }
).to_csv(
    output_dir / "no_vegetation_features.csv",
    index=False
)



# =========================
# 6. Summary
# =========================


print("\n====================")
print("Ablation summary")
print("====================")


print(
    "Full:",
    len(feature_cols)
)

print(
    "No fire:",
    len(no_fire)
)

print(
    "No drought:",
    len(no_drought)
)

print(
    "No vegetation:",
    len(no_vegetation)
)


print("\nSaved to:")
print(output_dir)