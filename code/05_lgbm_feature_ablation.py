"""
LightGBM Feature Ablation Experiment

Purpose:
Evaluate contribution of physical feature groups
using LightGBM classifier.

Experiments:

1. Full features
2. Without fire history
3. Without drought/water features
4. Without vegetation features


Protocol:
P1 time holdout

Train:
2015-2021

Validation:
2022


Model:
LightGBM


Metrics:
AUPRC
ROC-AUC
Brier Score


Outputs:

results/

    lgbm_feature_ablation_results.csv

    lgbm_feature_ablation_predictions.csv

    lgbm_feature_ablation_drop.png

"""


import os
from pathlib import Path

import pandas as pd
import numpy as np

import lightgbm as lgb


from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)





# ======================================================
# 1. Path settings
# ======================================================


DATA_PATH = (
    "../data/lead1_F1_wgrid50_100.parquet"
)


FEATURE_DIR = Path(
    "../data"
)


OUTPUT_DIR = Path(
    "../results"
)


OUTPUT_DIR.mkdir(
    exist_ok=True
)





# ======================================================
# 2. Load data
# ======================================================


print("\nLoading dataset...")


df = pd.read_parquet(
    DATA_PATH,
    engine="fastparquet"
)


print(
    "Dataset shape:",
    df.shape
)



df["date"] = pd.to_datetime(
    df["date"]
)



df["year"] = (
    df["date"]
    .dt.year
)





# ======================================================
# 3. Create P1 time split
# ======================================================


train_df = df[
    df["year"].between(
        2015,
        2021
    )
].copy()



val_df = df[
    df["year"] == 2022
].copy()



print("\nTrain:")
print(train_df.shape)



print("\nValidation:")
print(val_df.shape)



print(
    "Validation positive rate:",
    val_df["y"].mean()
)





# ======================================================
# 4. Feature sets
# ======================================================


feature_sets = {


    "Full":

    "full_features.csv",



    "No Fire History":

    "no_fire_features.csv",



    "No Drought":

    "no_drought_features.csv",



    "No Vegetation":

    "no_vegetation_features.csv"

}





# ======================================================
# 5. Exclude leakage columns
# ======================================================


exclude_cols = [

    # identifiers

    "cell_id",

    "date",


    # label

    "y",


    # split information

    "split_role",

    "protocol",

    "fold_id",



    # spatial IDs

    "grid10km_id",

    "grid50km_id",

    "grid100km_id",

    "spatial_block_id",

    "tile_id",



    # metadata

    "BIOME",

    "STRATUM",


    # time

    "year",

]





# ======================================================
# 6. LightGBM model
# ======================================================


def build_lgbm_model():


    model = lgb.LGBMClassifier(

        n_estimators=500,


        learning_rate=0.03,


        max_depth=5,


        num_leaves=31,


        class_weight="balanced",


        subsample=0.8,


        colsample_bytree=0.8,


        random_state=42,


        n_jobs=-1

    )


    return model





# ======================================================
# 7. Run experiments
# ======================================================


results = []

prediction_list = []





for name, file in feature_sets.items():


    print("\n")
    print("="*60)

    print(
        "Running:",
        name
    )

    print("="*60)



    feature_path = (
        FEATURE_DIR /
        file
    )



    feature_df = pd.read_csv(
        feature_path
    )



    feature_cols = (
        feature_df["feature"]
        .tolist()
    )



    print(
        "Number of features:",
        len(feature_cols)
    )



    # check missing

    missing = [

        c

        for c in feature_cols

        if c not in df.columns

    ]


    if missing:

        raise ValueError(
            f"Missing features:\n{missing}"
        )



    # remove leakage

    feature_cols = [

        c

        for c in feature_cols

        if c not in exclude_cols

    ]



    print(
        "Final model features:",
        len(feature_cols)
    )



    X_train = train_df[
        feature_cols
    ]


    y_train = train_df[
        "y"
    ].astype(int)



    X_val = val_df[
        feature_cols
    ]


    y_val = val_df[
        "y"
    ].astype(int)





    # ---------------------
    # Train
    # ---------------------


    model = build_lgbm_model()



    print(
        "Training LightGBM..."
    )



    model.fit(

        X_train,

        y_train,


        eval_set=[

            (
                X_val,

                y_val

            )

        ],


        eval_metric="aucpr",


        callbacks=[

            lgb.early_stopping(

                stopping_rounds=30,

                verbose=False

            )

        ]

    )





    # ---------------------
    # Prediction
    # ---------------------


    y_prob = (

        model

        .predict_proba(

            X_val

        )[:,1]

    )





    # ---------------------
    # Metrics
    # ---------------------


    auprc = average_precision_score(

        y_val,

        y_prob

    )



    roc_auc = roc_auc_score(

        y_val,

        y_prob

    )



    brier = brier_score_loss(

        y_val,

        y_prob

    )



    print()

    print(
        f"AUPRC: {auprc:.6f}"
    )


    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )


    print(
        f"Brier: {brier:.6f}"
    )





    results.append(

        {

            "Model":

            "LightGBM",


            "Feature_Set":

            name,


            "n_features":

            len(feature_cols),


            "AUPRC":

            auprc,


            "ROC_AUC":

            roc_auc,


            "Brier":

            brier

        }

    )





    pred_df = val_df[

        [

            "cell_id",

            "date",

            "y"

        ]

    ].copy()



    pred_df["Feature_Set"] = name



    pred_df["prediction"] = y_prob



    prediction_list.append(
        pred_df
    )





# ======================================================
# 8. Save results
# ======================================================


result_df = pd.DataFrame(
    results
)



full_score = (

    result_df

    .loc[

        result_df["Feature_Set"]

        =="Full",

        "AUPRC"

    ]

    .iloc[0]

)



result_df["AUPRC_change"] = (

    result_df["AUPRC"]

    -

    full_score

)



result_df["AUPRC_drop"] = (

    full_score

    -

    result_df["AUPRC"]

)



result_df.to_csv(

    OUTPUT_DIR /

    "lgbm_feature_ablation_results.csv",

    index=False

)



print("\nFinal Results:")

print(result_df)





# ======================================================
# 9. Save predictions
# ======================================================


predictions = pd.concat(

    prediction_list,

    ignore_index=True

)



predictions.to_csv(

    OUTPUT_DIR /

    "lgbm_feature_ablation_predictions.csv",

    index=False

)





# ======================================================
# 10. Plot
# ======================================================


import matplotlib.pyplot as plt


plot_df = result_df[
    result_df["Feature_Set"] != "Full"
].copy()


# 排序
order = [
    "No Fire History",
    "No Drought",
    "No Vegetation"
]

plot_df["Feature_Set"] = pd.Categorical(
    plot_df["Feature_Set"],
    categories=order,
    ordered=True
)

plot_df = plot_df.sort_values(
    "Feature_Set"
)


# 计算变化
plot_df["AUPRC_Change"] = (
    plot_df["AUPRC"]
    -
    full_score
)



# ==============================
# Morandi colors
# ==============================

colors = [
    "#B8D8E8",  
    "#F8E2CF",   
    "#D8C8E4"   
]



plt.figure(
    figsize=(9,5.5)
)



bars = plt.barh(
    plot_df["Feature_Set"],
    plot_df["AUPRC_Change"],
    color=colors,
    edgecolor="#666666",
    linewidth=0.8
)



# zero line

plt.axvline(
    0,
    color="#555555",
    linestyle="--",
    linewidth=1
)



# grid

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)



# title

plt.title(
    "LightGBM Feature Ablation Results",
    fontsize=18,
    fontweight="bold",
    pad=15
)



plt.xlabel(
    "AUPRC Change vs Full Model",
    fontsize=12
)



plt.ylabel(
    "",
    fontsize=12
)



# ==============================
# 调整数字位置
# ==============================


for bar, value in zip(
    bars,
    plot_df["AUPRC_Change"]
):

    y = (
        bar.get_y()
        +
        bar.get_height()/2
    )


    if value < 0:

        # 放柱子内部右侧

        plt.text(
            value + 0.005,
            y,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
            fontweight="bold"
        )


    else:

        # 正值放柱子右侧

        plt.text(
            value + 0.001,
            y,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
            fontweight="bold"
        )



# 去除边框

ax = plt.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)



plt.tight_layout()



plt.savefig(
    "../results/lgbm_feature_ablation_morandi.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()



print(

    "\nFigure saved:"

)

print(

    OUTPUT_DIR /

    "lgbm_feature_ablation_drop.png"

)



print(

    "\nLightGBM feature ablation finished."

)