import math

import pandas as pd


BASE_FEATURES = [
    "lat",
    "lon",
    "sst",
    "chlor_a",
    "precip",
    "flood_inundation",
    "drought_index",
    "population_density",
    "water_access_pct",
    "sanitation_score",
    "mobility_index",
    "clinic_reports",
]


def build_feature_frame(df):
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["lat_bin"] = work["lat"].round(1)
    work["lon_bin"] = work["lon"].round(1)
    work["month"] = work["date"].dt.month
    work["season_sin"] = work["date"].dt.dayofyear.apply(
        lambda x: math.sin(2 * math.pi * (x / 365.0))
    )
    work["season_cos"] = work["date"].dt.dayofyear.apply(
        lambda x: math.cos(2 * math.pi * (x / 365.0))
    )
    work["heatwave"] = (work["sst"] > 28).astype(int)
    work["post_flood"] = (work["precip"] > 140).astype(int)
    work["heatwave_flood"] = work["heatwave"] * work["flood_inundation"]

    work = work.sort_values(["lat_bin", "lon_bin", "date"])
    grouped = work.groupby(["lat_bin", "lon_bin"], sort=False)
    work["precip_7d_mean"] = grouped["precip"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    work["precip_14d_sum"] = grouped["precip"].transform(
        lambda s: s.rolling(14, min_periods=1).sum()
    )
    work["sst_7d_mean"] = grouped["sst"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    work["chlor_a_7d_mean"] = grouped["chlor_a"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    work["flood_3d_max"] = grouped["flood_inundation"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    work["drought_30d_mean"] = grouped["drought_index"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )

    for col in [
        "precip_7d_mean",
        "precip_14d_sum",
        "sst_7d_mean",
        "chlor_a_7d_mean",
        "flood_3d_max",
        "drought_30d_mean",
    ]:
        work[col] = work[col].fillna(work[col].mean())

    feature_cols = BASE_FEATURES + [
        "month",
        "season_sin",
        "season_cos",
        "heatwave",
        "post_flood",
        "heatwave_flood",
        "precip_7d_mean",
        "precip_14d_sum",
        "sst_7d_mean",
        "chlor_a_7d_mean",
        "flood_3d_max",
        "drought_30d_mean",
    ]
    return work, feature_cols
