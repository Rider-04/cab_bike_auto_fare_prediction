#!/usr/bin/env python3
"""
Ride-hailing dataset  ->  fully numeric, ML/DL-ready dataset
=============================================================

Usage
-----
    # Regression: predict fare_amount
    python preprocess.py --input df_normalized.csv --target fare_amount --output ml_ready_fare.csv

    # Classification: predict ride outcome (status)
    python preprocess.py --input df_normalized.csv --target status --output ml_ready_status.csv

Output
------
A CSV in which EVERY column is numeric, with no missing values.
All columns except the last one (`target`) are features.
A `<output>_meta.json` file is also written (feature list, scaler stats, label map).

Requirements: pandas, numpy, scikit-learn
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# These two sources are official rate cards / fixed tariffs, not real rides
# (no date, no drop location, constant ratings, status = official_rate_card/fixed/...).
NON_TRIP_SOURCES = ["delhi_notification_2023", "aru_dto_2019"]

# 11 raw payment labels -> 4 meaningful groups
PAYMENT_MAP = {
    "cash": "cash",
    "card": "card", "credit card": "card", "debit card": "card",
    "upi": "upi", "gpay": "upi", "qr scan": "upi",
    "wallet": "wallet", "uber wallet": "wallet", "paytm": "wallet", "amazon pay": "wallet",
}

# `success` and `completed` mean the same thing (different source datasets)
STATUS_MAP = {
    "success": "completed",
    "completed": "completed",
    "incomplete": "incomplete",
    "canceled by customer": "canceled by customer",
    "canceled by driver": "canceled by driver",
    "driver not found": "driver not found",
}
STATUS_CODES = {  # label encoding for the classification target
    "completed": 0,
    "incomplete": 1,
    "canceled by customer": 2,
    "canceled by driver": 3,
    "driver not found": 4,
}

# When predicting `status`, these columns leak the answer:
#  - ratings are filled with a constant 4.2 for every non-completed ride
#  - distance_km is 0 for every cancelled booking
#  - weather == "unknown" occurs only for cancelled bookings
LEAKY_FOR_STATUS = ["driver_rating", "customer_rating", "distance_km", "distance_missing", "weather"]

# When predicting `fare_amount`, `weather` leaks the answer: "raining" vs "sunny" was
# generated from fare per km (a single threshold of ~48.3 INR/km separates them with
# 100% accuracy in every source), so it is not real weather.
LEAKY_FOR_FARE = ["weather"]

ONE_HOT_COLS = ["source_dataset", "vehicle_type", "payment_group", "season", "weather"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def add_cyclical(df, col, period):
    """Encode a cyclic variable (hour, month, ...) as sin/cos so 23h is 'near' 0h."""
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)


def main(args):
    df = pd.read_csv(args.input)
    print(f"[load]      {df.shape[0]:,} rows x {df.shape[1]} cols")
    dropped = {}

    # ------------------------------------------------------------------ 1. remove non-ride rows
    mask = df["source_dataset"].isin(NON_TRIP_SOURCES)
    dropped["non_trip_rows"] = int(mask.sum())
    df = df.loc[~mask].copy()
    print(f"[filter]    dropped {dropped['non_trip_rows']:,} rate-card / tariff rows")

    # ------------------------------------------------------------------ 2. remove exact duplicates
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped["duplicate_rows"] = n - len(df)
    print(f"[dedupe]    dropped {dropped['duplicate_rows']:,} exact duplicate rows")

    # ------------------------------------------------------------------ 3. clean text / consolidate categories
    for c in ["source_dataset", "vehicle_type", "payment_method", "status", "state", "season", "weather"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["payment_group"] = df["payment_method"].map(PAYMENT_MAP)
    df["status"] = df["status"].map(STATUS_MAP)
    assert df["payment_group"].notna().all(), "unmapped payment_method"
    assert df["status"].notna().all(), "unmapped status"

    # ------------------------------------------------------------------ 4. distance: 0 == "not recorded"
    df["distance_missing"] = (df["distance_km"] <= 0).astype(int)
    df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan
    df["distance_km"] = df["distance_km"].fillna(df.groupby("source_dataset")["distance_km"].transform("median"))

    # ------------------------------------------------------------------ 5. date / time features
    # two date formats exist ("YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS") -> keep first 10 chars
    date = pd.to_datetime(df["date"].astype(str).str[:10], format="%Y-%m-%d")
    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["day"] = date.dt.day
    df["dow"] = date.dt.dayofweek  # 0 = Monday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # hour is missing where phase_of_day was imputed -> fill with the typical hour of that phase
    df["is_phase_imputed"] = df["is_phase_imputed"].astype(int)
    phase_hour = df.loc[df["hour"].notna()].groupby("phase_of_day")["hour"].median().round()
    df["hour"] = df["hour"].fillna(df["phase_of_day"].map(phase_hour))
    assert df["hour"].notna().all()

    add_cyclical(df, "hour", 24)
    add_cyclical(df, "month", 12)
    add_cyclical(df, "day", 31)
    add_cyclical(df, "dow", 7)

    # ------------------------------------------------------------------ 6. location features (12.8k unique values)
    # One-hot would create ~13k columns, so use log-frequency ("how popular is this place")
    # + a same-pickup/drop flag. Frequency is unsupervised, so it is not target leakage.
    df["pickup_location"] = df["pickup_location"].astype(str).str.strip().str.lower()
    df["drop_location"] = df["drop_location"].astype(str).str.strip().str.lower()
    counts = pd.concat([df["pickup_location"], df["drop_location"]]).value_counts()
    df["pickup_freq"] = np.log1p(df["pickup_location"].map(counts))
    df["drop_freq"] = np.log1p(df["drop_location"].map(counts))
    df["same_pickup_drop"] = (df["pickup_location"] == df["drop_location"]).astype(int)
    if args.keep_location_ids:  # integer ids for nn.Embedding / tree models
        ids = {loc: i for i, loc in enumerate(counts.index)}
        df["pickup_id"] = df["pickup_location"].map(ids)
        df["drop_id"] = df["drop_location"].map(ids)

    # ------------------------------------------------------------------ 7. target + leakage handling
    if args.target == "fare_amount":
        y = df["fare_amount"].astype(float)
        drop_cols = ["fare_amount"]
        if not args.keep_leaky:
            drop_cols += LEAKY_FOR_FARE
            print(f"[leakage]   dropped columns that leak `fare_amount`: {LEAKY_FOR_FARE}")
    else:
        y = df["status"].map(STATUS_CODES).astype(int)
        drop_cols = []
        if not args.keep_leaky:
            drop_cols += [c for c in LEAKY_FOR_STATUS if c in df.columns]
            print(f"[leakage]   dropped columns that leak `status`: {LEAKY_FOR_STATUS}")

    # ------------------------------------------------------------------ 8. drop everything that is now redundant / raw
    drop_cols += [
        "date", "hour", "month", "day", "dow",          # replaced by sin/cos + flags
        "phase_of_day",                                  # fully determined by hour
        "weekday",                                       # duplicate of dow
        "payment_method", "status",                      # replaced by payment_group / target
        "state",                                         # fully determined by source_dataset
        "pickup_location", "drop_location",              # replaced by frequency features
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ------------------------------------------------------------------ 9. one-hot encode low-cardinality categoricals
    ohe_cols = [c for c in ONE_HOT_COLS if c in X.columns]
    X = pd.get_dummies(X, columns=ohe_cols, drop_first=True, dtype=int)

    # ------------------------------------------------------------------ 10. standardise continuous columns
    to_scale = [c for c in ["distance_km", "fare_amount", "driver_rating", "customer_rating",
                            "year", "pickup_freq", "drop_freq"] if c in X.columns]
    scaler = StandardScaler()
    X[to_scale] = scaler.fit_transform(X[to_scale])

    # ------------------------------------------------------------------ 11. assemble + validate
    out = X.copy()
    out["target"] = y.values
    assert not out.isna().any().any(), "NaNs remain"
    assert np.isfinite(out.to_numpy(dtype=float)).all(), "inf remains"
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes), "non-numeric column remains"

    out.to_csv(args.output, index=False, float_format="%.6g")
    meta = {
        "target": args.target,
        "n_rows": int(len(out)),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "scaled_columns": {c: {"mean": float(m), "std": float(s)}
                           for c, m, s in zip(to_scale, scaler.mean_, scaler.scale_)},
        "rows_dropped": dropped,
        "status_label_map": STATUS_CODES if args.target == "status" else None,
    }
    with open(args.output.rsplit(".", 1)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done]      {out.shape[0]:,} rows x {X.shape[1]} features + 1 target  ->  {args.output}")
    print(f"[features]  {list(X.columns)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="df_normalized.csv")
    p.add_argument("--output", default="ml_ready.csv")
    p.add_argument("--target", choices=["fare_amount", "status"], default="fare_amount")
    p.add_argument("--keep-leaky", action="store_true",
                   help="keep the columns that leak the target (weather for fare; ratings/distance/weather for status). Not recommended")
    p.add_argument("--keep-location-ids", action="store_true",
                   help="also output pickup_id/drop_id integer codes (for embedding layers)")
    main(p.parse_args())
