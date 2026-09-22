#!/usr/bin/env python3
"""
Ride-fare regression: leak-free pipeline + embedding MLP (PyTorch)
==================================================================
Run:      python fare_dl.py            (full run)
          QUICK=1 python fare_dl.py    (2-epoch smoke test, ~1 min)
Needs:    pip install torch pandas numpy scikit-learn
Input:    df_normalized.csv  = the RAW file you fed to preprocess.py (edit CSV_PATH)

What it does
------------
1. Cleans the raw data exactly like your preprocess.py (drops rate-card rows and
   duplicates, groups payment types, DROPS `weather` because it leaks the fare).
2. Splits 70/15/15 FIRST, then fits every statistic (scaler, medians, location
   counts, vocabularies) on the TRAIN part only.
3. Keeps information your pipeline threw away: location identity and exact date.
4. Trains baselines + 3 embedding-MLP variants (ablation) + a seed ensemble of the
   best one, and prints one compact results table.

Paste the printed output back to me and I will tell you what to change next.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ================================ CONFIG ====================================
CSV_PATH = "C:\Users\lenovo\OneDrive\Desktop\Folder\cab_bike_auto_fare_prediction\M1\ml_ready_fare.csv"
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15

MIN_LOC_COUNT = 5      # location seen < 5x in TRAIN  -> shared "unknown" id
MIN_DATE_COUNT = 3     # same idea for exact dates

EPOCHS = 60
PATIENCE = 8           # early stopping on validation R2
BATCH = 2048
LR = 2e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.25
HIDDEN = (512, 256, 128)
LOSS = "mse"           # "mse" is aligned with R2; try "huber" if the fare tail hurts
FINAL_SEEDS = 3        # size of the final ensemble (best config only)

if os.environ.get("QUICK") == "1":
    EPOCHS, FINAL_SEEDS = 2, 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NON_TRIP_SOURCES = ["delhi_notification_2023", "aru_dto_2019"]
PAYMENT_MAP = {
    "cash": "cash",
    "card": "card", "credit card": "card", "debit card": "card",
    "upi": "upi", "gpay": "upi", "qr scan": "upi",
    "wallet": "wallet", "uber wallet": "wallet", "paytm": "wallet", "amazon pay": "wallet",
}

BASE_CATS = ["source", "vehicle", "payment", "season", "year", "month", "dow", "hour", "day"]
LOC_CATS = ["pickup_id", "drop_id"]
DATE_CATS = ["date_id"]
CONFIGS = {
    "MLP A: base (no ids)": BASE_CATS,
    "MLP B: + location emb": BASE_CATS + LOC_CATS,
    "MLP C: + location + date emb": BASE_CATS + LOC_CATS + DATE_CATS,
}


# ============================ DATA PREPARATION ==============================
def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]:,} rows x {df.shape[1]} cols")

    src = df["source_dataset"].astype(str).str.strip().str.lower()
    df = df.loc[~src.isin(NON_TRIP_SOURCES)].copy()
    n_after_filter = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] dropped {len(src) - n_after_filter:,} rate-card rows and "
          f"{n_after_filter - len(df):,} duplicates -> {len(df):,} rows")

    for c in ["source_dataset", "vehicle_type", "payment_method", "season"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["payment_group"] = df["payment_method"].map(PAYMENT_MAP)
    assert df["payment_group"].notna().all(), "unmapped payment_method"

    date = pd.to_datetime(df["date"].astype(str).str[:10], format="%Y-%m-%d")
    df["date_str"] = date.dt.strftime("%Y-%m-%d")
    df["year"], df["month"], df["day"], df["dow"] = date.dt.year, date.dt.month, date.dt.day, date.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_phase_imputed"] = df["is_phase_imputed"].astype(int)

    for c in ["pickup_location", "drop_location"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["same_pickup_drop"] = (df["pickup_location"] == df["drop_location"]).astype(int)
    df["fare_amount"] = df["fare_amount"].astype(float)
    return df          # NOTE: `weather` and `status` are intentionally never used


def diagnostics(df):
    y = df["fare_amount"]
    print(f"[target] mean={y.mean():.1f}  std={y.std():.1f}  skew={y.skew():.2f}  "
          f"median={y.median():.0f}  max={y.max():.0f}")
    print("[corr(distance_km, fare) inside each source, rows with distance > 0]")
    for s, g in df[df["distance_km"] > 0].groupby("source_dataset"):
        print(f"   {s:<16s} n={len(g):>7,}  corr={g['distance_km'].corr(g['fare_amount']):+.3f}"
              f"   mean fare={g['fare_amount'].mean():7.1f}")


def _cyc(d, name, values, period):
    d[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
    d[f"{name}_cos"] = np.cos(2 * np.pi * values / period)


def build_data(df, tr_idx):
    """Builds numeric matrix + categorical id arrays. Every statistic uses TRAIN rows only."""
    tr = df.iloc[tr_idx]
    d = pd.DataFrame(index=df.index)

    # distance: 0 means "not recorded" -> NaN -> per-source TRAIN median, keep a flag
    pos = df["distance_km"].where(df["distance_km"] > 0)
    med = tr["distance_km"].where(tr["distance_km"] > 0).groupby(tr["source_dataset"]).median()
    dist = pos.fillna(df["source_dataset"].map(med)).fillna(med.median())
    d["distance_km"] = dist
    d["log_distance"] = np.log1p(dist)
    d["distance_missing"] = (df["distance_km"] <= 0).astype(int)

    d["driver_rating"] = df["driver_rating"]
    d["customer_rating"] = df["customer_rating"]
    for c in ["is_weekend", "is_phase_imputed", "same_pickup_drop"]:
        d[c] = df[c]

    # hour: fill with the TRAIN median hour of its phase_of_day, keep a "known" flag
    hour_raw = df["hour"]
    d["hour_known"] = hour_raw.notna().astype(int)
    phase_med = tr.loc[tr["hour"].notna()].groupby("phase_of_day")["hour"].median().round()
    hour = hour_raw.fillna(df["phase_of_day"].map(phase_med)).fillna(12.0)
    _cyc(d, "hour", hour, 24)
    _cyc(d, "month", df["month"], 12)
    _cyc(d, "day", df["day"], 31)
    _cyc(d, "dow", df["dow"], 7)

    # location popularity (log count in TRAIN)
    loc_counts = pd.concat([tr["pickup_location"], tr["drop_location"]]).value_counts()
    d["pickup_freq"] = np.log1p(df["pickup_location"].map(loc_counts).fillna(0))
    d["drop_freq"] = np.log1p(df["drop_location"].map(loc_counts).fillna(0))

    d = d.fillna(d.iloc[tr_idx].median())
    num = d.to_numpy(dtype=np.float64)
    mu, sd = num[tr_idx].mean(0), num[tr_idx].std(0)
    sd[sd == 0] = 1.0
    num = ((num - mu) / sd).astype(np.float32)

    # categorical ids (0 = unknown / rare, vocabulary built on TRAIN only)
    cat, cards = {}, {}

    def add(name, series, min_count=1):
        counts = series.iloc[tr_idx].value_counts()
        vocab = {v: i + 1 for i, v in enumerate(counts[counts >= min_count].index)}
        cat[name] = series.map(vocab).fillna(0).astype(np.int64).to_numpy()
        cards[name] = len(vocab) + 1

    add("source", df["source_dataset"])
    add("vehicle", df["vehicle_type"])
    add("payment", df["payment_group"])
    add("season", df["season"])
    add("year", df["year"])
    add("month", df["month"])
    add("dow", df["dow"])
    add("hour", hour_raw.fillna(-1).astype(int))
    add("day", df["day"])
    add("date_id", df["date_str"], MIN_DATE_COUNT)

    keep = loc_counts[loc_counts >= MIN_LOC_COUNT].index
    loc_vocab = {v: i + 1 for i, v in enumerate(keep)}
    cat["pickup_id"] = df["pickup_location"].map(loc_vocab).fillna(0).astype(np.int64).to_numpy()
    cat["drop_id"] = df["drop_location"].map(loc_vocab).fillna(0).astype(np.int64).to_numpy()
    cards["loc"] = len(loc_vocab) + 1          # pickup & drop share ONE embedding table

    y = df["fare_amount"].to_numpy(dtype=np.float64)
    y_mean, y_std = y[tr_idx].mean(), y[tr_idx].std()
    print(f"[data]  {num.shape[1]} numeric features | known locations: {len(loc_vocab):,} "
          f"| known dates: {cards['date_id'] - 1:,}")
    return dict(num=num, cat=cat, cards=cards, y=y, y_mean=y_mean, y_std=y_std,
                y_scaled=((y - y_mean) / y_std).astype(np.float32))


# ================================ METRICS ===================================
def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


# =============================== BASELINES ==================================
def baseline_source_mean(df, tr_idx, idx_list):
    m = df.iloc[tr_idx].groupby("source_dataset")["fare_amount"].mean()
    g = df.iloc[tr_idx]["fare_amount"].mean()
    return [df.iloc[i]["source_dataset"].map(m).fillna(g).to_numpy() for i in idx_list]


def baseline_hgb(data, tr_idx, va_idx, te_idx):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.column_stack([data["num"]] + [data["cat"][c] for c in BASE_CATS]).astype(np.float32)
    cat_idx = list(range(data["num"].shape[1], X.shape[1]))
    m = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features=cat_idx, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
    m.fit(X[tr_idx], data["y"][tr_idx])
    return m.predict(X[va_idx]), m.predict(X[te_idx])


# ============================== EMBEDDING MLP ===============================
def table_of(col):
    return "loc" if col in ("pickup_id", "drop_id") else col


def emb_dim(card):
    return int(max(2, min(50, round(1.6 * card ** 0.56))))


class EmbMLP(nn.Module):
    def __init__(self, cols, cards, n_num, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.tables = [table_of(c) for c in cols]
        self.emb = nn.ModuleDict({t: nn.Embedding(cards[t], emb_dim(cards[t])) for t in set(self.tables)})
        d = sum(self.emb[t].embedding_dim for t in self.tables) + n_num
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.emb_drop = nn.Dropout(0.1)

    def forward(self, xc, xn):
        z = torch.cat([self.emb[t](xc[:, i]) for i, t in enumerate(self.tables)] + [xn], dim=1)
        return self.mlp(self.emb_drop(z)).squeeze(1)


@torch.no_grad()
def predict(model, xc, xn, bs=16384):
    model.eval()
    return torch.cat([model(xc[i:i + bs], xn[i:i + bs]) for i in range(0, len(xc), bs)]).cpu().numpy()


def fit_mlp(data, cols, tr_idx, va_idx, te_idx, seed, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_t(idx):
        xc = torch.tensor(np.stack([data["cat"][c][idx] for c in cols], 1), dtype=torch.long, device=DEVICE)
        xn = torch.tensor(data["num"][idx], dtype=torch.float32, device=DEVICE)
        y = torch.tensor(data["y_scaled"][idx], dtype=torch.float32, device=DEVICE)
        return xc, xn, y

    tr, va, te = to_t(tr_idx), to_t(va_idx), to_t(te_idx)
    y_va = data["y_scaled"][va_idx]      # R2 is unchanged by the affine target scaling

    model = EmbMLP(cols, data["cards"], data["num"].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = nn.MSELoss() if LOSS == "mse" else nn.HuberLoss(delta=1.0)

    best, best_ep, best_state, bad = -1e9, 0, None, 0
    n = len(tr_idx)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:                       # BatchNorm needs > 1 sample
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(tr[0][b], tr[1][b]), tr[2][b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        v = r2(y_va, predict(model, va[0], va[1]))
        sched.step(v)
        if v > best + 1e-5:
            best, best_ep, bad = v, ep + 1, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or bad == 0 and ep < 3):
            print(f"      epoch {ep + 1:>3d}  val R2 = {v:+.4f}  (best {best:+.4f} @ {best_ep})")
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    inv = lambda p: p * data["y_std"] + data["y_mean"]
    return inv(predict(model, va[0], va[1])), inv(predict(model, te[0], te[1])), best_ep


# ================================== MAIN ====================================
def main():
    t0 = time.time()
    print(f"[device] {DEVICE}   [epochs] {EPOCHS}")
    df = load_and_clean(CSV_PATH)
    diagnostics(df)

    perm = np.random.RandomState(SEED).permutation(len(df))
    n_te, n_va = int(len(df) * TEST_FRAC), int(len(df) * VAL_FRAC)
    te_idx, va_idx, tr_idx = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    print(f"[split] train {len(tr_idx):,} | val {len(va_idx):,} | test {len(te_idx):,}")

    data = build_data(df, tr_idx)
    y_va, y_te = data["y"][va_idx], data["y"][te_idx]
    rows = []

    def record(name, pv, pt, note=""):
        rows.append((name, r2(y_va, pv), r2(y_te, pt), rmse(y_te, pt), mae(y_te, pt), note))

    pv, pt = baseline_source_mean(df, tr_idx, [va_idx, te_idx])
    record("Baseline: mean per source", pv, pt)

    try:
        pv, pt = baseline_hgb(data, tr_idx, va_idx, te_idx)
        record("Baseline: HistGradientBoosting", pv, pt, "no location/date ids")
        print(f"[hgb]   done ({time.time() - t0:.0f}s)")
    except Exception as e:                                   # noqa: BLE001
        print(f"[hgb]   skipped: {e}")

    preds = {}
    for name, cols in CONFIGS.items():
        print(f"[train] {name}")
        pv, pt, ep = fit_mlp(data, cols, tr_idx, va_idx, te_idx, seed=SEED)
        preds[name] = [(pv, pt)]
        record(name, pv, pt, f"best epoch {ep}")
        print(f"[train] {name}: val R2 {rows[-1][1]:+.4f} | test R2 {rows[-1][2]:+.4f} ({time.time() - t0:.0f}s)")

    # ensemble of the best config (chosen on VALIDATION R2, never on test)
    mlp_rows = [r for r in rows if r[0] in CONFIGS]
    best_name = max(mlp_rows, key=lambda r: r[1])[0]
    if FINAL_SEEDS > 1:
        print(f"[ens]   {best_name}: training {FINAL_SEEDS - 1} more seeds")
        for k in range(1, FINAL_SEEDS):
            pv, pt, _ = fit_mlp(data, CONFIGS[best_name], tr_idx, va_idx, te_idx, seed=SEED + k, verbose=False)
            preds[best_name].append((pv, pt))
    pv = np.mean([p[0] for p in preds[best_name]], axis=0)
    pt = np.mean([p[1] for p in preds[best_name]], axis=0)
    record(f"ENSEMBLE x{len(preds[best_name])} of '{best_name[:5].strip()}'", pv, pt)

    # ------------------------------ report ------------------------------
    lines = [f"{'model':<40s}{'val R2':>9s}{'test R2':>9s}{'RMSE':>9s}{'MAE':>9s}  note"]
    for name, v, t, rm, ma, note in rows:
        lines.append(f"{name:<40s}{v:>9.4f}{t:>9.4f}{rm:>9.1f}{ma:>9.1f}  {note}")
    lines.append("")
    lines.append("per-source test R2 of the ensemble:")
    src_te = df["source_dataset"].iloc[te_idx].to_numpy()
    for s in np.unique(src_te):
        m = src_te == s
        lines.append(f"   {s:<16s} n={m.sum():>6,}  R2={r2(y_te[m], pt[m]):+.4f}  "
                     f"RMSE={rmse(y_te[m], pt[m]):7.1f}")
    lines.append(f"\ntotal time {time.time() - t0:.0f}s | LOSS={LOSS} | HIDDEN={HIDDEN} | DROPOUT={DROPOUT} | LR={LR}")
    report = "\n".join(lines)
    print("\n" + "=" * 100 + "\n" + report + "\n" + "=" * 100)
    with open("results.txt", "w") as f:
        f.write(report)
    print("saved -> results.txt   (paste that file's content back to me)")


if __name__ == "__main__":
    main()
