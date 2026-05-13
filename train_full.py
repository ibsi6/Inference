"""
Train the production in-play tennis model on ALL available data (no OOS holdout).

Reads `inplay_dataset_v2.parquet` from the same directory as this script
(or pass --data to override). Writes the trained booster + meta into
`./artifacts/`.

Run once after upstream `build_inplay_v2.py`:

    python train_full.py

This script is included for transparency; the dataset it requires lives in
the upstream research repo and is not shipped with this inference package.
The artifact in `artifacts/` was produced by this exact script.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
DATA = HERE / "inplay_dataset_v2.parquet"
ART = HERE / "artifacts"

SEED = 17
ALPHA = 0.61
KAPPA = 40.0
MATCH_FIRST_SERVER = 1

GBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.01,
    num_leaves=4,
    min_data_in_leaf=120,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=3,
    lambda_l2=8.0,
    verbose=-1,
    seed=SEED,
)
NUM_BOOST_ROUND = 200

FEATURES = [
    "p_markov_logit",
    "p_markov_bayes_logit",
    "sets_diff",
    "games_diff",
    "point_diff",
]


def main():
    ART.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. This file is produced by the upstream "
            f"build_inplay_v2.py — copy it next to train_full.py to retrain."
        )
    print(f"Loading {DATA.name}…")
    df = pl.read_parquet(DATA)
    print(f"  shape={df.shape}, matches={df['event_key'].n_unique()}")

    state_cols = ["event_key", "p1_sets", "p2_sets", "p1_games", "p2_games",
                  "p1_pt", "p2_pt", "server", "p1_total", "p2_total"]
    train_states = df.unique(subset=state_cols, keep="first")
    print(f"  unique (state x serve-stats) rows used for training: {train_states.shape[0]}")

    Xtr = train_states.select(FEATURES).to_pandas()
    ytr = train_states["y_p1"].to_numpy()
    init_score = train_states["p_markov_logit"].to_numpy()

    print(f"\nTraining LightGBM residual…")
    dtrain = lgb.Dataset(Xtr, label=ytr, init_score=init_score, free_raw_data=False)
    booster = lgb.train(GBM_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)

    raw_pred = booster.predict(Xtr, raw_score=True) + init_score
    p_full = 1 / (1 + np.exp(-raw_pred))
    p_markov_only = 1 / (1 + np.exp(-init_score))
    brier_in = float(np.mean((p_full - ytr) ** 2))
    brier_markov = float(np.mean((p_markov_only - ytr) ** 2))

    booster.save_model(str(ART / "gbm_residual.txt"))

    meta = {
        "model_version": "1.0.0",
        "alpha": ALPHA,
        "kappa": KAPPA,
        "match_first_server": MATCH_FIRST_SERVER,
        "features": FEATURES,
        "gbm_params": GBM_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "training_rows": int(len(ytr)),
        "training_matches": int(df["event_key"].n_unique()),
        "in_sample_brier_gbm": brier_in,
        "in_sample_brier_markov_only": brier_markov,
        "note": (
            "In-sample Brier is for sanity only — the honest OOS metrics come "
            "from leave-one-match-out CV in the upstream research repo. This "
            "artifact is fit on all data and should be used for production "
            "inference, not for evaluation."
        ),
    }
    (ART / "model_meta.json").write_text(json.dumps(meta, indent=2))

    fi = pd.DataFrame({
        "feature": FEATURES,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(ART / "feature_importance.csv", index=False)

    log_lines = [
        f"Trained on {len(ytr)} unique (state x stats) rows from "
        f"{df['event_key'].n_unique()} matches.",
        f"In-sample Brier (GBM+Markov): {brier_in:.4f}",
        f"In-sample Brier (Markov only): {brier_markov:.4f}",
        f"Residual gain (in-sample): {brier_markov - brier_in:+.4f}",
        "",
        "Feature importance (mean gain):",
        fi.to_string(index=False),
    ]
    (ART / "training_log.txt").write_text("\n".join(log_lines))

    print("\n=== Training complete ===")
    for line in log_lines:
        print(line)
    print(f"\nArtifacts written to: {ART}/")


if __name__ == "__main__":
    main()
