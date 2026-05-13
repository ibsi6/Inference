# In-play Tennis Win-Probability Model — Inference Guide

This document explains how to use the trained model in production. It covers the
files involved, the exact inputs the model needs, the outputs it returns, how
to derive the pre-match prior from bookmaker odds, and an honest list of the
caveats you should know before deploying it.

---

## 1. What the model is

A four-stage pipeline that turns a pre-match bookmaker prior + live match state
+ (optionally) live service-point stats into a calibrated probability that
player 1 wins.

```
   prior_p1  (from bookmaker odds, de-vigged)
        │
        ▼
   ┌─────────────────────────────────────────────┐
   │ STAGE 1 — invert Markov for prior serve probs │
   │   (p1_serve_prior, p2_serve_prior) = solve_β  │
   │   so that closed-form Markov(p1s, p2s) = prior│
   └─────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────┐
   │ STAGE 2 — Bayesian update of serve probs     │
   │   posterior = (κ·prior + won) / (κ + total)   │
   │   κ = 40 equivalent points (a soft prior)     │
   │   Skip this stage if you don't have live      │
   │   service-point counts.                       │
   └─────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────┐
   │ STAGE 3 — Markov closed-form re-evaluation    │
   │   p_markov       using prior serve probs      │
   │   p_markov_bayes using posterior serve probs  │
   │   (current sets, games, points, server)       │
   └─────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────┐
   │ STAGE 4 — LightGBM residual                  │
   │   init_score = logit(p_markov)               │
   │   features: markov logits + score-diffs      │
   │   final_logit = init_score + booster(X)      │
   │   p1_win_prob = sigmoid(final_logit)         │
   └─────────────────────────────────────────────┘
```

Stages 1–3 are deterministic and have no learned parameters — only stage 4
(the LightGBM residual) is a trained model.

---

## 2. Quick start

```python
from model import TennisModel

m = TennisModel.load("artifacts")

out = m.predict(
    prior_p1=0.65,
    p1_sets=1, p2_sets=0,
    p1_games=3, p2_games=2,
    p1_pt=2, p2_pt=1,        # 30-15 in points
    server=1,
    # optional — leave these out (or at 0) for a pre-Bayes prediction
    p1_serve_won=24, p1_serve_total=35,
    p2_serve_won=18, p2_serve_total=29,
)
print(out["p1_win_prob"])    # → 0.78xx
```

A first call on a new match incurs ~10ms (Markov cache warm-up). Subsequent
calls on the same match are <0.5 ms each.

---

## 3. Files involved

```
Inference/
├── README.md                This document.
├── model.py                 TennisModel class — load() and predict().
├── tennis_markov.py         Closed-form Markov chain + prior inversion (no learned params).
├── inference_example.py     Five worked examples.
├── train_full.py            (reference) script used to fit the residual on the
│                            full in-play dataset; the dataset itself lives in
│                            the upstream research repo and isn't shipped here.
├── requirements.txt         numpy, pandas, lightgbm
└── artifacts/               Pre-trained model produced by train_full.py:
    ├── gbm_residual.txt         LightGBM booster (portable text format).
    ├── model_meta.json          α, κ, feature list, hyperparameters.
    ├── feature_importance.csv   GBM gain/split per feature.
    └── training_log.txt         Human-readable training summary.
```

To use the model in another project you need exactly these four files:
`model.py`, `tennis_markov.py`, `artifacts/gbm_residual.txt`,
`artifacts/model_meta.json`. Python deps: `numpy`, `pandas`, `lightgbm`.

---

## 4. Inputs to `predict(...)`

### 4.1 Required

| Argument | Type | Description |
|---|---|---|
| `prior_p1` | `float` ∈ (0,1) | Pre-match P(player 1 wins) derived from bookmaker odds. **Must be de-vigged** (the two implied probabilities should sum to 1). See §6 for the recipe. Values are clipped to [1e-4, 1−1e-4]. |

### 4.2 Score state (default = pre-match 0–0)

Pass these every tick. Player 1 is whichever player you used to anchor
`prior_p1` and whichever player corresponds to `first_player_key` in your
odds source — keep this consistent across the whole pipeline.

| Argument | Type | Description |
|---|---|---|
| `p1_sets`, `p2_sets` | `int` | Sets won so far. The match is terminal at 2 in best-of-3. |
| `p1_games`, `p2_games` | `int` | Games in the current set (0..7). Once a player reaches 6 with a 2-game lead, or 7-5, or wins a 6-6 tiebreak, that set is closed and these reset. |
| `p1_pt`, `p2_pt` | `int` | Points in the current game. `0/1/2/3` = `0/15/30/40`. If both ≥ 3 and equal = deuce; if both ≥ 3 and differ by 1 = advantage. At 6-6 in games these are tiebreak points (`0..7+`). |
| `server` | `int ∈ {1, 2}` | Who serves the current game. |
| `match_first_server` | `int ∈ {1, 2}` or `None` | Who served the very first game of the match. Defaults to 1 (the value stored in `model_meta.json`). Set this if you know otherwise. |

### 4.3 Live service-point stats (optional — pass to enable Bayes update)

| Argument | Type | Description |
|---|---|---|
| `p1_serve_won` | `int` | Service points won by player 1 so far in the match. |
| `p1_serve_total` | `int` | Service points played by player 1 so far. |
| `p2_serve_won` | `int` | Same for player 2. |
| `p2_serve_total` | `int` | Same for player 2. |

If you don't have these, leave them at 0 — the Bayes posterior then collapses
back to the prior and `p_markov_bayes` becomes equal to `p_markov`.

---

## 5. Outputs

`predict(...)` returns a dict:

| Key | Type | Description |
|---|---|---|
| `p1_win_prob` | `float` | **The headline.** Final calibrated P(player 1 wins) — Markov + Bayes + GBM residual. Use this as your fair value. |
| `p_markov` | `float` | Markov closed-form using prior-derived serve probs only. Model-free, just a function of (prior, score state). |
| `p_markov_bayes` | `float` | Same as `p_markov` but with Bayesian-updated serve probs (so it reacts to how the players are actually serving). |
| `p1_serve_prior` | `float` | Prior point-on-serve win rate for player 1, solved from the prior. |
| `p2_serve_prior` | `float` | Same for player 2. |
| `p1_serve_post` | `float` | Posterior point-on-serve rate for player 1 given live counts. Equals prior if no counts were passed. |
| `p2_serve_post` | `float` | Same for player 2. |
| `gbm_logit_adjustment` | `float` | The residual the GBM adds to the Markov logit, in nats. Useful for debugging — a value near 0 means the GBM agrees with the Markov closed-form on this state. |

If you want a quote without the GBM layer (see §8 on why this is sometimes
preferable), use `out["p_markov"]` or `out["p_markov_bayes"]` directly.

---

## 6. Deriving `prior_p1` from bookmaker odds

The model expects a de-vigged consensus prior. The recipe used in this project:

1. Collect home/away decimal odds from N bookmakers at match start.
2. For each book, implied probabilities are `q_home = 1/o_home` and
   `q_away = 1/o_away`. These sum to slightly more than 1 (the overround / vig).
3. **De-vig** the pair: `p_home = q_home / (q_home + q_away)`. Now they sum to 1.
4. Take the **median** of `p_home` across the N books — this is your `prior_p1`
   (assuming player 1 is "home"; if not, take 1 − median).

That's it. There's no need to track sharper books separately for this model;
the median across ≥ 6 books is more stable than any single book.

A two-book quick recipe in Python:

```python
def devig_prior(odds_home: float, odds_away: float) -> float:
    qh, qa = 1/odds_home, 1/odds_away
    return qh / (qh + qa)

prior_p1 = devig_prior(1.55, 2.40)  # → 0.6076 if "player 1" is home
```

---

## 7. Live data flow

You typically call `predict(...)` at every snapshot you get (e.g. once per
second). The Markov solver's `@lru_cache` means after the first call for a
given (prior, match_first_server) the serve-prob inversion is free, and after
the first call for each state the closed-form Markov is free too. So in steady
state, the per-tick cost is the LightGBM `predict(X)` call — single-row, ~0.1 ms.

Things you should keep consistent across ticks for a given match:

- **The identity of player 1.** Once you anchor `prior_p1` on a specific player,
  every subsequent tick must report that same player's score, that same player's
  serve stats, and `server=1` when that player is serving.
- **`match_first_server`.** Don't change this mid-match — it affects which player
  serves which set's first game and the tiebreak server order.

Things that genuinely update tick-to-tick:

- **The score state** (sets/games/points/server) — read from your live feed.
- **The live service-point counts** — read from your live feed.

---

## 8. Honest caveats — read this before deploying

### 8.1 Trained on 6 matches.

The LightGBM residual was fit on ~1,000 unique states from **6 matches** (ATP
Rome 2026, single calendar day, clay). Of those, **5 of 6 matches were won by
player 1** (anchored to the home/first player). The GBM learns that fact:
on most inputs it adds a positive offset to the Markov logit (the
`gbm_logit_adjustment` field).

This is *correct in-sample but optimistic out-of-sample*. The honest OOS
metrics (leave-one-match-out, see `results_inplay/metrics_pooled.csv`) are:

| Model | Pooled Brier | Match-mean Brier |
|---|---|---|
| LightGBM residual | 0.2569 | 0.1369 |
| Markov closed-form (no GBM) | 0.2785 | 0.1733 |
| Kalshi mid (market) | 0.2990 | 0.1892 |

These are what to trust. The in-sample 0.0487 reported in `training_log.txt`
is for debugging only — it just tells you the booster fit fine.

### 8.2 What to do if you don't trust the GBM residual.

You have three reasonable options:

1. **Use the headline `p1_win_prob`** if you accept the panel bias.
2. **Use `p_markov_bayes`** (skips the GBM residual; just Markov with the
   Bayesian-updated serve probs). This is data-free in the structural sense —
   it has no fit parameters except κ — and is well-calibrated for any new
   match.
3. **Use `p_markov`** (no Bayes either). This is the most conservative output:
   a pure function of (prior, current state), exactly the closed-form Markov
   tennis chain from O'Malley (2008).

All three are returned by every call to `predict(...)`, so you can switch by
just picking a different key from the output dict.

### 8.3 Retrain when you have more matches.

`build_inplay_v2.py` + `train_full.py` is the full pipeline; rerun both once
you have ≥50 matches across multiple tournaments/surfaces. The hyper-
parameters in `train_full.py` are conservative on purpose so the residual
captures genuine score-state effects rather than panel-specific outcomes.

### 8.4 Best-of-3 only.

The Markov closed-form is hard-coded for best-of-3. Slams (best-of-5 on the
men's side) need a 1-line change in `tennis_markov.py:244` (`p1_sets >= 2`
→ `>= 3`). Until that's done, *do not call this model on slam main draws*.

### 8.5 What this model is NOT.

- It is **not a market-following model.** Kalshi mid is deliberately excluded
  from the feature set so the model is independent of the market. If you want
  best-possible probability rather than a market-blind model, add Kalshi mid
  (or any other market) as a feature in `build_inplay_v2.py` and retrain.
- It does **not handle retirement risk**, weather, court-side stats (aces,
  break points outside the score), or momentum beyond what the score state +
  serve-stat update encodes.
- It does **not output a confidence interval.** A bootstrap interval would
  cost ~50× more compute per call and isn't currently exposed.

---

## 9. Reproducing the artifact from scratch

This repo ships the pre-trained artifact (`artifacts/gbm_residual.txt`). If you
want to retrain from scratch you need the upstream in-play dataset
(`inplay_dataset_v2.parquet`) which is produced by the research repo's
`build_inplay.py` + `build_inplay_v2.py`. With that file in place:

```bash
pip install -r requirements.txt
python train_full.py            # writes artifacts/gbm_residual.txt
python inference_example.py     # smoke-test inference on the trained artifact
```

Seed is 17 throughout. Retraining is fully deterministic given the same input
parquet.

---

## 10. Example: backtest-style loop on a stream

```python
from model import TennisModel

m = TennisModel.load("artifacts")

# `feed` is whatever live-stream object you have. Each tick must give you the
# score state + accumulated serve counts.
for tick in feed:
    out = m.predict(
        prior_p1=tick.prior_p1,
        p1_sets=tick.p1_sets, p2_sets=tick.p2_sets,
        p1_games=tick.p1_games, p2_games=tick.p2_games,
        p1_pt=tick.p1_pt, p2_pt=tick.p2_pt,
        server=tick.server,
        p1_serve_won=tick.p1_serve_won,
        p1_serve_total=tick.p1_serve_total,
        p2_serve_won=tick.p2_serve_won,
        p2_serve_total=tick.p2_serve_total,
    )
    fair = out["p1_win_prob"]
    # Compare against the live Kalshi best bid/ask:
    if fair > tick.ask + 0.02:  publish_buy(qty=10, price=tick.ask)
    if fair < tick.bid - 0.02:  publish_sell(qty=10, price=tick.bid)
```

This is exactly the rule used in `backtest.py`. Position sizing,
cooldown, max-spread filter and bounded exposure are all defined in that file.

---

## 11. Reference

- O'Malley, A. J. (2008). *Probability formulas for a tennis match.* JQAS 4(2).
- Klaassen, F. & Magnus, J. (2014). *Analyzing Wimbledon: The Power of
  Statistics.* Oxford University Press.
- The full research write-up is in `results_inplay/report.pdf`.
