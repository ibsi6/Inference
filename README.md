# In-play Tennis Win-Probability Model — Live-Odds-Integration Markov

A single closed-form Markov chain, anchored at a pre-match bookmaker prior,
re-calibrated at every tick by an L2-regularised 1-D fit against a live
market mid (Kalshi or any other source), and decomposed into **five
markets** that are self-consistent by construction:

1. **Main**             — `P(player 1 wins match)`
2. **Set 1 winner**     — `P(player 1 wins set 1)`
3. **Set 2 winner**     — `P(player 1 wins set 2)`
4. **Exact match score** — `p1 2-0`, `p1 2-1`, `p2 2-0`, `p2 2-1`
5. **Total games O/U N** — `P(total games at match end > N)`  (default ATP line: N = 20)

No machine learning, no offline training, no learned residual. Stages 1–5
are deterministic closed-form expressions of the calibrated serve
probabilities (or a finite BFS over the game tree, for total games).

---

## 1.  How it works

### 1.1  The state space

Best-of-3 tennis is well represented by a Markov chain on `(p1_sets,
p2_sets, p1_games, p2_games, p1_pt, p2_pt, server)`, where every transition
is governed by two scalars:

- `p1_serve` — probability player 1 wins a point on her own serve
- `p2_serve` — same for player 2

The closed-form `p_win_match(state, p1_serve, p2_serve)` from
`tennis_markov.py` is the O'Malley (2008) recursion. From the same chain
we can read off `P(p1 wins this set)`, `P(p1 wins a future set)`, and
combinations of these for the exact-score and total-games markets.

### 1.2  Pre-match calibration

Given the de-vigged bookmaker prior `prior_p1`, we parameterise

```
p1_serve = σ(α + β),   p2_serve = σ(α − β)
```

`α` is the average serve strength (fixed at 0.61 ≈ logit(0.65), the
long-run pro-tennis figure across surfaces). We 1-D bisect for `β_prior`
such that the closed-form Markov, evaluated at the pre-match state
(0–0–0–0), reproduces `prior_p1`.

### 1.3  Live calibration (the "live odds integration")

At every tick `t` we re-fit `β` against the current market mid `m(t)`
(Kalshi yes-mid for the match-winner contract is the natural choice), with
an L2 anchor toward `β_prior`:

```
β(t) = argmin  (m(t) − p_win_match(state(t), σ(α+β), σ(α−β)))²
              + λ · (β − β_prior)²
```

`λ` is the only knob:

- `λ = ∞` — no live update; the model collapses to "prior solve, then drift
  with the score state". This is the baseline.
- `λ = 0` — full market-following; `β(t)` exactly reproduces `m(t)` at the
  current state. Useful when you trust the market and just want a
  self-consistent decomposition.
- `λ = 0.5` (default) — honest compromise. Anchored enough that single-tick
  noise doesn't whipsaw the params, loose enough that real market moves
  are absorbed.

α is held fixed because a single market mid is one equation in two
unknowns; you'd need at least one *additional* market (e.g. a set-winner
mid) to identify both. See §5 for how to do that.

### 1.4  Decomposition into the five markets

The same calibrated `(p1_serve, p2_serve)` is fed into:

- **Main**         — `p_win_match(state, p1s, p2s)` directly.
- **Set N winner** — `_p_set_from_full` if the set is in progress;
                     `_p_set_from_games` from fresh 0-0 with the correct
                     set-first-server if the set is in the future; or the
                     known {0, 1} indicator if the set is decided.
- **Exact score**  — sum-of-products over the per-set winner probabilities
                     (the constant-serve assumption makes this a closed-form
                     computation across four cases — see
                     `LiveMarkovModel.p_exact_score`).
- **Total games**  — for each possible terminal set score (6-0, 6-1, 6-2,
                     6-3, 6-4, 7-5, 7-6 and their mirrors), the Markov
                     chain gives a probability via BFS over the game tree.
                     Convolve across the up-to-three sets to get the
                     distribution over total match games; sum the tail
                     beyond the O/U line.

All decompositions are closed-form (no Monte Carlo).

---

## 2.  Quick start

```python
from model import LiveMarkovModel

m = LiveMarkovModel(prior_p1=0.65, match_first_server=1)

# Pre-match — only the prior is in play.
m.update(games_in_completed_sets=0)
m.p_match                     # ≈ 0.65
m.p_set_winner(1)             # ≈ 0.60
m.p_set_winner(2)
m.p_exact_score()             # {'p1_2_0': 0.36, 'p1_2_1': 0.29, ...}
m.p_total_games_over(20)      # ≈ 0.74

# Mid-match: set 1 won 6-4 by p1, currently 3-2 30-15 in set 2 with p1 serving,
# and Kalshi yes-mid is 0.81.
m.update(p1_sets=1, p2_sets=0,
         p1_games=3, p2_games=2,
         p1_pt=2, p2_pt=1, server=1,
         kalshi_mid=0.81, lam=0.5,
         games_in_completed_sets=10)   # set 1 was 6-4 → 10 games

m.p_match                     # ≈ 0.89 (between Kalshi 0.81 and the prior baseline)
m.p_set_winner(1)             # 1.0 — auto-tracked from the (0,0) → (1,0) transition
m.p_set_winner(2)             # ≈ 0.80
m.p_exact_score()             # heavy on p1_2_0
m.p_total_games_over(20)      # ≈ 0.32
```

`m.summary()` returns all five market probabilities plus the calibration
internals (`beta`, `beta_prior`, `p1_serve_prob`, `p2_serve_prob`) in a
single dict.

---

## 3.  Inputs to `update(...)`

| Argument | Type | Description |
|---|---|---|
| `p1_sets`, `p2_sets` | int | Sets won so far. Match ends at 2. |
| `p1_games`, `p2_games` | int | Games in the current set (0..7). |
| `p1_pt`, `p2_pt` | int | Points in the current game. 0/1/2/3 = 0/15/30/40; deuce/advantage when both ≥ 3. At 6-6 in games these are tiebreak points (0..). |
| `server` | int | Player serving the current game (1 or 2). |
| `kalshi_mid` | float, optional | Live market mid for P(p1 wins match). If `None`, `β` stays at `β_prior` (no live update). |
| `lam` | float | L2 strength. Default `0.5`. `inf` = no live update; `0` = exact fit. |
| `games_in_completed_sets` | int, optional | Total games played in already-finished sets (10 if set 1 was 6-4, 22 if it was 6-4 + 4-6 = 10+10, etc.). REQUIRED for `p_total_games_over/under(...)` queries; not needed for the other four markets. |
| `set_winners` | dict, optional | `{set_idx: 1 or 2}` for past sets. Usually auto-tracked across update() calls; pass explicitly if you skipped ticks or want to bootstrap. |

The constructor takes only `prior_p1`, `match_first_server` (1 or 2; who
serves the very first game of the match), and `alpha` (fixed at 0.61 by
default — don't tune this unless you have very compelling data).

---

## 4.  Caveats

### 4.1  Constant-serve assumption

The chain assumes `(p1_serve, p2_serve)` are constant across the match.
That under-prices in-set / between-set momentum effects that the market
typically does price. Empirically the residual on exact-score is small
(~3–5pp per outcome) but systematic: the constant-serve Markov over-prices
3-set outcomes and under-prices 2-set sweeps relative to consensus books.
**Don't fit the residual away — log it.** The gap *is* the tradeable
signal.

### 4.2  α is fixed

With only one live equation (`m(t)`), `α` and `β` are not jointly
identified. The model fixes `α` at the long-run pro average and updates
only `β`. If you have a second live market mid (e.g. set-1-winner Kalshi),
you can extend the fit to two unknowns and let `α` move — see §5.

### 4.3  Kalshi side markets

This implementation is designed to ingest a single live mid (typically
the match-winner). Side-market mids (KXATPSETWINNER, KXATPEXACTMATCH,
KXATPTOTALSETS, KXATPGAMETOTAL, …) are *outputs* of the model — they're
what you decompose to, and what you'd trade against the market. To use
them as additional calibration *inputs* you'd extend `_fit_beta` to
multiple equations (e.g. weighted least squares across markets).

### 4.4  Total games requires `games_in_completed_sets`

The chain doesn't track historical per-set game counts. Pass
`games_in_completed_sets` in `update(...)` whenever you want to query
total games. Pre-match it's 0; after a 6-4 set 1 it's 10; etc.

### 4.5  Best-of-3 only

The total-games closed form and the exact-score decomposition are
hard-coded for best-of-3. Slam main draws (best-of-5 men's) need a
one-line change in `tennis_markov.py:244` and a corresponding extension
of `_total_games_distribution`.

---

## 5.  Extending to multi-market calibration

The single-equation fit `(m_match − markov_match)² + λ(β − β_prior)²`
generalises naturally to any number of live mids:

```
β(t) = argmin  Σᵢ wᵢ · (mᵢ(t) − markov_marketᵢ(state(t), β))²
              + λ · (β − β_prior)²
```

where `markov_marketᵢ` is the model's prediction for market *i* (set
winner, exact score sub-outcome, ...). With ≥ 2 independent equations you
can also let `α` move (2 unknowns, ≥ 2 constraints). This is what makes
joint calibration over correlated Kalshi tennis markets attractive:

- the params get properly identified (no fixed α assumption)
- residuals across markets indicate which one the market is mis-pricing
  relative to a consistent constant-serve view

The current implementation keeps it at one market (match winner) for
simplicity. Extension is a ~20-line patch in `_fit_beta`.

---

## 6.  Files

```
Inference/
├── README.md          (this file)
├── model.py           LiveMarkovModel class + all five decompositions.
├── tennis_markov.py   Closed-form Markov chain (O'Malley 2008). No learned params.
├── example.py         Six worked scenarios.
└── requirements.txt   numpy, scipy.
```

No artifacts directory, no model weights. Everything is closed-form from
two scalars (`prior_p1`, `match_first_server`) plus a tunable `lam`.

---

## 7.  Performance

Cold start on a match: ~10 ms (Markov memoisation warm-up + β-prior
bisection). Subsequent in-play ticks are sub-millisecond per call because
`tennis_markov`'s `@lru_cache` makes the Markov closed-form essentially
free once the state has been seen, and `_set_score_dist_fresh` is also
cached per `(p1s, p2s, set_first_server)`.

The L2-regularised β fit uses `scipy.optimize.minimize_scalar` with
bounded Brent — ~20–30 function evaluations per call, each evaluating
the closed-form Markov once. Sub-millisecond.

---

## 8.  Reference

- O'Malley, A. J. (2008). *Probability formulas for a tennis match.* JQAS 4(2).
- Klaassen, F. & Magnus, J. (2014). *Analyzing Wimbledon: The Power of
  Statistics.* Oxford University Press.

The regularised live-fit + multi-market decomposition idea is the
contribution of this implementation; the underlying Markov chain is the
standard one.
