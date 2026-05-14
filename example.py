"""Worked examples for LiveMarkovModel.

Run:
    python example.py

Each scenario shows the same calibrated chain decomposed into the five
markets: match winner, set 1 winner, set 2 winner, exact match score, and
total-games over/under N.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import LiveMarkovModel


def show(label: str, m: LiveMarkovModel) -> None:
    s = m.summary()
    print(f"\n=== {label} ===")
    print(f"  β prior / β fit   : {s['beta_prior']:+.4f}  →  {s['beta']:+.4f}")
    print(f"  p1_serve / p2_serve : {s['p1_serve_prob']:.4f} / {s['p2_serve_prob']:.4f}")
    print(f"  P(match)          : {s['p_match']:.4f}")
    print(f"  P(set 1)          : {s['p_set1']:.4f}")
    print(f"  P(set 2)          : {s['p_set2']:.4f}")
    es = s["p_exact_score"]
    print(f"  Exact score       : p1 2-0={es['p1_2_0']:.4f}  "
          f"p1 2-1={es['p1_2_1']:.4f}  "
          f"p2 2-0={es['p2_2_0']:.4f}  "
          f"p2 2-1={es['p2_2_1']:.4f}")
    if "p_total_games_over_20" in s:
        print(f"  P(total > 20)     : {s['p_total_games_over_20']:.4f}")
        print(f"  P(total > 21)     : {s['p_total_games_over_21']:.4f}")
        print(f"  P(total > 22)     : {s['p_total_games_over_22']:.4f}")


# --------------------------------------------------------------------------
# 1. Pre-match, moderate favorite (prior 0.65).
# --------------------------------------------------------------------------
m = LiveMarkovModel(prior_p1=0.65, match_first_server=1)
m.update(games_in_completed_sets=0)   # pre-match → all state defaults to 0
show("Pre-match, prior 0.65", m)


# --------------------------------------------------------------------------
# 2. Mid-match WITHOUT live market mid (just the score state).
#    Set 1 won by p1 (6-4 = 10 games), now in set 2 leading 3-2, 30-15, p1 serving.
# --------------------------------------------------------------------------
m = LiveMarkovModel(prior_p1=0.65, match_first_server=1)
m.update(p1_sets=1, p2_sets=0,
         p1_games=3, p2_games=2,
         p1_pt=2, p2_pt=1, server=1,
         games_in_completed_sets=10,
         set_winners={1: 1})           # we missed the set-1 transition, so tell the model
show("Mid-match, no live mid (β stays at prior)", m)


# --------------------------------------------------------------------------
# 3. Same state, this time anchored to a live Kalshi mid of 0.81.
# --------------------------------------------------------------------------
m = LiveMarkovModel(prior_p1=0.65, match_first_server=1)
m.update(p1_sets=1, p2_sets=0,
         p1_games=3, p2_games=2,
         p1_pt=2, p2_pt=1, server=1,
         kalshi_mid=0.81, lam=0.5,
         games_in_completed_sets=10,
         set_winners={1: 1})
show("Mid-match, Kalshi mid 0.81, λ=0.5", m)


# --------------------------------------------------------------------------
# 4. Upset in progress: heavy underdog (prior 0.20) lost set 1 (3-6) but is
#    up a break in set 2.
# --------------------------------------------------------------------------
m = LiveMarkovModel(prior_p1=0.20, match_first_server=1)
m.update(p1_sets=0, p2_sets=1,
         p1_games=3, p2_games=1,
         p1_pt=0, p2_pt=0, server=2,
         kalshi_mid=0.35, lam=0.5,
         games_in_completed_sets=9,    # 3-6 = 9 games
         set_winners={1: 2})
show("Upset in progress: prior 0.20, lost set 1 3-6, up 3-1 in set 2", m)


# --------------------------------------------------------------------------
# 5. Match point: even match (prior 0.50), one set each, p1 serving for
#    the match at 5-3, 40-15 in the deciding set.
# --------------------------------------------------------------------------
m = LiveMarkovModel(prior_p1=0.50, match_first_server=1)
m.update(p1_sets=1, p2_sets=1,
         p1_games=5, p2_games=3,
         p1_pt=3, p2_pt=1, server=1,
         games_in_completed_sets=22,   # e.g. 6-4 then 4-6 = 10+10
         set_winners={1: 1, 2: 2})
show("Match point: prior 0.50, 1-1 sets, 5-3 / 40-15 p1 serving", m)


# --------------------------------------------------------------------------
# 6. Sweep of total-games O/U lines at the same state.
# --------------------------------------------------------------------------
print("\n=== Total-games O/U sweep — pre-match prior 0.65 (no live mid) ===")
m = LiveMarkovModel(prior_p1=0.65, match_first_server=1)
m.update(games_in_completed_sets=0)
for N in (18, 19, 20, 21, 22, 23, 24):
    print(f"  P(total > {N}) = {m.p_total_games_over(N):.4f}    "
          f"P(total < {N}) = {m.p_total_games_under(N):.4f}")
