"""Example usage of the trained TennisModel.

Run `python ml/train_full.py` once first to produce the artifacts, then:

    python ml/inference_example.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TennisModel

HERE = Path(__file__).resolve().parent
m = TennisModel.load(HERE / "artifacts")


def show(label: str, out: dict):
    print(f"\n=== {label} ===")
    print(f"  P(player 1 wins) ........ {out['p1_win_prob']:.4f}")
    print(f"  Markov (no Bayes) ....... {out['p_markov']:.4f}")
    print(f"  Markov + Bayes update ... {out['p_markov_bayes']:.4f}")
    print(f"  GBM logit adjustment .... {out['gbm_logit_adjustment']:+.4f}")
    print(f"  Prior serve probs ....... p1={out['p1_serve_prior']:.4f}, "
          f"p2={out['p2_serve_prior']:.4f}")
    print(f"  Posterior serve probs ... p1={out['p1_serve_post']:.4f}, "
          f"p2={out['p2_serve_post']:.4f}")


# 1) Pre-match — only the prior is needed.
show("Pre-match, prior 0.65, no live state",
     m.predict(prior_p1=0.65))

# 2) Mid-match without live stats — Bayes update is a no-op (counts are 0).
show("Mid-match, prior 0.65, p1 leads 1-0 sets / 3-2 games / 30-15, p1 serving",
     m.predict(prior_p1=0.65,
               p1_sets=1, p2_sets=0,
               p1_games=3, p2_games=2,
               p1_pt=2, p2_pt=1,
               server=1))

# 3) Same state, but with live service-point stats — Bayes update kicks in.
show("...and now with live service counts (p1: 24/35, p2: 18/29)",
     m.predict(prior_p1=0.65,
               p1_sets=1, p2_sets=0,
               p1_games=3, p2_games=2,
               p1_pt=2, p2_pt=1,
               server=1,
               p1_serve_won=24, p1_serve_total=35,
               p2_serve_won=18, p2_serve_total=29))

# 4) Upset in progress: heavy underdog (prior 0.20) is up a break in set 2 having lost set 1.
show("Upset scenario: prior 0.20, lost set 1, up a break in set 2",
     m.predict(prior_p1=0.20,
               p1_sets=0, p2_sets=1,
               p1_games=3, p2_games=1,
               p1_pt=0, p2_pt=0,
               server=2,
               p1_serve_won=18, p1_serve_total=27,
               p2_serve_won=20, p2_serve_total=34))

# 5) Match point: p1 serving for the match at 5-3, 40-15 in deciding set.
show("Match point: prior 0.50, level on sets, 5-3 third set, 40-15 p1 serve",
     m.predict(prior_p1=0.50,
               p1_sets=1, p2_sets=1,
               p1_games=5, p2_games=3,
               p1_pt=3, p2_pt=1,
               server=1))
