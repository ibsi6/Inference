"""
Production inference wrapper for the in-play tennis win-probability model.

Quick start
-----------
    from model import TennisModel
    m = TennisModel.load("ml/artifacts")
    out = m.predict(prior_p1=0.65,
                    p1_sets=1, p2_sets=0,
                    p1_games=3, p2_games=2,
                    p1_pt=2, p2_pt=1,
                    server=1,
                    p1_serve_won=24, p1_serve_total=35,
                    p2_serve_won=18, p2_serve_total=29)
    out["p1_win_prob"]  # 0.78xx — your fair price for "Player 1 wins"

The model is the four-stage pipeline described in MODEL_README.md:

    prior_p1
        |
        v
    solve_serve_for_prior   (1-D root over beta) -> (p1_serve_prior, p2_serve_prior)
        |
        v
    Beta-Binomial update    (kappa equivalent points, default 40) -> (p1_serve_post, p2_serve_post)
        |
        v
    Markov closed-form      (current sets/games/points/server)    -> p_markov, p_markov_bayes
        |
        v
    LightGBM residual       init_score = logit(p_markov) + booster.predict(...) -> p1_win_prob

All four stages run in <1 ms per call on a modern CPU; the Markov solver is
memoized so subsequent calls on the same match are nearly free.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tennis_markov import solve_serve_for_prior, MatchState, p_win_match


def _logit_clip(p: float, eps: float = 1e-9) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TennisModel:
    """In-play tennis win-probability model.

    Load with `TennisModel.load(artifact_dir)` and call `predict(...)` for each
    snapshot. The artifact_dir must contain `gbm_residual.txt` (LightGBM booster
    in text format) and `model_meta.json` (hyperparameters).
    """

    DEFAULT_FEATURES = [
        "p_markov_logit",
        "p_markov_bayes_logit",
        "sets_diff",
        "games_diff",
        "point_diff",
    ]

    def __init__(self, booster: lgb.Booster, meta: dict):
        self.booster = booster
        self.meta = meta
        self.kappa = float(meta.get("kappa", 40.0))
        self.alpha = float(meta.get("alpha", 0.61))
        self.match_first_server = int(meta.get("match_first_server", 1))
        self.features = list(meta.get("features", self.DEFAULT_FEATURES))

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "TennisModel":
        artifact_dir = Path(artifact_dir)
        booster_path = artifact_dir / "gbm_residual.txt"
        meta_path = artifact_dir / "model_meta.json"
        if not booster_path.exists():
            raise FileNotFoundError(f"Missing booster: {booster_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta: {meta_path}")
        booster = lgb.Booster(model_file=str(booster_path))
        meta = json.loads(meta_path.read_text())
        return cls(booster, meta)

    @staticmethod
    def _bayes_serve(prior_p_serve: float, won: int, total: int,
                     kappa: float) -> float:
        if total <= 0:
            return prior_p_serve
        return (kappa * prior_p_serve + won) / (kappa + total)

    def predict(
        self,
        prior_p1: float,
        p1_sets: int = 0,
        p2_sets: int = 0,
        p1_games: int = 0,
        p2_games: int = 0,
        p1_pt: int = 0,
        p2_pt: int = 0,
        server: int = 1,
        match_first_server: int | None = None,
        p1_serve_won: int = 0,
        p1_serve_total: int = 0,
        p2_serve_won: int = 0,
        p2_serve_total: int = 0,
    ) -> dict:
        """Compute P(player 1 wins) and intermediate quantities.

        Required
        --------
        prior_p1 : float
            Pre-match P(player 1 wins) derived from bookmaker odds (de-vigged).
            Should be in (0, 1). Values are clipped to [1e-4, 1-1e-4].

        Score state (default = pre-match 0-0)
        -------------------------------------
        p1_sets, p2_sets : int                Sets won so far (terminal at 2 for best-of-3).
        p1_games, p2_games : int              Games in the current set (0..7).
        p1_pt, p2_pt : int                    Points in the current game. 0..3 = 0/15/30/40;
                                              if both >=3, equal = deuce, diff by 1 = advantage.
                                              At 6-6 in games these are tiebreak points (0..7+).
        server : int                          1 if player 1 serves the current game, 2 otherwise.
        match_first_server : int | None       1 or 2 — who served the very first game of the match.
                                              Defaults to the value baked into meta (1).

        Live service stats (optional — leave at 0 to skip the Bayesian update)
        ---------------------------------------------------------------------
        p1_serve_won, p1_serve_total : int    Service points won and played by player 1
                                              so far in the match.
        p2_serve_won, p2_serve_total : int    Same for player 2.

        Returns
        -------
        dict with keys:
            p1_win_prob              float    The headline fair value in [0, 1].
            p_markov                 float    Markov closed-form using prior-derived serves only.
            p_markov_bayes           float    Markov with Bayesian-updated serve probs.
            p1_serve_prior           float    Prior point-on-serve win prob for player 1.
            p2_serve_prior           float    Prior point-on-serve win prob for player 2.
            p1_serve_post            float    Posterior serve prob given live counts.
            p2_serve_post            float    Posterior serve prob given live counts.
            gbm_logit_adjustment     float    Residual added to the Markov logit (in nats).
        """
        prior_p1 = float(np.clip(prior_p1, 1e-4, 1 - 1e-4))
        mfs = int(match_first_server) if match_first_server is not None else self.match_first_server
        if server not in (1, 2):
            raise ValueError(f"server must be 1 or 2, got {server}")
        if mfs not in (1, 2):
            raise ValueError(f"match_first_server must be 1 or 2, got {mfs}")

        # 1) Prior serve probs from the prior win prob
        p1_prior_serve, p2_prior_serve = solve_serve_for_prior(
            prior_p1, match_first_server=mfs, alpha=self.alpha,
        )

        # 2) Bayes update
        p1_post = self._bayes_serve(p1_prior_serve, p1_serve_won, p1_serve_total, self.kappa)
        p2_post = self._bayes_serve(p2_prior_serve, p2_serve_won, p2_serve_total, self.kappa)

        # 3) Markov closed-form for both serve estimates
        state = MatchState(
            p1_sets=int(p1_sets), p2_sets=int(p2_sets),
            p1_games=int(p1_games), p2_games=int(p2_games),
            p1_pt=int(p1_pt), p2_pt=int(p2_pt),
            server=int(server), match_first_server=mfs,
        )
        p_markov = float(p_win_match(state, p1_prior_serve, p2_prior_serve))
        # The training pipeline rounds posterior serve probs to 4dp for the Bayes Markov call
        # (see build_inplay_v2.py:133). We match that here so inference == training.
        p1_post_q, p2_post_q = round(p1_post, 4), round(p2_post, 4)
        p_markov_bayes = float(p_win_match(state, p1_post_q, p2_post_q))

        p_markov_logit = _logit_clip(p_markov)
        p_markov_bayes_logit = _logit_clip(p_markov_bayes)

        # 4) GBM residual on logit space, init_score = Markov logit (re-added at predict)
        sets_diff = int(p1_sets) - int(p2_sets)
        games_diff = int(p1_games) - int(p2_games)
        point_diff = int(p1_pt) - int(p2_pt)
        feat_values = {
            "p_markov_logit": p_markov_logit,
            "p_markov_bayes_logit": p_markov_bayes_logit,
            "sets_diff": sets_diff,
            "games_diff": games_diff,
            "point_diff": point_diff,
        }
        X = pd.DataFrame([[feat_values[c] for c in self.features]], columns=self.features)
        residual_logit = float(self.booster.predict(X, raw_score=True)[0])
        final_logit = residual_logit + p_markov_logit
        p_final = _sigmoid(final_logit)

        return {
            "p1_win_prob": p_final,
            "p_markov": p_markov,
            "p_markov_bayes": p_markov_bayes,
            "p1_serve_prior": float(p1_prior_serve),
            "p2_serve_prior": float(p2_prior_serve),
            "p1_serve_post": float(p1_post),
            "p2_serve_post": float(p2_post),
            "gbm_logit_adjustment": residual_logit,
        }


def predict_one(artifact_dir: str | Path = "ml/artifacts", **kwargs) -> dict:
    """Convenience wrapper: load model and run one prediction."""
    return TennisModel.load(artifact_dir).predict(**kwargs)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    m = TennisModel.load(here / "artifacts")
    out = m.predict(prior_p1=0.65,
                    p1_sets=1, p2_sets=0,
                    p1_games=3, p2_games=2,
                    p1_pt=2, p2_pt=1,
                    server=1,
                    p1_serve_won=24, p1_serve_total=35,
                    p2_serve_won=18, p2_serve_total=29)
    for k, v in out.items():
        print(f"  {k:24s} = {v:.4f}")
