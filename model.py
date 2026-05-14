"""Live-odds-integration Markov model for in-play tennis (best of 3).

A pre-match bookmaker prior is solved for the latent serve probabilities
(p1_serve, p2_serve) under a fixed mean-serve-strength `alpha`. At every
in-play tick those serve probabilities can be re-calibrated from a live
market mid (Kalshi or any other source) by a 1-D L2-regularised fit on a
single scalar `beta` (the player-strength asymmetry). The same calibrated
(p1_serve, p2_serve) is then decomposed via the closed-form Markov chain
into FIVE markets:

    1. Main      — P(player 1 wins match)
    2. Set 1     — P(player 1 wins set 1)
    3. Set 2     — P(player 1 wins set 2)
    4. Exact     — 4 outcomes (p1 2-0, p1 2-1, p2 2-0, p2 2-1)
    5. Total games O/U N — P(total games at match end > N)

All five markets are read off the SAME chain, so they are self-consistent
by construction (no separate models, no ML training step).

Usage
-----
    from model import LiveMarkovModel

    m = LiveMarkovModel(prior_p1=0.65, match_first_server=1)

    # First tick — pre-match
    m.update(p1_sets=0, p2_sets=0, p1_games=0, p2_games=0,
             p1_pt=0, p2_pt=0, server=1)
    print(m.p_match,                 # ≈ 0.65
          m.p_set_winner(1),         # ≈ 0.60ish
          m.p_set_winner(2),         # ≈ 0.60ish
          m.p_exact_score(),         # {'p1_2_0': ..., ...}
          m.p_total_games_over(20))  # P(total games > 20)

    # Later tick — feed in a live market mid
    m.update(p1_sets=1, p2_sets=0,
             p1_games=3, p2_games=2, p1_pt=2, p2_pt=1, server=1,
             kalshi_mid=0.81, lam=0.5,
             games_in_completed_sets=10)  # set 1 was 6-4

The L2 regularisation strength `lam` controls how strongly the live fit is
pulled toward the pre-match prior solve:

    lam = inf  -> no live update (= pure prior-only baseline)
    lam = 0    -> exactly match Kalshi mid (no regularisation)
    lam = 0.5  -> default; honest compromise

See README.md for the mathematical setup and how the decomposition works.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

from scipy.optimize import minimize_scalar

from tennis_markov import (
    MatchState, p_win_match,
    _p_set_from_full, _p_set_from_games,
    _p_game, _p_tb,
    solve_serve_for_prior,
)


# ---------------------------------------------------------------------------
# Calibration: 1-D solve for beta from a target win prob, then L2-regularised
# live update against a market mid.
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _beta_to_serves(beta: float, alpha: float) -> tuple[float, float]:
    return _sigmoid(alpha + beta), _sigmoid(alpha - beta)


def _solve_beta_for_prior(prior_p1: float, match_first_server: int,
                          alpha: float) -> float:
    """Find β such that the pre-match closed-form Markov P(p1) equals prior_p1.

    Uses the existing solver in tennis_markov (which itself bisects on β with
    α fixed). We then invert one of the serve probs to recover β explicitly.
    """
    p1s, _ = solve_serve_for_prior(prior_p1, match_first_server=match_first_server,
                                    alpha=alpha)
    return _logit(p1s) - alpha


def _fit_beta(state: MatchState, market_mid: float, beta_prior: float,
              alpha: float, lam: float) -> float:
    """β(t) = argmin (market_mid − markov_match(state, β))² + λ · (β − β_prior)².

    λ = ∞  → β_prior (no live update).
    λ = 0  → exact 1-D bisection on the monotone match-win function.
    """
    if math.isinf(lam):
        return beta_prior

    if lam == 0.0:
        # Pure bisection on a monotone function.
        lo, hi = -3.0, 3.0
        flo = p_win_match(state, *_beta_to_serves(lo, alpha)) - market_mid
        fhi = p_win_match(state, *_beta_to_serves(hi, alpha)) - market_mid
        if flo > 0:
            return lo
        if fhi < 0:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = p_win_match(state, *_beta_to_serves(mid, alpha)) - market_mid
            if abs(fm) < 1e-6:
                return mid
            if flo * fm < 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    # Regularised — bounded scalar minimisation.
    def loss(beta: float) -> float:
        pwin = p_win_match(state, *_beta_to_serves(beta, alpha))
        return (market_mid - pwin) ** 2 + lam * (beta - beta_prior) ** 2

    res = minimize_scalar(loss, bounds=(-3.0, 3.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


# ---------------------------------------------------------------------------
# Per-set helpers (which player serves the first game of set N).
# ---------------------------------------------------------------------------

def _set_first_server(set_idx: int, mfs: int) -> int:
    """1-based set index. Set 1 server = mfs; set 2 flips; set 3 flips back."""
    return mfs if set_idx % 2 == 1 else 3 - mfs


# ---------------------------------------------------------------------------
# Set-score distribution (closed form, BFS over game tree).
#
# Returns a dict {(p1_games, p2_games): probability} where (p1_games, p2_games)
# is a terminal set score: (6,0..4), (7,5), (7,6), or symmetric.
# ---------------------------------------------------------------------------

def _is_set_terminal(p1g: int, p2g: int) -> bool:
    if p1g >= 6 and p1g >= p2g + 2:
        return True
    if p2g >= 6 and p2g >= p1g + 2:
        return True
    if p1g == 7 and p2g == 5:
        return True
    if p1g == 5 and p2g == 7:
        return True
    if p1g == 7 and p2g == 6:
        return True
    if p1g == 6 and p2g == 7:
        return True
    return False


def _p_p1_wins_game_fresh(server: int, p1s: float, p2s: float) -> float:
    if server == 1:
        return _p_game(0, 0, p1s)
    return 1.0 - _p_game(0, 0, p2s)


def _p_p1_wins_game_from_pts(server: int, p1_pt: int, p2_pt: int,
                              p1s: float, p2s: float) -> float:
    if server == 1:
        return _p_game(p1_pt, p2_pt, p1s)
    # When player 2 serves, _p_game returns P(server wins) = P(p2 wins).
    return 1.0 - _p_game(p2_pt, p1_pt, p2s)


@lru_cache(maxsize=4096)
def _set_score_dist_fresh(p1s: float, p2s: float,
                          set_first_server: int) -> tuple[tuple[int, int, float], ...]:
    """Distribution over terminal set scores starting fresh from 0-0.

    Returned as a tuple of (p1_games, p2_games, probability) triples so the
    result is hashable / cache-friendly.
    """
    dist: dict[tuple[int, int], float] = {}
    frontier = [(0, 0, set_first_server, 1.0)]
    while frontier:
        p1g, p2g, srv, w = frontier.pop()
        if _is_set_terminal(p1g, p2g):
            dist[(p1g, p2g)] = dist.get((p1g, p2g), 0.0) + w
            continue
        if p1g == 6 and p2g == 6:
            p_tb = _p_tb(0, 0, set_first_server, p1s, p2s)
            dist[(7, 6)] = dist.get((7, 6), 0.0) + w * p_tb
            dist[(6, 7)] = dist.get((6, 7), 0.0) + w * (1.0 - p_tb)
            continue
        p_p1 = _p_p1_wins_game_fresh(srv, p1s, p2s)
        next_srv = 2 if srv == 1 else 1
        frontier.append((p1g + 1, p2g, next_srv, w * p_p1))
        frontier.append((p1g, p2g + 1, next_srv, w * (1.0 - p_p1)))
    return tuple((a, b, p) for (a, b), p in dist.items())


def _set_score_dist_from_state(p1s: float, p2s: float,
                                set_first_server: int,
                                p1_games: int, p2_games: int,
                                server: int,
                                p1_pt: int, p2_pt: int) -> dict[tuple[int, int], float]:
    """Distribution over terminal set scores starting from a given in-set state."""
    if _is_set_terminal(p1_games, p2_games):
        return {(p1_games, p2_games): 1.0}
    dist: dict[tuple[int, int], float] = {}
    if p1_pt == 0 and p2_pt == 0:
        frontier = [(p1_games, p2_games, server, 1.0)]
    else:
        if p1_games == 6 and p2_games == 6:
            # In a tiebreak — use _p_tb at the current tiebreak point state.
            p_tb = _p_tb(p1_pt, p2_pt, set_first_server, p1s, p2s)
            return {(7, 6): p_tb, (6, 7): 1.0 - p_tb}
        p_p1 = _p_p1_wins_game_from_pts(server, p1_pt, p2_pt, p1s, p2s)
        next_srv = 2 if server == 1 else 1
        frontier = [
            (p1_games + 1, p2_games, next_srv, p_p1),
            (p1_games, p2_games + 1, next_srv, 1.0 - p_p1),
        ]
    while frontier:
        p1g, p2g, srv, w = frontier.pop()
        if _is_set_terminal(p1g, p2g):
            dist[(p1g, p2g)] = dist.get((p1g, p2g), 0.0) + w
            continue
        if p1g == 6 and p2g == 6:
            p_tb = _p_tb(0, 0, set_first_server, p1s, p2s)
            dist[(7, 6)] = dist.get((7, 6), 0.0) + w * p_tb
            dist[(6, 7)] = dist.get((6, 7), 0.0) + w * (1.0 - p_tb)
            continue
        p_p1 = _p_p1_wins_game_fresh(srv, p1s, p2s)
        next_srv = 2 if srv == 1 else 1
        frontier.append((p1g + 1, p2g, next_srv, w * p_p1))
        frontier.append((p1g, p2g + 1, next_srv, w * (1.0 - p_p1)))
    return dist


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class LiveMarkovModel:
    """Live-calibrated Markov chain for in-play tennis (best of 3)."""

    def __init__(self, prior_p1: float, match_first_server: int = 1,
                 alpha: float = 0.61):
        """
        Parameters
        ----------
        prior_p1 : float
            Pre-match P(player 1 wins the match), already de-vigged across
            books (the two implied probs should sum to 1).
        match_first_server : int
            Player (1 or 2) who serves the very first game of the match.
        alpha : float
            Mean serve-point-win rate logit. 0.61 ≈ logit(0.65), the long-run
            average across surfaces in pro tennis. Fixed across calibration
            (only β moves).
        """
        if prior_p1 <= 0 or prior_p1 >= 1:
            raise ValueError(f"prior_p1 must be in (0, 1), got {prior_p1}")
        if match_first_server not in (1, 2):
            raise ValueError(f"match_first_server must be 1 or 2, got {match_first_server}")

        self.prior_p1 = float(prior_p1)
        self.mfs = int(match_first_server)
        self.alpha = float(alpha)
        self.beta_prior = _solve_beta_for_prior(self.prior_p1, self.mfs, self.alpha)

        # Live state — populated by update().
        self.state: Optional[MatchState] = None
        self.beta = self.beta_prior
        self.p1s, self.p2s = _beta_to_serves(self.beta, self.alpha)
        self.set_winners: dict[int, int] = {}
        self.games_in_completed_sets: Optional[int] = None
        self._prev_sets: Optional[tuple[int, int]] = None
        self.last_market_mid: Optional[float] = None
        self.last_lam: Optional[float] = None

    # --- update ------------------------------------------------------------

    def update(self, *,
               p1_sets: int = 0, p2_sets: int = 0,
               p1_games: int = 0, p2_games: int = 0,
               p1_pt: int = 0, p2_pt: int = 0,
               server: int = 1,
               kalshi_mid: Optional[float] = None,
               lam: float = 0.5,
               games_in_completed_sets: Optional[int] = None,
               set_winners: Optional[dict[int, int]] = None) -> None:
        """Update the model with the current state and (optional) market mid.

        Parameters
        ----------
        p1_sets, p2_sets : int
            Sets won so far. Match ends at 2.
        p1_games, p2_games : int
            Games in the current set (0..7).
        p1_pt, p2_pt : int
            Points in the current game. 0/1/2/3 = 0/15/30/40; deuce/advantage
            when both ≥ 3. At 6-6 in games these are tiebreak points (0..).
        server : int
            Player serving the current game (1 or 2).
        kalshi_mid : float, optional
            Live market mid for P(p1 wins match). If provided, β is fit
            against it with strength `lam`. If None, β stays at β_prior.
        lam : float
            L2 strength: λ=∞ → no live update; λ=0 → exact fit; λ=0.5 default.
        games_in_completed_sets : int, optional
            Total games played in already-completed sets (e.g. 10 if set 1
            was 6-4). REQUIRED for `p_total_games_over(...)` queries.
        set_winners : dict, optional
            Map set_idx -> 1 or 2 for past sets. Usually auto-tracked from
            (p1_sets, p2_sets) transitions across `update()` calls; pass
            explicitly to override or to bootstrap if you skip ticks.
        """
        # Track set transitions so set_winners is filled in automatically.
        if self._prev_sets is not None:
            prev_p1, prev_p2 = self._prev_sets
            if p1_sets > prev_p1:
                self.set_winners[prev_p1 + prev_p2 + 1] = 1
            if p2_sets > prev_p2:
                self.set_winners[prev_p1 + prev_p2 + 1] = 2
        if set_winners is not None:
            self.set_winners.update(set_winners)
        self._prev_sets = (p1_sets, p2_sets)

        self.state = MatchState(
            p1_sets=p1_sets, p2_sets=p2_sets,
            p1_games=p1_games, p2_games=p2_games,
            p1_pt=p1_pt, p2_pt=p2_pt,
            server=server, match_first_server=self.mfs,
        )
        self.games_in_completed_sets = games_in_completed_sets
        self.last_market_mid = kalshi_mid
        self.last_lam = lam

        if kalshi_mid is None:
            self.beta = self.beta_prior
        else:
            self.beta = _fit_beta(self.state, float(kalshi_mid), self.beta_prior,
                                   self.alpha, float(lam))
        self.p1s, self.p2s = _beta_to_serves(self.beta, self.alpha)

    # --- accessors: market probabilities -----------------------------------

    @property
    def p_match(self) -> float:
        """P(player 1 wins the match)."""
        self._require_update()
        return float(p_win_match(self.state, self.p1s, self.p2s))

    def p_set_winner(self, set_idx: int) -> float:
        """P(player 1 wins set `set_idx`). Sets are 1-indexed."""
        self._require_update()
        if set_idx not in (1, 2, 3):
            raise ValueError(f"set_idx must be 1, 2 or 3, got {set_idx}")
        st = self.state
        sets_done = st.p1_sets + st.p2_sets
        if sets_done >= set_idx:
            return 1.0 if self.set_winners.get(set_idx) == 1 else 0.0
        if sets_done == set_idx - 1:
            sfs = _set_first_server(set_idx, self.mfs)
            return float(_p_set_from_full(
                st.p1_games, st.p2_games, st.p1_pt, st.p2_pt, st.server,
                self.p1s, self.p2s, sfs,
            ))
        # Future set — under constant (p1s, p2s) it is independent of past sets.
        sfs = _set_first_server(set_idx, self.mfs)
        return float(_p_set_from_games(0, 0, sfs, self.p1s, self.p2s, sfs))

    def p_exact_score(self) -> dict[str, float]:
        """4 outcomes: p1_2_0, p1_2_1, p2_2_0, p2_2_1. Sum to 1."""
        self._require_update()
        st = self.state
        a, b = st.p1_sets, st.p2_sets

        if a >= 2:
            if b == 0:
                return {"p1_2_0": 1.0, "p1_2_1": 0.0, "p2_2_0": 0.0, "p2_2_1": 0.0}
            return {"p1_2_0": 0.0, "p1_2_1": 1.0, "p2_2_0": 0.0, "p2_2_1": 0.0}
        if b >= 2:
            if a == 0:
                return {"p1_2_0": 0.0, "p1_2_1": 0.0, "p2_2_0": 1.0, "p2_2_1": 0.0}
            return {"p1_2_0": 0.0, "p1_2_1": 0.0, "p2_2_0": 0.0, "p2_2_1": 1.0}

        current_set = a + b + 1
        sfs_curr = _set_first_server(current_set, self.mfs)
        q_curr = float(_p_set_from_full(
            st.p1_games, st.p2_games, st.p1_pt, st.p2_pt, st.server,
            self.p1s, self.p2s, sfs_curr,
        ))

        def q_fresh(k: int) -> float:
            sfs = _set_first_server(k, self.mfs)
            return float(_p_set_from_games(0, 0, sfs, self.p1s, self.p2s, sfs))

        if (a, b) == (0, 0):
            q1, q2, q3 = q_curr, q_fresh(2), q_fresh(3)
            return {
                "p1_2_0": q1 * q2,
                "p1_2_1": q1 * (1 - q2) * q3 + (1 - q1) * q2 * q3,
                "p2_2_0": (1 - q1) * (1 - q2),
                "p2_2_1": (1 - q1) * q2 * (1 - q3) + q1 * (1 - q2) * (1 - q3),
            }
        if (a, b) == (1, 0):
            q2, q3 = q_curr, q_fresh(3)
            return {
                "p1_2_0": q2,
                "p1_2_1": (1 - q2) * q3,
                "p2_2_0": 0.0,
                "p2_2_1": (1 - q2) * (1 - q3),
            }
        if (a, b) == (0, 1):
            q2, q3 = q_curr, q_fresh(3)
            return {
                "p1_2_0": 0.0,
                "p1_2_1": q2 * q3,
                "p2_2_0": 1 - q2,
                "p2_2_1": q2 * (1 - q3),
            }
        if (a, b) == (1, 1):
            q3 = q_curr
            return {
                "p1_2_0": 0.0,
                "p1_2_1": q3,
                "p2_2_0": 0.0,
                "p2_2_1": 1 - q3,
            }
        raise RuntimeError(f"unexpected state ({a}, {b})")

    def total_games_distribution(self) -> dict[int, float]:
        """Distribution over total games at match end (including completed sets)."""
        self._require_update()
        if self.games_in_completed_sets is None:
            raise ValueError(
                "games_in_completed_sets must be passed to update(...) for "
                "total-games queries. Sum of all games played in already-"
                "finished sets (e.g. 10 if set 1 was 6-4)."
            )
        return _total_games_distribution(
            self.state, self.p1s, self.p2s, self.mfs,
            int(self.games_in_completed_sets),
        )

    def p_total_games_over(self, threshold: int) -> float:
        """P(final total games > threshold). Default ATP O/U line is 20."""
        dist = self.total_games_distribution()
        return float(sum(p for T, p in dist.items() if T > threshold))

    def p_total_games_under(self, threshold: int) -> float:
        """P(final total games < threshold). Push at == is venue-defined."""
        dist = self.total_games_distribution()
        return float(sum(p for T, p in dist.items() if T < threshold))

    # --- introspection -----------------------------------------------------

    def _require_update(self):
        if self.state is None:
            raise RuntimeError("Call update(...) at least once before reading probabilities.")

    def summary(self) -> dict:
        """All five market probabilities + calibration internals in one dict."""
        self._require_update()
        out = {
            "p_match": self.p_match,
            "p_set1": self.p_set_winner(1),
            "p_set2": self.p_set_winner(2),
            "p_exact_score": self.p_exact_score(),
            "beta": self.beta,
            "beta_prior": self.beta_prior,
            "p1_serve_prob": self.p1s,
            "p2_serve_prob": self.p2s,
            "set_winners_so_far": dict(self.set_winners),
        }
        if self.games_in_completed_sets is not None:
            out["p_total_games_over_20"] = self.p_total_games_over(20)
            out["p_total_games_over_21"] = self.p_total_games_over(21)
            out["p_total_games_over_22"] = self.p_total_games_over(22)
        return out


# ---------------------------------------------------------------------------
# Total games — closed-form (best of 3 enumeration).
# ---------------------------------------------------------------------------

def _total_games_distribution(state: MatchState, p1s: float, p2s: float,
                                mfs: int,
                                games_in_completed_sets: int) -> dict[int, float]:
    """Total games at match end given current state + (p1s, p2s)."""
    sets_played = state.p1_sets + state.p2_sets
    if state.p1_sets >= 2 or state.p2_sets >= 2:
        T = games_in_completed_sets + state.p1_games + state.p2_games
        return {T: 1.0}

    current_set_idx = sets_played + 1
    sfs_curr = _set_first_server(current_set_idx, mfs)
    D_curr = _set_score_dist_from_state(
        p1s, p2s, sfs_curr,
        state.p1_games, state.p2_games, state.server,
        state.p1_pt, state.p2_pt,
    )

    next_set_idx = current_set_idx + 1
    if next_set_idx <= 3:
        sfs_next = _set_first_server(next_set_idx, mfs)
        D_next = {(a, b): p for (a, b, p) in
                   _set_score_dist_fresh(p1s, p2s, sfs_next)}
    else:
        D_next = None

    if next_set_idx + 1 <= 3:
        sfs_3rd = _set_first_server(next_set_idx + 1, mfs)
        D_3rd = {(a, b): p for (a, b, p) in
                  _set_score_dist_fresh(p1s, p2s, sfs_3rd)}
    else:
        D_3rd = None

    p1_sets_now = state.p1_sets
    p2_sets_now = state.p2_sets

    dist: dict[int, float] = {}
    for (a_c, b_c), p_c in D_curr.items():
        curr_winner = 1 if a_c > b_c else 2
        s1 = p1_sets_now + (1 if curr_winner == 1 else 0)
        s2 = p2_sets_now + (1 if curr_winner == 2 else 0)
        g_after_c = games_in_completed_sets + a_c + b_c

        if s1 >= 2 or s2 >= 2:
            dist[g_after_c] = dist.get(g_after_c, 0.0) + p_c
            continue

        if D_next is None:
            continue

        for (a_n, b_n), p_n in D_next.items():
            next_winner = 1 if a_n > b_n else 2
            s1n = s1 + (1 if next_winner == 1 else 0)
            s2n = s2 + (1 if next_winner == 2 else 0)
            g_after_n = g_after_c + a_n + b_n

            if s1n >= 2 or s2n >= 2:
                dist[g_after_n] = dist.get(g_after_n, 0.0) + p_c * p_n
                continue

            if D_3rd is None:
                continue
            for (a_3, b_3), p_3 in D_3rd.items():
                T = g_after_n + a_3 + b_3
                dist[T] = dist.get(T, 0.0) + p_c * p_n * p_3

    return dist
