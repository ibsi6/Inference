"""
Closed-form Markov chain for ATP/WTA best-of-3 tennis match win probabilities.

Reference: O'Malley (2008), "Probability formulas for a tennis match".

Public API
----------
    p_win_match(state, p1_serve, p2_serve) -> float
        Returns probability that PLAYER 1 wins the match from a given state.

State is a tuple/dict with:
    p1_sets, p2_sets          : sets won so far (best of 3: terminal at 2)
    p1_games, p2_games        : games in current set
    p1_pt, p2_pt              : points in current game
                                  (0,1,2,3 = 0/15/30/40, deuce when both >=3 and equal,
                                   advantage when both >=3 and differ by 1)
    server                    : 1 if PLAYER 1 serves the current game, 2 otherwise
    in_tiebreak               : bool (deduced from games)

The functions handle the standard best-of-3 scoring (no fifth set).
At games 6-6 a 7-point tiebreak is played, win by 2.

Implementation
--------------
We memoize on integer state tuples. Game and tiebreak transitions are
recursive with closed-form terminal solutions for deuce / advantage.

The function `solve_serve_diff_for_prior` recovers (p1_serve, p2_serve)
from a target pre-match P(player 1 wins). We parameterize via two scalars
α (mean logit) and β (difference) and solve a 1-D root in β with α fixed.
"""

from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass
import math

# ----- Game (regular) ------------------------------------------------------

@lru_cache(maxsize=None)
def _p_game(server_pts: int, returner_pts: int, p: float) -> float:
    """Server's prob of winning a regular game from (server_pts, returner_pts).

    p = probability server wins any individual point.
    """
    # Terminal
    if server_pts >= 4 and server_pts >= returner_pts + 2:
        return 1.0
    if returner_pts >= 4 and returner_pts >= server_pts + 2:
        return 0.0
    # Deuce / advantage closed forms
    if server_pts >= 3 and returner_pts >= 3:
        d_server = p * p / (p * p + (1 - p) * (1 - p))
        if server_pts == returner_pts:  # deuce
            return d_server
        elif server_pts == returner_pts + 1:  # adv server
            return p + (1 - p) * d_server
        else:  # adv returner
            return p * d_server
    # Recursion (server pts < 4 and returner pts < 4 and not both >= 3)
    return p * _p_game(server_pts + 1, returner_pts, p) + (1 - p) * _p_game(server_pts, returner_pts + 1, p)


# ----- Tiebreak ------------------------------------------------------------
# Tiebreak: first to 7 points, win by 2. Serving order: server S serves point 1,
# then R serves 2-3, then S serves 4-5, ... Alternates every 2 points starting
# after the first.

def _tb_server_at(point_idx_zero: int, first_server: int) -> int:
    """Return 1 or 2 — who serves the next point. first_server is 1 or 2.
    point_idx_zero counts the number of points already played in the tiebreak.
    """
    # First point: first_server serves. Then points 2,3: opp serves; 4,5: first; ...
    if point_idx_zero == 0:
        return first_server
    # After the first point, server flips every 2 points.
    # Group 0 -> [point 1] (first server)
    # Group 1 -> [points 2,3] (other)
    # Group 2 -> [points 4,5] (first)
    # Group 3 -> [points 6,7] (other)
    # Effective group index:
    group = (point_idx_zero + 1) // 2  # 1 for points 1..2, 2 for 3..4, 3 for 5..6 ...
    # Actually let's be more careful: define server(k) for k=0,1,2,...
    # server(0) = first
    # server(1) = other, server(2) = other
    # server(3) = first, server(4) = first
    # server(5) = other, server(6) = other ...
    # So server toggles after every 2 points, starting from k=1.
    if point_idx_zero == 0:
        return first_server
    k = point_idx_zero - 1  # k=0..1 -> other, k=2..3 -> first, k=4..5 -> other, ...
    flips = (k // 2) % 2
    return (first_server if flips == 1 else (3 - first_server))


@lru_cache(maxsize=None)
def _p_tb(sa: int, sb: int, first_server: int, p1_pt: float, p2_pt: float) -> float:
    """Probability player 1 wins the tiebreak from state (player1_pts=sa, player2_pts=sb).

    first_server: who served the first tiebreak point (1 or 2).
    p1_pt: P(player 1 wins a point when player 1 is serving).
    p2_pt: P(player 2 wins a point when player 2 is serving).
    """
    if sa >= 7 and sa >= sb + 2:
        return 1.0
    if sb >= 7 and sb >= sa + 2:
        return 0.0
    # Tiebreak "deuce" at (k, k) for k >= 6: next two points decide or return to tie.
    # Closed form using the two server identities for the next two points.
    if sa >= 6 and sb >= 6 and sa == sb:
        points_played = sa + sb
        s1 = _tb_server_at(points_played, first_server)
        s2 = _tb_server_at(points_played + 1, first_server)
        a = p1_pt if s1 == 1 else (1 - p2_pt)
        b = p1_pt if s2 == 1 else (1 - p2_pt)
        denom = 1 - a - b + 2 * a * b
        return (a * b) / denom if denom > 0 else 0.5
    points_played = sa + sb
    server = _tb_server_at(points_played, first_server)
    p_p1_wins_pt = p1_pt if server == 1 else (1 - p2_pt)
    return p_p1_wins_pt * _p_tb(sa + 1, sb, first_server, p1_pt, p2_pt) + \
           (1 - p_p1_wins_pt) * _p_tb(sa, sb + 1, first_server, p1_pt, p2_pt)


# ----- Set -----------------------------------------------------------------

@lru_cache(maxsize=None)
def _p_set_from_games(p1_games: int, p2_games: int, server: int,
                     p1_pt: float, p2_pt: float, set_first_server: int) -> float:
    """Probability player 1 wins the set from a games-only state (assumes 0-0 in current game).

    server: 1 or 2 — who serves the next game.
    set_first_server: who served the first game of the set (needed for tiebreak server order).
    """
    # Terminal: someone wins the set
    if p1_games >= 6 and p1_games >= p2_games + 2:
        return 1.0
    if p2_games >= 6 and p2_games >= p1_games + 2:
        return 0.0
    # Set decided at 7-5 or 7-6
    if p1_games == 7:
        return 1.0
    if p2_games == 7:
        return 0.0
    # Tiebreak at 6-6
    if p1_games == 6 and p2_games == 6:
        # Tiebreak first server: opponent of the player who served the last set game
        # The 13th game would have been served by set_first_server (since games alternate).
        # In a tiebreak, the first server is the player who would have served game 13.
        # Game k served by set_first_server if k odd. Game 13 is odd. So first_server for tiebreak
        # = set_first_server. But wait — common rule: tiebreak first server is the player who
        # would have served the 13th game (next after 6-6). Game count after this hypothetical
        # game would be 13 = odd → set_first_server.
        first_tb_server = set_first_server
        p_win_tb = _p_tb(0, 0, first_tb_server, p1_pt, p2_pt)
        return p_win_tb
    # Play current game (no tiebreak)
    if server == 1:
        p_p1_holds = _p_game(0, 0, p1_pt)
    else:
        p_p1_holds = 1 - _p_game(0, 0, p2_pt)
    next_server = 2 if server == 1 else 1
    return (
        p_p1_holds * _p_set_from_games(p1_games + 1, p2_games, next_server, p1_pt, p2_pt, set_first_server) +
        (1 - p_p1_holds) * _p_set_from_games(p1_games, p2_games + 1, next_server, p1_pt, p2_pt, set_first_server)
    )


@lru_cache(maxsize=None)
def _p_set_from_full(p1_games: int, p2_games: int, p1_pt_g: int, p2_pt_g: int, server: int,
                    p1_pt: float, p2_pt: float, set_first_server: int) -> float:
    """Probability player 1 wins the set from a full state including current game points.

    p1_pt_g, p2_pt_g: current point score (0..3 normally; if both >=3 indicates deuce/adv).
    server: 1 or 2 — who serves the current game.
    """
    # If both points are zero, defer to games-only state
    if p1_pt_g == 0 and p2_pt_g == 0:
        return _p_set_from_games(p1_games, p2_games, server, p1_pt, p2_pt, set_first_server)
    # If we're at 6-6 we'd be in a tiebreak — but tiebreak points use different scoring.
    if p1_games == 6 and p2_games == 6:
        first_tb_server = set_first_server
        return _p_tb(p1_pt_g, p2_pt_g, first_tb_server, p1_pt, p2_pt)
    # Play out current game from its current point state
    if server == 1:
        p_p1_holds_now = _p_game(p1_pt_g, p2_pt_g, p1_pt)
    else:
        # When player 2 serves, _p_game returns prob player 2 wins, so player 1 wins = 1 - that.
        # Note: server's points and returner's points have to be ordered for _p_game.
        # When server is player 2: server_pts = p2_pt_g, returner_pts = p1_pt_g.
        p_p2_holds = _p_game(p2_pt_g, p1_pt_g, p2_pt)
        p_p1_holds_now = 1 - p_p2_holds
    next_server = 2 if server == 1 else 1
    # If game ends: transition to next game (points reset)
    # We compute as: this game's outcome is binary
    return (
        p_p1_holds_now * _p_set_from_games(p1_games + 1, p2_games, next_server, p1_pt, p2_pt, set_first_server) +
        (1 - p_p1_holds_now) * _p_set_from_games(p1_games, p2_games + 1, next_server, p1_pt, p2_pt, set_first_server)
    )


# ----- Match ---------------------------------------------------------------

def _set_first_server(p1_games: int, p2_games: int, server_now: int) -> int:
    """Given that current set has played (p1_games + p2_games) games and the
    next game is served by `server_now`, return the player who served the first
    game of this set.

    Game 1 of the set served by F. Game 2 by ~F (other). Game k by F if k odd.
    Next game number = p1_games + p2_games + 1.
    If next_game_idx is odd → next server = first server.
    If next_game_idx is even → next server = other = (3 - first server).
    """
    next_game = p1_games + p2_games + 1
    if next_game % 2 == 1:
        return server_now
    return 3 - server_now


@dataclass
class MatchState:
    p1_sets: int = 0
    p2_sets: int = 0
    p1_games: int = 0
    p2_games: int = 0
    p1_pt: int = 0
    p2_pt: int = 0
    server: int = 1   # who serves the *current* game (1 or 2)
    # Player who served the very first game of the MATCH. The first server of each
    # subsequent set alternates. Most ATP matches: match_first_server = the player
    # who served the first point. We default to player 1.
    match_first_server: int = 1


@lru_cache(maxsize=None)
def _p_match_inner(p1_sets: int, p2_sets: int, p1_games: int, p2_games: int,
                   p1_pt: int, p2_pt: int, server: int, match_first_server: int,
                   p1_serve: float, p2_serve: float) -> float:
    """Probability PLAYER 1 wins the best-of-3 match.

    Cache-keyed on the discrete state and the two serve probs; very fast across
    many snapshots from the same match.
    """
    if p1_sets >= 2:
        return 1.0
    if p2_sets >= 2:
        return 0.0
    set_idx_1based = p1_sets + p2_sets + 1
    set_first = match_first_server if set_idx_1based % 2 == 1 else (3 - match_first_server)
    p_win_set = _p_set_from_full(
        p1_games, p2_games, p1_pt, p2_pt, server, p1_serve, p2_serve, set_first
    )
    next_set_first = 3 - set_first
    # Win current set
    pwin = _p_match_inner(p1_sets + 1, p2_sets, 0, 0, 0, 0, next_set_first,
                          match_first_server, p1_serve, p2_serve)
    plose = _p_match_inner(p1_sets, p2_sets + 1, 0, 0, 0, 0, next_set_first,
                           match_first_server, p1_serve, p2_serve)
    return p_win_set * pwin + (1 - p_win_set) * plose


def p_win_match(state: MatchState, p1_serve: float, p2_serve: float) -> float:
    """Public entry point. p1_serve = prob player 1 wins point on her own serve;
    p2_serve = same for player 2."""
    return _p_match_inner(
        state.p1_sets, state.p2_sets, state.p1_games, state.p2_games,
        state.p1_pt, state.p2_pt, state.server, state.match_first_server,
        p1_serve, p2_serve,
    )


# ----- Calibration ---------------------------------------------------------

def _logit(x: float) -> float:
    return math.log(x / (1 - x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def solve_serve_for_prior(prior_p1: float, match_first_server: int = 1,
                          alpha: float = 0.61,
                          beta_lo: float = -1.5, beta_hi: float = 1.5,
                          tol: float = 1e-5, max_iter: int = 60) -> tuple[float, float]:
    """Solve for (p1_serve, p2_serve) such that the pre-match Markov P(p1 wins) equals `prior_p1`.

    Parameterization: p1_serve = sigmoid(alpha + beta), p2_serve = sigmoid(alpha - beta).
    alpha defaults to logit(0.65) ≈ 0.619 — the long-run average serve point win rate
    in pro tennis (~63-65% across surfaces).

    Returns (p1_serve, p2_serve).
    """
    prior_p1 = min(max(prior_p1, 1e-4), 1 - 1e-4)

    def f(beta: float) -> float:
        p1s = _sigmoid(alpha + beta)
        p2s = _sigmoid(alpha - beta)
        # Clear caches that depend on (p1s, p2s) — they're parameters, so caches are
        # per-call and we can simply not use lru_cache for these...
        # Instead, since serve probs are arguments, the lru_cache is keyed on them.
        return p_win_match(MatchState(match_first_server=match_first_server), p1s, p2s) - prior_p1

    a, b = beta_lo, beta_hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        # Expand bounds if needed
        for _ in range(10):
            a -= 0.5; b += 0.5
            fa, fb = f(a), f(b)
            if fa * fb <= 0:
                break
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol:
            a = b = mid
            break
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    beta = 0.5 * (a + b)
    p1s = _sigmoid(alpha + beta)
    p2s = _sigmoid(alpha - beta)
    return p1s, p2s


if __name__ == "__main__":
    # Quick sanity check
    for pr in [0.20, 0.50, 0.75, 0.90]:
        p1s, p2s = solve_serve_for_prior(pr)
        achieved = p_win_match(MatchState(), p1s, p2s)
        print(f"prior_p1={pr:.2f} → p1_serve={p1s:.3f}, p2_serve={p2s:.3f}, achieved={achieved:.4f}")
    # Demo at 1-0 sets, 5-3 games in current set, 40-30 with p1 serving
    p1s, p2s = solve_serve_for_prior(0.5)
    s = MatchState(p1_sets=1, p2_sets=0, p1_games=5, p2_games=3, p1_pt=3, p2_pt=2, server=1)
    print(f"\nState 1-0, 5-3, 40-30 p1 serve, equal players: P(p1 wins) = {p_win_match(s, p1s, p2s):.4f}")
