"""
orientation_solver.py

Provides classes and functions to determine the correct board orientation
for backgammon video analysis by scoring move legality across four candidate
orientations. Designed to work with sequences of stable `BoardState` objects
extracted from video frames.

Key features:
- Generate four candidate orientations (identity, reverse, 180-rotate, 180-rotate+reverse).
- Map raw per-point detections into standard `1..24` point numbering for each
  candidate orientation.
- Infer moves between consecutive stable board states (including bar and borne-off).
- Score each inferred transition using a configurable MoveScorer that accounts
  for movement direction, dice consistency, bar entry, bearing off, hits, and
  impossible events.

Usage:
    Provide a list of `BoardState` snapshots (raw detected per-point counts and
    bar/borne counts). Call `OrientationSolver.score_orientations()` with the
    list and optionally a parallel list of detected dice tuples (or None).

Complexity and robustness notes are included at the bottom of the file.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Dict, NamedTuple
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Candidate raw-index mappings for the four board orientations.
# Each list says which raw point index (0-based) should become standard point
# position 1..24 at that candidate orientation.
ORIENTATION_POINT_MAPPINGS: List[List[int]] = [
    list(range(24)),
    list(reversed(range(24))),
    list(range(12, 24)) + list(range(0, 12)),
    list(reversed(list(range(12, 24)) + list(range(0, 12)))),
]


def renumber_points_by_orientation(points: List[Any], orientation_index: int) -> List[Any]:
    """Return a reordered copy of `points` for one of the four orientations.

    The returned list is in standard point order 1..24, and each point's `id`
    is updated to match that standard numbering. The original list is not
    modified.
    """

    mapping = ORIENTATION_POINT_MAPPINGS[orientation_index]
    renamed: List[object] = []

    for standard_idx, raw_idx in enumerate(mapping, start=1):
        point = points[raw_idx].copy()
        point.id = standard_idx
        renamed.append(point)

    return renamed


@dataclass
class BoardState:
    """Representation of a stable board state in standard point order 1..24.

    Attributes:
        white: list of 24 ints: number of white checkers on points 1..24.
        black: list of 24 ints: number of black checkers on points 1..24.
        bar_white: checkers on white's bar.
        bar_black: checkers on black's bar.
        borne_white: white checkers borne off.
        borne_black: black checkers borne off.
    """

    white: List[int]
    black: List[int]
    bar_white: int = 0
    bar_black: int = 0
    borne_white: int = 0
    borne_black: int = 0

    def total_checkers(self) -> Tuple[int, int]:
        return (sum(self.white) + self.bar_white + self.borne_white,
                sum(self.black) + self.bar_black + self.borne_black)

    def copy(self) -> "BoardState":
        return BoardState(list(self.white), list(self.black),
                          self.bar_white, self.bar_black,
                          self.borne_white, self.borne_black)


class Move(NamedTuple):
    color: str  # 'white' or 'black'
    src: Optional[int]  # 1..24 for board points, None for bar, -1 for borne
    dst: Optional[int]  # 1..24 for board points, None for borne, -1 for bar
    checker_count: int
    distance: int
    is_bearing_off: bool


def _permute_points(lst: List[int], mapping: List[int]) -> List[int]:
    return [lst[i] for i in mapping]


def generate_orientations(raw: BoardState) -> List[BoardState]:
    """Generate four candidate orientations from a raw per-point ordering.

    We keep the raw list order as provided and produce four permutations using
    combinations of reversing and 12-point rotation (180-degree).

    Reason: with only a per-point (24) linear index and no other calibration,
    the four orientation candidates can be produced by identity, reverse,
    rotate-12, rotate-12+reverse. This covers the four possible corners that
    may correspond to point 1 in video.

    Args:
        raw: BoardState whose `white` and `black` lists are in the camera's
            raw per-point ordering (length 24 each).

    Returns:
        List[BoardState] of length 4, each in standard 1..24 ordering under a
        different orientation hypothesis.
    """

    assert len(raw.white) == 24 and len(raw.black) == 24, "expected 24 points"

    identity_map = list(range(24))
    reverse_map = list(reversed(identity_map))
    rotate12_map = list(range(12, 24)) + list(range(0, 12))
    rotate12_reverse = list(reversed(rotate12_map))

    mappings = [identity_map, reverse_map, rotate12_map, rotate12_reverse]

    results: List[BoardState] = []
    for m in mappings:
        w = _permute_points(raw.white, m)
        b = _permute_points(raw.black, m)
        bs = BoardState(w, b, raw.bar_white, raw.bar_black,
                        raw.borne_white, raw.borne_black)
        results.append(bs)

    return results


def _signed_delta(a: List[int], b: List[int], idx: int) -> int:
    return b[idx] - a[idx]


def infer_moves_between_states(from_state: BoardState,
                               to_state: BoardState,
                               color: str) -> List[Move]:
    """Infer a multiset of moves for `color` between two board states.

    This function produces a plausible mapping of decreases -> increases.
    The inference is greedy and conservative (matches nearest legal targets
    first). It also detects bar entries and bearing-off moves.

    Returns:
        List of Move objects (may be empty). Distances are positive integers;
        for bearing off distance is computed as number of pips moved past 1 or
        24 depending on color.
    """

    assert color in ("white", "black")
    src = from_state
    dst = to_state

    if color == "white":
        a = src.white
        b = dst.white
        bar_from = src.bar_white
        bar_to = dst.bar_white
    else:
        a = src.black
        b = dst.black
        bar_from = src.bar_black
        bar_to = dst.bar_black

    decreases: List[Tuple[int, int]] = []  # (index 1..24, count)
    increases: List[Tuple[int, int]] = []

    for i in range(24):
        delta = b[i] - a[i]
        if delta < 0:
            decreases.append((i + 1, -delta))
        elif delta > 0:
            increases.append((i + 1, delta))

    moves: List[Move] = []

    # Handle bar -> board transitions first (bar decreases, board increases)
    bar_entries = 0
    if bar_from > bar_to:
        bar_entries = bar_from - bar_to

    # If bar decreased, consume increases that represent entries (highest
    # priority). We'll match bar entries to increases in the entry quadrant.
    entry_targets: List[int] = []
    if color == "white":
        # white enters on points 24..19 (indices 24 down to 19)
        entry_targets = [i for i in range(19, 25)]
    else:
        # black enters on points 1..6
        entry_targets = [i for i in range(1, 7)]

    # Flatten increases to a list of single-checker targets for greedy matching
    flat_increases: List[int] = []
    for idx, cnt in increases:
        flat_increases.extend([idx] * cnt)

    # First match bar entries
    for _ in range(bar_entries):
        # find an increase in entry_targets if available, else pop any increase
        chosen = None
        for t in entry_targets:
            if t in flat_increases:
                chosen = t
                flat_increases.remove(t)
                break
        if chosen is None and flat_increases:
            chosen = flat_increases.pop(0)
        if chosen is not None:
            # distance: from bar to entry point: for scoring we treat as point
            # index of chosen (white moves downward: higher->lower) distance
            # computed later by scorer; here use proxy distance as chosen
            moves.append(Move(color, None, chosen, 1, abs(chosen), False))

    # Now build lists of flat decreases (sources) and remaining increases
    flat_decreases: List[int] = []
    for idx, cnt in decreases:
        flat_decreases.extend([idx] * cnt)

    # match decreases to remaining increases greedily by nearest legal target
    # build remaining increases list
    remaining_increases = list(flat_increases)

    while flat_decreases and remaining_increases:
        s = flat_decreases.pop(0)
        # choose target closest in index, prefer legal direction
        best_t = None
        best_score = math.inf
        for t in remaining_increases:
            dist = abs(s - t)
            dir_ok = (color == "white" and s > t) or (color == "black" and s < t)
            penalty = 0 if dir_ok else 10_000  # big penalty for illegal direction
            if dist + penalty < best_score:
                best_score = dist + penalty
                best_t = t
        if best_t is None:
            # fallback
            best_t = remaining_increases.pop(0)
        else:
            remaining_increases.remove(best_t)

        moves.append(Move(color, s, best_t, 1, abs(s - best_t), False))

    # Any unmatched decreases may represent bearing off (to borne) or illegal
    # disappearance; detect borne_off increases
    borne_increase = 0
    if color == "white":
        if dst.borne_white > src.borne_white:
            borne_increase = dst.borne_white - src.borne_white
    else:
        if dst.borne_black > src.borne_black:
            borne_increase = dst.borne_black - src.borne_black

    for _ in range(borne_increase):
        if flat_decreases:
            s = flat_decreases.pop(0)
            # bearing off distance: for white from s to 0 (off board), distance = s
            if color == "white":
                dist = s
            else:
                dist = 25 - s
            moves.append(Move(color, s, None, 1, dist, True))

    # any remaining decreases or increases are suspicious; add them as moves
    while flat_decreases:
        s = flat_decreases.pop(0)
        moves.append(Move(color, s, None, 1, 0, False))
    while remaining_increases:
        t = remaining_increases.pop(0)
        moves.append(Move(color, None, t, 1, 0, False))

    # merge same-source-destination moves to counts
    merged: Dict[Tuple[Optional[int], Optional[int]], Move] = {}
    for m in moves:
        key = (m.src, m.dst)
        if key in merged:
            old = merged[key]
            merged[key] = Move(m.color, m.src, m.dst, old.checker_count + m.checker_count, m.distance, m.is_bearing_off)
        else:
            merged[key] = m

    return list(merged.values())


class MoveScorer:
    """Score a set of moves for a color according to backgammon legality heuristics.

    Scoring decisions are tunable via weights below. The scorer returns a
    floating point score: higher is better (more legal). Large negative
    penalties reduce the cumulative score for illegal orientations.
    """

    # Weights (tunable)
    W_DIRECTION_OK = 3.0
    W_DIRECTION_BAD = -50.0
    W_DICE_MATCH = 5.0
    W_DICE_SUM_MATCH = 2.0
    W_IMPOSSIBLE = -200.0
    W_BAR_ENTRY_OK = 10.0
    W_BAR_ENTRY_BAD = -50.0
    W_BEAR_OFF_OK = 10.0
    W_BEAR_OFF_BAD = -100.0
    W_HIT_OK = 8.0
    W_HIT_BAD = -20.0

    def __init__(self, dice: Optional[Tuple[int, int]] = None):
        self.dice = dice

    def score_transition(self,
                         from_state: BoardState,
                         to_state: BoardState) -> float:
        score = 0.0

        # Quick impossible checks
        white_total_from, black_total_from = from_state.total_checkers()
        white_total_to, black_total_to = to_state.total_checkers()
        if white_total_to > 15 or black_total_to > 15:
            logger.debug("Impossible checker counts")
            return self.W_IMPOSSIBLE

        # detect hits: opponent blot disappears and bar increases accordingly
        # For efficiency, compute per-point deltas for both colors
        w_moves = infer_moves_between_states(from_state, to_state, "white")
        b_moves = infer_moves_between_states(from_state, to_state, "black")

        # Movement direction and dice consistency scoring
        for m in w_moves + b_moves:
            # direction
            if m.src is None or m.dst is None:
                dir_ok = True  # bar/borne handled separately
            else:
                if m.color == "white":
                    dir_ok = (m.src > m.dst)
                else:
                    dir_ok = (m.src < m.dst)
            if dir_ok:
                score += self.W_DIRECTION_OK * m.checker_count
            else:
                score += self.W_DIRECTION_BAD * m.checker_count

            # dice
            if self.dice and not m.is_bearing_off:
                d1, d2 = self.dice
                dist = m.distance
                if dist == d1 or dist == d2:
                    score += self.W_DICE_MATCH * m.checker_count
                elif dist == d1 + d2:
                    score += self.W_DICE_SUM_MATCH * m.checker_count
                else:
                    # Penalize large impossible distances (e.g., > max(d1,d2)
                    # unless legal bearing off or doubling of same die allowed)
                    max_die = max(d1, d2)
                    if dist > max_die:
                        score += -1.0 * m.checker_count

        # Bar entry rule: if player had checkers on bar at `from_state`, they
        # must be the ones moved first. We approximate by checking that the
        # number of bar checkers in `to_state` decreased only by entries.
        # If there remain checkers on bar while other checkers moved, penalize.
        if from_state.bar_white > 0:
            # if white moved anything other than bar entries, penalize
            non_bar_moves = [mv for mv in w_moves if mv.src is not None]
            if non_bar_moves and to_state.bar_white > 0:
                score += self.W_BAR_ENTRY_BAD
            else:
                score += self.W_BAR_ENTRY_OK
        if from_state.bar_black > 0:
            non_bar_moves = [mv for mv in b_moves if mv.src is not None]
            if non_bar_moves and to_state.bar_black > 0:
                score += self.W_BAR_ENTRY_BAD
            else:
                score += self.W_BAR_ENTRY_OK

        # Bearing off rule: only allowed when all checkers are in home board
        # For white home is points 1..6; for black home is 19..24.
        # Penalize bearing off if any checker outside home remains.
        for color in ("white", "black"):
            moves = w_moves if color == "white" else b_moves
            for mv in moves:
                if mv.is_bearing_off:
                    # check all checkers are in home in from_state
                    if color == "white":
                        outside = sum(from_state.white[6:])
                    else:
                        outside = sum(from_state.black[:18])
                    if outside == 0:
                        score += self.W_BEAR_OFF_OK * mv.checker_count
                    else:
                        score += self.W_BEAR_OFF_BAD * mv.checker_count

        # Hits: detect when opponent point decreases by 1 and opponent bar increases
        # match patterns: single-opponent-decrease and bar increase
        # compute opponent deltas
        for i in range(24):
            dw = to_state.white[i] - from_state.white[i]
            db = to_state.black[i] - from_state.black[i]
            # white hit black at i -> black decreased and bar_black increased
            if db < 0 and to_state.bar_black > from_state.bar_black:
                score += self.W_HIT_OK
            if dw < 0 and to_state.bar_white > from_state.bar_white:
                score += self.W_HIT_OK

        # Impossible events: more than 4 checker moves per player in one turn
        if len(w_moves) > 4 or len(b_moves) > 4:
            score += self.W_IMPOSSIBLE

        return score


class OrientationSolver:
    """High-level solver to score all four orientations across sequences.

    Methods:
        score_orientations(raw_states, dice_sequence): returns (scores dict, best_index)
    """

    def __init__(self):
        pass

    def score_orientations(self,
                           raw_states: List[BoardState],
                           dice_sequence: Optional[Sequence[Optional[Tuple[int, int]]]] = None
                           ) -> Tuple[Dict[int, float], int]:
        """Score the four orientation candidates across a sequence of stable states.

        Args:
            raw_states: list of BoardState objects in the same raw per-point order
                as produced by the detector.
            dice_sequence: optional list of same length-1 of detected dice tuples
                (d1,d2) or None for each transition. If None, dice are not used.

        Returns:
            (orientation_scores, best_index)
        """

        assert len(raw_states) >= 1
        if dice_sequence is None:
            dice_sequence = [None for _ in range(max(0, len(raw_states) - 1))]
        dice_sequence = list(dice_sequence)
        assert len(dice_sequence) == max(0, len(raw_states) - 1)

        orientation_scores: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        # Pre-generate orientation hypotheses for each raw snapshot
        oriented_sequences: List[List[BoardState]] = []  # [orientation][time]
        for idx in range(4):
            oriented_sequences.append([])

        for raw in raw_states:
            cands = generate_orientations(raw)
            for i, bs in enumerate(cands):
                oriented_sequences[i].append(bs)

        # Score each orientation by summing transition scores over time
        for i in range(4):
            seq = oriented_sequences[i]
            total = 0.0
            for t in range(len(seq) - 1):
                dice = dice_sequence[t]
                scorer = MoveScorer(dice)
                s = scorer.score_transition(seq[t], seq[t + 1])
                total += s
            orientation_scores[i] = total

        best_idx = max(orientation_scores.items(), key=lambda kv: kv[1])[0]
        return orientation_scores, best_idx


if __name__ == "__main__":
    # Minimal example showing how to call OrientationSolver. In practice the
    # `raw_states` should be produced by your detection/reconstruction pipeline
    # where each `BoardState` white/black lists follow detector raw ordering.

    # Example synthetic raw: empty board
    empty = BoardState([0] * 24, [0] * 24)

    # Example: white moves one checker from point 13 to point 8 (a 5-pip move)
    s1 = BoardState([0] * 24, [0] * 24)
    s1.white[12] = 1  # point 13 has a white checker

    s2 = s1.copy()
    s2.white[12] -= 1
    s2.white[7] += 1  # point 8

    solver = OrientationSolver()
    scores, best = solver.score_orientations([s1, s2], dice_sequence=[(5, 2)])
    print("scores:", scores)
    print("best orientation:", best)


"""
Complexity discussion

Let N be the number of stable snapshots and P=24 the fixed number of points.

- Generating the 4 candidate orientations per snapshot is O(P) each (constant
  factor), so O(4*P*N) = O(N).
- Inferring moves between two states runs in O(P) for deltas and greedy
  matching (we flatten at most 15 checkers), again O(1) per transition with
  small constant factors; across all transitions O(N).
- Scoring each transition requires examining inferred moves and point deltas,
  again O(P) per transition. Entire pipeline is O(N) time with small constants.

Memory usage is O(4*P*N) to hold oriented copies if done naively; we keep only
small per-orientation sequences here, which is acceptable for typical N.

Robustness suggestions

- Use probabilistic matching instead of deterministic greedy: incorporate
  detection confidence scores for individual checkers (if available) and run
  a maximum-likelihood assignment (Hungarian algorithm) for ambiguous moves.
- Smooth detection noise: aggregate detections across a short window to
  stabilize counts and avoid transient occlusion effects (e.g., hand passes).
- Use temporal priors: prefer moves that are consistent with recent moves and
  dice history (e.g., prefer bar entries when bar exists). Weighting can be
  learned from labeled example games.
- If checker occlusion is frequent, integrate a hidden Markov model (HMM)
  where latent true board states emit noisy detections; infer most likely
  orientation jointly with state sequence via Viterbi-like dynamic programming.
- Add heuristics for matching ambiguous multiple-checker moves (e.g., a
  stack split) and for noisy counts (allow +/-1 tolerance and prefer
  explanations requiring fewer simultaneous changes).

"""
