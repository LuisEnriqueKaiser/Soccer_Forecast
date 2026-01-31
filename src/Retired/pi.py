from math import log10, exp

class PiRating:
    """A Python implementation of the Pi Rating system (Constantinou & Fenton, 2013).

    Parameters
    ----------
    lambda_ : float, default 0.035
        Learning rate controlling how quickly ratings react to new information.
    gamma : float, default 0.7
        Catch‑up rate that propagates venue‑specific updates to the opposite venue rating.
    b : float, default 10
        Logarithmic base used in the non‑linear transformation of rating magnitudes to
        expected score differentials.
    c : float, default 3
        Scaling constant used both in the transformation above and in the weighting of
        prediction errors.

    Notes
    -----
    * Each team holds two ratings – one for home matches (``rating['home']``) and one
      for away matches (``rating['away']``).
    * After every game the expected goal difference is compared with the actual goal
      difference.  The surprise (weighted error) drives the rating update.
    * The algorithm is zero‑centred by construction; improvements to one team imply
      declines of equal magnitude elsewhere.
    """

    def __init__(self, lambda_: float = 0.035, gamma: float = 0.7, b: float = 10, c: float = 3):
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.b = float(b)
        self.c = float(c)
        # team -> {"home": rating, "away": rating}
        self._ratings: dict[str, dict[str, float]] = {}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _ensure_team(self, team: str) -> dict[str, float]:
        """Initialise a team with zero ratings if it does not yet exist."""
        if team not in self._ratings:
            self._ratings[team] = {"home": 0.0, "away": 0.0}
        return self._ratings[team]

    def _transform_rating(self, r: float) -> float:
        """Convert a rating to an expected score difference versus an average side."""
        val = (self.b ** (abs(r) / self.c) - 1)
        return -val if r < 0 else val

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def expected_score_diff(self, home_team: str, away_team: str) -> float:
        """Return the expected goal difference (home – away) before a match."""
        h_home = self._transform_rating(self._ensure_team(home_team)["home"])
        a_away = self._transform_rating(self._ensure_team(away_team)["away"])
        return h_home - a_away

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        """Update ratings after a completed match."""
        score_diff = home_goals - away_goals

