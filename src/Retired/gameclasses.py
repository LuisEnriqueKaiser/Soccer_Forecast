from dataclasses import dataclass


@dataclass
class Game:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    date: str


first_game = Game(
    home_team="Lions",
    away_team="Tigers",
    home_score=24,
    away_score=30,
    date="2023-09-15"
)
print(first_game.date)