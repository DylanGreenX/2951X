"""Game entities: shapes, player, NPC."""
from dataclasses import dataclass, field


@dataclass
class Shape:
    color: str        # "red", "blue", "green", "yellow"
    shape_type: str   # "triangle", "circle", "square"
    x: int
    y: int
    collected: bool = False

    @property
    def label(self) -> str:
        return f"{self.color}_{self.shape_type}"

    @property
    def display_label(self) -> str:
        return f"{self.color} {self.shape_type}"


@dataclass
class Player:
    x: int
    y: int


@dataclass
class NPC:
    x: int
    y: int
    sight_range: int = 2
    steps_taken: int = 0
    blue_circles_collected: int = 0
