"""Game entities: shapes, player, NPC."""
from dataclasses import dataclass, field


@dataclass
class Shape:
    color: str        # see config.COLORS
    shape_type: str   # see config.SHAPES
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
    sight_range: int = 2
    observed_cells: set = field(default_factory=set)

@dataclass
class NPC:
    x: int
    y: int
    sight_range: int = 2
    steps_taken: int = 0
    # goal_label is set when NPC_GOAL is active (e.g. "blue_circle").
    # None means this NPC is a baseline wanderer with no goal.
    goal_label: str | None = None
