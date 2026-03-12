"""Grid world generation and state."""
import random
from entities import Shape
import config


class GameWorld:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.size = config.GRID_SIZE
        self.shapes: list[Shape] = []
        self._generate()

    def _generate(self):
        """Place shapes randomly on the grid, avoiding collisions."""
        occupied = set()

        # Reserve NPC and player starts
        occupied.add(config.NPC_START)
        occupied.add(config.PLAYER_START)

        shape_specs = (
            [("blue", "circle")] * config.NUM_BLUE_CIRCLES +
            [("red", "triangle")] * config.NUM_RED_TRIANGLES +
            [("green", "square")] * config.NUM_GREEN_SQUARES +
            [("yellow", "triangle")] * config.NUM_YELLOW_TRIANGLES
        )

        for color, shape_type in shape_specs:
            pos = self._random_free_pos(occupied)
            occupied.add(pos)
            self.shapes.append(Shape(color, shape_type, pos[0], pos[1]))

    def _random_free_pos(self, occupied: set) -> tuple[int, int]:
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) not in occupied:
                return (x, y)

    def shape_at(self, x: int, y: int) -> Shape | None:
        for s in self.shapes:
            if s.x == x and s.y == y and not s.collected:
                return s
        return None

    def get_visible_cells(self, cx: int, cy: int, r: int) -> list[tuple[int, int, Shape | None]]:
        """Return all cells within sight range r of (cx, cy)."""
        cells = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    cells.append((nx, ny, self.shape_at(nx, ny)))
        return cells

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size
