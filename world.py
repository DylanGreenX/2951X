"""Grid world generation and state."""
import random
from entities import Shape
import config
from entities import Player


class GameWorld:
    def __init__(self, target_color: str, target_shape: str, seed=None):
        if seed is not None:
            random.seed(seed)

        self.size = config.GRID_SIZE
        self.shapes: list[Shape] = []
        self.target_color = target_color
        self.target_shape = target_shape
        self._generate(target_color, target_shape)

    def _generate(self, target_color: str, target_shape: str):
        """Place shapes on the grid, avoiding collisions.

        When ``config.RANDOM_SPAWN`` is False, honour
        ``config.FIXED_SHAPE_POSITIONS`` for any label with a fixed cell (used
        to line shapes up with the painted map for demos). Anything without a
        fixed cell — or any fixed cell already claimed — falls back to random
        free placement. The experiment pipeline keeps RANDOM_SPAWN=True so
        per-seed variation is preserved.
        """
        occupied = set()

        # Reserve actor/player starts by default. Benchmarks can provide a
        # broader stable set so paired seeds keep object placement comparable
        # even when the actual number of NPC starts changes.
        reserved_positions = getattr(config, "WORLD_RESERVED_POSITIONS", None)
        if reserved_positions is None:
            reserved_positions = [
                *(getattr(config, "NPC_STARTS", None) or [config.NPC_START]),
                config.PLAYER_START,
            ]
        for start in reserved_positions:
            occupied.add(tuple(start))

        use_fixed = not getattr(config, "RANDOM_SPAWN", True)
        fixed = getattr(config, "FIXED_SHAPE_POSITIONS", {}) if use_fixed else {}

        shape_specs = [(target_color, target_shape)]
        shape_specs += [
            (color, shape)
            for color in config.COLORS
            for shape in config.SHAPES
            if not (color == target_color and shape == target_shape)
        ]

        for color, shape_type in shape_specs:
            label = f"{color}_{shape_type}"
            pos = fixed.get(label)
            if pos is None or pos in occupied:
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
    
    def update_player_vision(self, player: Player):
        visible = self.get_visible_cells(player.x, player.y, player.sight_range)
        for x, y, shape in visible:
            player.observed_cells.add((x,y))

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size
