"""Grid world generation and state."""
import random
from entities import Shape
import config
from entities import Player


class GameWorld:
    def __init__(
        self,
        target_color: str,
        target_shape: str,
        player_start: tuple[int, int] | None = None,
        npc_start: tuple[int, int] | None = None,
    ):
        self.size = config.GRID_SIZE
        self.shapes: list[Shape] = []
        self.target_color = target_color
        self.target_shape = target_shape
        self._player_start = player_start or config.PLAYER_START
        self._npc_start = npc_start or config.NPC_START
        self._generate(target_color, target_shape)

    def _generate(self, target_color: str, target_shape: str):
        """Place shapes randomly on the grid, avoiding collisions."""
        occupied = set()

        # Reserve NPC and player starts
        occupied.add(self._npc_start)
        occupied.add(self._player_start)

        shape_specs = [(target_color, target_shape)]
        shape_specs += [
            (color, shape)
            for color in config.COLORS
            for shape in config.SHAPES
            if not (color == target_color and shape == target_shape)
        ]

        for color, shape_type in shape_specs:
            label = f"{color}_{shape_type}"
            
            # Use fixed position if RANDOM_SPAWN is False and label exists in config
            if not config.RANDOM_SPAWN and label in config.FIXED_SHAPE_POSITIONS:
                pos = config.FIXED_SHAPE_POSITIONS[label]
                if pos in occupied:
                    pos = self._random_free_pos(occupied)
            else:
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