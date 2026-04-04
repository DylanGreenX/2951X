# THIS IS OUT-DATED --> UDPATE[KAM]

# RLang NPC Demo: Needle in a Haystack

A pygame demo showing NPCs with **embodied knowledge** — they only know what they've personally observed while pursuing their own goals.

## Core Concept

Unlike omniscient NPCs that have access to the full game state, our NPCs:
1. **Have their own goals** (not the player's goal)
2. **Explore autonomously** to achieve those goals
3. **Accumulate incidental observations** as a side effect
4. **Communicate only what they've witnessed** to players/LLMs

This creates realistic, non-omniscient NPCs whose knowledge feels earned and believable.

---

## Architecture

### 1. The Setup

- **Player Goal**: Find the Red Triangle (needle in haystack)
- **NPC Goal**: Collect Blue Circles (its own reward function)
- **Side Effect**: While chasing blue circles, the NPC observes red triangles, green squares, etc.

### 2. RLang Grounding Layer (`rlang_engine.py`)

We implement core RLang concepts as lightweight Python classes:

```python
class RLangState:
    # FACTORS (raw state slices)
    npc_pos: tuple[int, int]           # Factor npc_pos := S[0:2]

    # ACCUMULATED MEMORY (built from observations over time)
    observed_cells: set                # Every cell the NPC has "seen"
    observed_shapes: list[Shape]       # Shapes personally encountered

    # PROPOSITIONS (boolean beliefs computed from memory)
    @property
    def seen_any_red_triangle(self) -> bool:
        return "red_triangle" in self.shape_locations

    @property
    def coverage(self) -> float:
        return len(self.observed_cells) / (self.world_size ** 2)
```

**Key insight**: The NPC's knowledge is **partial and earned**. It doesn't know about shapes outside its sight range or exploration history.

### 3. Goal-Driven Exploration (`npc_brain.py`)

The NPC uses a simple greedy policy:
1. **If blue circles are known** → move toward nearest one
2. **Else** → explore (biased random walk favoring unvisited areas)

```python
def _choose_direction(self) -> str:
    # Goal-directed: move toward nearest known blue circle
    if self.state.known_blue_circle_positions:
        target = self._nearest_blue_circle()
        if target:
            return self._direction_toward(target)

    # Exploration: biased random walk
    return self._explore_direction()
```

This creates **realistic knowledge gaps**. Early game: *"I haven't explored much yet."* Late game: *"I saw triangles near (5,12) while heading toward blue circles."*

### 4. LLM Integration (`to_llm_context()`)

The RLang state serializes to natural language strings:

```
[FACTOR] I am at position (12, 9).
[FACTOR] I have explored 124/225 cells (55% of the world).
[GOAL] I have collected 2 blue circles.
[PROPOSITION] I have explored the NW, NE, SW, SE region(s).
[OBSERVATION] I saw red triangle(s) at: (10,11), (14,8).
[PROPOSITION] I HAVE seen red triangle(s) at: (10,11), (14,8).
```

These strings get injected into the LLM system prompt. The LLM can then respond authentically: *"I saw a red triangle at (10,11) while I was collecting blue circles in the northeast area."*

---

## Architecture Decisions

### Why not use the `rlang` pip package directly?

The RLang package is designed for **training RL agents**, not runtime NPC grounding. We implement RLang *concepts* (Factors, Propositions, Effects) as lightweight classes purpose-built for our use case.

**Migration path**: Our classes mirror RLang semantics exactly. Later, we can replace them with parsed `.rlang` files and trained RL agents without changing the serialization layer.

### Why give the NPC its own goal?

**Real NPCs should be characters, not sensors.** A random-walking NPC that gathers info is just a mobile database. An NPC that's **trying to do something** and happens to observe things along the way feels like a person with their own agenda.

The player's quest target (red triangle) becomes **incidental knowledge** in the NPC's pursuit of blue circles. This asymmetry creates natural, believable dialogue patterns.

### Why greedy policy instead of random walk?

Even simple goal-directed behavior produces more realistic exploration patterns than random movement. The NPC has **reasons** for being in different areas, which translates to more interesting dialogue: *"I was heading toward the blue circle cluster when I spotted triangles near the forest."*

---

## Extending to Full RL + RLang

### Phase 1: Current Implementation
- ✅ RLang grounding layer (Factors, Propositions, Effects)
- ✅ Goal-driven exploration (greedy policy)
- ✅ LLM context serialization
- ✅ Partial, embodied knowledge

### Phase 2: RL Integration

Replace the greedy policy with a trained RL agent:

1. **Write RLang program** defining the NPC's task:
```rlang
# npc_collector.rlang
Factor npc_pos := S[0:2]
Factor sight_grid := S[2:51]

Goal collect_blue := blue_circle_collected >= 5
Reward blue_circle_reward := 1.0
Reward step_penalty := -0.01

Policy explore:
  if blue_circle_nearby:
    Execute collect
  else:
    Execute frontier_explore
```

2. **Use RLang parser** to compile this into:
   - **Dynamics & Task Knowledge** — partial T(s,a,s'), R(s,a,s')
   - **Solution Knowledge** — partial policy hints, options

3. **Train RL agent** (Q-learning, PPO, etc.) that exploits the RLang knowledge for faster learning

4. **Keep serialization layer unchanged** — `to_llm_context()` works the same regardless of whether the underlying policy is greedy or trained

### Phase 3: Advanced Features

- **Multiple NPCs** with different goals and knowledge domains
- **NPC-to-NPC communication** (gossip, information trading)
- **Temporal knowledge decay** (observations become stale)
- **Hierarchical goals** using RLang Options framework

---

## Running the Demo

```bash
cd toy_game
python main.py
```

**Controls**:
- Arrow keys: Move player
- R: Reset world with new random seed
- ESC: Quit

Watch the bottom panel to see how RLang state evolves as the NPC explores!

---

## Key Files

- `config.py` — All tunable parameters
- `entities.py` — Player, NPC, Shape classes
- `world.py` — Grid world generation
- `rlang_engine.py` — **Core RLang grounding layer**
- `npc_brain.py` — **Goal-driven exploration logic**
- `main.py` — Pygame rendering + game loop

The heart of the system is in `rlang_engine.py` (knowledge representation) and `npc_brain.py` (behavior). Everything else is just scaffolding for the demo.

---

## Future: Real Game Integration

This architecture scales to real games:

1. **Replace pygame grid** with actual game world
2. **Replace Shape objects** with game entities (NPCs, items, locations)
3. **Expand RLang schema** to match game's state representation
4. **Add domain-specific goals** (trader NPC collects gold, guard NPC patrols routes)
5. **Train specialized policies** for each NPC type using RLang knowledge

The `to_llm_context()` serialization remains the clean interface between game logic and AI dialogue systems.
