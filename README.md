# RLang NPC Demo: Needle in a Haystack

A pygame demo showing NPCs with **embodied knowledge** — they only know what they've personally observed while pursuing their own goals.

## Core Concept

Unlike omniscient NPCs that have access to the full game state, our NPCs:

1. Have their own goals (not the player's goal)
2. Explore autonomously to achieve those goals
3. Accumulate incidental observations as a side effect
4. Communicate only what they've witnessed to players

## Architecture

**Player Goal**: Find the Red Triangle (needle in haystack)
**NPC Goal**: Collect Blue Circles (its own reward function)
**Side Effect**: While chasing blue circles, the NPC observes other shapes

The NPC's knowledge about the player's quest target is incidental — a byproduct of pursuing its own agenda.

### RLang Grounding Layer (`rlang_engine.py`)

Implements RLang concepts (Factors, Propositions) as lightweight Python classes.
The key method `to_llm_context()` serializes NPC knowledge into natural language
strings that get injected into an LLM prompt.

### NPC Brains (`npc_brain.py`)

- `NPCBrainWandering` — baseline random exploration, no goal
- `NPCBrainGoalDriven` — pursues a configurable target greedily
- `MemoryDecayNPCBrain` — observations fade over time (LLM/SLM only)
- `SelectiveAttentionNPCBrain` — only notices one attribute (LLM/SLM only)
- `CompetitiveNPCBrain` — withholds info from player (LLM/SLM only)

### Response Generation (`interaction.py`)

- `deterministic` — direct lookup against NPC's observed locations
- `llm` — full language model generates response from NPC knowledge context
- `slm` — small language model for practical latency

## Experimental Design

We evaluate NPC response quality across two independent variables:
**knowledge type** (how the NPC learns) and **response type** (how it answers).

### Core Baselines (2x3 Matrix)

```
                     Deterministic    LLM         SLM
                    ┌──────────────┬───────────┬───────────┐
  Perfect Knowledge │  baseline    │ quality   │ practical │
                    │  (games now) │ ceiling   │ ceiling   │
                    ├──────────────┼───────────┼───────────┤
  Embodied Knowledge│  embodied    │ full      │ main      │
                    │  baseline    │ system    │ contrib.  │
                    └──────────────┴───────────┴───────────┘
```

Reading left to right tells the **response quality story**:

- **Deterministic**: What games do today. Instant lookup, accurate but robotic.
Binary responses — either gives exact coordinates or says "I don't know."
- **LLM**: What's possible if latency doesn't matter. Natural language,
can express partial knowledge ("I saw triangles nearby but not red ones").
Establishes the quality ceiling.
- **SLM**: What's actually deployable in games. Fast enough for real-time
interaction while retaining most of the LLM's language quality.
This is the practical contribution.

Reading top to bottom tells the **embodiment story**:

- **Perfect Knowledge**: NPC knows everything — traditional game NPC behavior.
Controls for the effect of language model quality in isolation.
- **Embodied Knowledge**: NPC only knows what it explored — realistic but
incomplete. Shows the cost of embodiment (some accuracy loss) and the
benefit (responses feel grounded in actual experience).

The six conditions together answer:

1. How much quality does a language model add over deterministic lookup?
2. How much quality does an SLM retain compared to a full LLM?
3. How much accuracy do we lose from embodied vs perfect knowledge?
4. Is the embodiment cost worth the realism gain?

### Extended Modalities (4x2 Matrix)

Extensions only test LLM and SLM. Deterministic responses are pure lookup
tables — they can't express uncertainty, strategically withhold information,
or acknowledge forgotten observations.

```
                          LLM         SLM
                        ┌───────────┬───────────┐
  Memory Decay          │           │           │  knowledge fades over time
                        ├───────────┼───────────┤
  Selective Attention   │           │           │  only notices one attribute
                        ├───────────┼───────────┤
  Competitive Sharing   │           │           │  strategically withholds info
                        ├───────────┼───────────┤
  Social Learning       │           │           │  NPCs share with each other
  (stretch goal)        │           │           │
                        └───────────┴───────────┘
```

Each row modifies a different aspect of the NPC's knowledge pipeline:

**Knowledge Acquisition** (what the NPC knows when asked):

- **Memory Decay**: Observations expire after N steps. Tests whether
language models can naturally express temporal uncertainty
("I think I saw something red, but that was a while ago...").
- **Selective Attention**: NPC only retains observations matching its
goal attribute (e.g., only notices blue objects). Tests the impact
of goal-directed attention on incidental learning about the player's target.

**Information Sharing** (what the NPC chooses to reveal):

- **Competitive Sharing**: NPC and player want the same target.
The NPC knows the answer but strategically gives vague or misleading
responses. Tests whether language models can produce believable
deception grounded in actual knowledge.

**Knowledge Source** (stretch goal):

- **Social Learning**: Multiple NPCs share observations with each
other. An NPC might know about the red triangle because another
NPC told it, not because it saw it directly. Tests information
propagation and second-hand knowledge.

### Full Test Space: 14 Conditions

```
Core baselines:    6 conditions  (2 knowledge x 3 response)
Extensions:        6 conditions  (3 modalities x 2 response)
Stretch:           2 conditions  (social learning x 2 response)
Total:            14 conditions
```

### Evaluation Metrics

Each trial captures:

- **Response text** — full content for qualitative analysis
- **Response time** — milliseconds from question to answer
- **Accuracy** — does the response contain correct coordinates?
- **Relevance** — does the response help the player find the target?
- **Groundedness** — does the response stay within NPC's actual knowledge?

Naturalness will be evaluated separately via subjective human rating.

## Running

```bash
# Pygame demo
python main.py

# Run deterministic baseline experiments
python experiment.py

**Demo controls**: Arrow keys to move, R to reset, ESC to quit.

## Extending to Full RL + RLang

The greedy policy in `NPCBrainGoalDriven` is a stand-in for a trained RL agent.
To integrate real RL:

1. Write `.rlang` file defining NPC's task (Factors, Propositions, Effects, Goal)
2. Use RLang parser to compile into partial world model + policy hints
3. Train RL agent (Q-learning, PPO) that exploits the RLang knowledge
4. Replace `_choose_direction()` with trained policy

The `to_llm_context()` serialization layer stays unchanged regardless of
whether the underlying policy is greedy or trained.