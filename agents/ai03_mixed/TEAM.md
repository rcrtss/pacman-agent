# ai03_mixed

**Strategy:** Balanced adaptive team — strategic offender + opportunistic counter-defender.

- **StrategicOffender**: Multi-priority offensive agent.
  - Returns home when carrying ≥ 4 food or ≤ 2 food remain on the board.
  - Seeks capsules when an active ghost is nearby.
  - Hunts scared ghosts for bonus points after a capsule is eaten.
  - Avoids active ghosts via weighted feature penalty.
- **CounterDefender**: Dual-mode defensive agent.
  - Directly chases the nearest visible invader when one is spotted.
  - Switches to light offense (food grabbing) when the home side is clear.
  - Penalizes stopping and reversing to keep pressure on.

**Training value:** Medium-difficulty balanced opponent. Forces your Q-agent to handle
both a smart attacker and a defender that can shift roles mid-game.
