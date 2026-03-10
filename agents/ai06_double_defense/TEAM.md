# ai06_double_defense

**Strategy:** Two coordinated defenders — zero intentional offense.

- **SentinelAgent**: Patrols the boundary entrance column top-to-bottom in a continuous sweep.
  The moment an invader is visible it drops the patrol and chases via BFS.
  Acts as the outer defensive ring.
- **InteriorGuard**: Roams between the N food positions on our side that are most exposed
  (i.e. closest to the boundary). Provides interior coverage for any pacman that bypasses
  the Sentinel. Also chases visible invaders directly via BFS.

**Training value:** Forces the Q-agent to score despite maximum defensive pressure.
Tests whether it has learned creative routing, luring defenders, capsule usage,
and patience — skills that pure offensive agents can mask.
