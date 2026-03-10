# ai05_bfs

**Strategy:** BFS path-following agents — harder than reflex, harder than greedy.

- **BFSAttacker**: Computes and follows the actual BFS-shortest path to the nearest food.
  Detours to the nearest capsule when an active ghost is within 5 steps.
  Returns home once carrying ≥ 5 food or ≤ 2 food remain.
- **BFSDefender**: Uses BFS to reach the nearest visible invader. Falls back to patrolling
  the home-side boundary column when no invader is visible.

**Training value:** Harder baseline than reflex agents. The attacker takes globally
optimal food routes (not just greedy one-step), and the defender never "misses" a
chase due to local feature evaluation errors. Good for testing whether the Q-agent
has learned to genuinely outsmart opponents, not just exploit reflex weaknesses.
