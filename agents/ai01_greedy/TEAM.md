# ai01_greedy

**Strategy:** Dual greedy offense — both agents rush for food with zero ghost fear.

- **GreedyAttacker**: Ignores ghosts entirely; minimizes distance to nearest food every step.
- **CapsuleAttacker**: Also ignores ghosts; prioritizes capsules heavily before going for food.

Both agents return home only when ≤2 food remain.

**Training value:** Tests that your Q-agent learns to intercept and punish reckless attackers.
