# ai04_random

**Strategy:** Epsilon-greedy stochastic team.

- **EpsilonAttacker** (ε=0.25): Seeks food + avoids ghosts 75 % of turns; takes a fully random action 25 % of turns.
- **EpsilonDefender** (ε=0.45): Standard reflex defense 55 % of turns; fully random action 45 % of turns.

**Training value:** Injects unpredictability to prevent the Q-agent from overfitting to deterministic opponents.
Forces generalisation across a wide variety of game states per episode.
