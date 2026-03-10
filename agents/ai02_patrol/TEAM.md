# ai02_patrol

**Strategy:** Strong boundary patrol defender + very timid/scared offensive agent.

- **PatrolAgent**: Walks a fixed set of positions along the home-side boundary column.
  Immediately abandons the patrol route and chases the nearest visible invader.
- **TimidAttacker**: Offensive agent that flees home the moment any active ghost comes within
  5 steps. Only moves forward when the coast is clear.

**Training value:** Tests that your Q-agent learns to score against a hard wall of defense,
and that it can exploit the harmless TimidAttacker on the other side.
