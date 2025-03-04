# evolution_simulation
Agent-based simulations are useful for studying emergent behavior in complex systems. In our simulation, agents navigate a 25x25 grid, search for food, expend energy based on movement and intrinsic traits (speed and size), and reproduce when energy exceeds a threshold. This experiment compares two approaches:
- Genetic Algorithm (GA) with Food-Driven Movement: Agents check neighboring cells for food and move to the one with the highest energy. If no food is found, they randomly decide to move or stay.
- Reinforcement Learning (RL) Approach: Agents have 9 options for actions (8 directions plus "stay"). The RL agent updates its policy based on food rewards and energy costs.
