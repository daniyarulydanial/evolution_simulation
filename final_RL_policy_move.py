import random
import copy
import csv
import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
WORLD_SIZE = 25         # 25x25 grid
NUM_CREATURES = 20      # Population size per epoch
NUM_FOOD = 80
FOOD_LIFESPAN = 1       # Food respawns every generation
MUTATION_RATE = 0.1
REPRODUCTION_THRESHOLD = 100
GENERATIONS_PER_EPOCH = 50    # Number of generations in one epoch
NUM_EPOCHS = 10             # Total number of epochs

# RL-specific constant
RL_LEARNING_RATE = 0.2

# Food energy values and movement directions
FOOD_TYPES = {1: 25, 2: 35, 3: 45}

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

# Visualization settings
BACKGROUND_COLOR = (1, 1, 1)  # White background
CREATURE_BASE_SIZE = 60       # Base size for creature visualization
CREATURE_SIZE_INCREMENT = 30  # Extra size per creature size level
FOOD_BASE_SIZE = 20           # Base size for food visualization
FOOD_SIZE_INCREMENT = 10      # Extra size per food type
CREATURE_COLORS = {
    1: (1.0, 0.6, 0.6),
    2: (1.0, 0.0, 0.0),
    3: (0.6, 0.0, 0.0)
}
FOOD_COLORS = {
    1: (0, 1, 0.5),
    2: (0, 0.8, 0.3),
    3: (0, 0.6, 0.2)
}

def mutate_trait(trait, mutation_rate=MUTATION_RATE):
    """Mutate a trait value (e.g., speed or size) to one of the other possible values {1,2,3}."""
    if random.random() < mutation_rate:
        possible_values = [t for t in [1, 2, 3] if t != trait]
        return random.choice(possible_values)
    return trait

def mutate_policy(policy, mutation_rate=MUTATION_RATE, noise_std=0.05):
    """Return a mutated copy of an RL policy (a list of probabilities)."""
    new_policy = []
    for p in policy:
        if random.random() < mutation_rate:
            new_policy.append(max(0.001, min(1, p + random.gauss(0, noise_std))))
        else:
            new_policy.append(p)
    total = sum(new_policy)
    return [p / total for p in new_policy]

class Creature:
    next_id = 0  # Unique ID counter for creatures

    def __init__(self, x, y, speed, size, seed=None):
        self.x = x
        self.y = y
        self.speed = speed    # Trait: 1, 2, or 3
        self.size = size      # Trait: 1, 2, or 3
        self.energy = self.max_energy()
        self.id = Creature.next_id
        Creature.next_id += 1
        self.offspring_count = 0  # For logging reproduction events
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)
        self.rng = random.Random(seed)
        # Initialize a uniform RL policy over 9 actions (8 directions + stay)
        self.policy = [1/9] * 9

    def max_energy(self):
        return 150 + (self.size - 1) * 50

    def update_policy(self, action_index, reward):
        """
        Update the RL policy immediately for the chosen action.
        A high reward (from eating food) increases the probability of that action.
        """
        new_policy = []
        for i in range(len(self.policy)):
            if i == action_index:
                new_val = self.policy[i] + RL_LEARNING_RATE * reward * (1 - self.policy[i])
            else:
                new_val = self.policy[i] - RL_LEARNING_RATE * reward * self.policy[i]
            new_policy.append(max(0.001, new_val))
        total = sum(new_policy)
        self.policy = [p / total for p in new_policy]

    def move(self, food_map):
        """
        For each movement step (up to self.speed):
          - Scan adjacent cells (8 neighbors) and the current cell for food.
          - Compute effective action probabilities by combining the RL policy and the scanned food values.
          - Choose an action based on these effective probabilities.
          - Execute the chosen action (move or stay) and, if food is present in the target cell, collect it.
          - Update the RL policy with the immediate reward (which is the food actually collected).
        After all steps, subtract the movement cost from energy.
        Energy increases only through actual food consumption.
        """
        total_food_collected = 0
        num_moves = 0

        # Constant to scale scanned food information in decision making.
        SCAN_FACTOR = 0.25

        for _ in range(self.speed):
            # Scan 8 neighboring cells and the current cell.
            scanned_values = []
            for dx, dy in DIRECTIONS:
                new_x = self.x + dx
                new_y = self.y + dy
                if 0 <= new_x < WORLD_SIZE and 0 <= new_y < WORLD_SIZE:
                    cell_food = food_map.get((new_x, new_y), [])
                    scanned_values.append(sum(FOOD_TYPES[ft] for ft in cell_food))
                else:
                    scanned_values.append(0)
            # Scan the current cell (for the "stay" action)
            current_cell_food = food_map.get((self.x, self.y), [])
            scanned_values.append(sum(FOOD_TYPES[ft] for ft in current_cell_food))

            # Combine RL policy with scanning info.
            # Effective probability = policy[i] + SCAN_FACTOR * (scanned food value)
            effective_prob = [self.policy[i] + SCAN_FACTOR * scanned_values[i] for i in range(9)]
            total_prob = sum(effective_prob)
            effective_prob = [p / total_prob for p in effective_prob]

            # Choose an action based on the effective probabilities.
            action_index = self.rng.choices(range(9), weights=effective_prob)[0]
            immediate_reward = 0

            if action_index < 8:
                dx, dy = DIRECTIONS[action_index]
                new_x = max(0, min(WORLD_SIZE - 1, self.x + dx))
                new_y = max(0, min(WORLD_SIZE - 1, self.y + dy))
                self.x, self.y = new_x, new_y
                num_moves += 1
                if (self.x, self.y) in food_map:
                    food_value = sum(FOOD_TYPES[ft] for ft in food_map[(self.x, self.y)])
                    immediate_reward += food_value
                    del food_map[(self.x, self.y)]
            else:
                # Stay action.
                if (self.x, self.y) in food_map:
                    food_value = sum(FOOD_TYPES[ft] for ft in food_map[(self.x, self.y)])
                    immediate_reward += food_value
                    del food_map[(self.x, self.y)]

            # Update the policy based on the immediate reward from food collection.
            self.update_policy(action_index, immediate_reward)
            total_food_collected += immediate_reward

        # Calculate movement cost as before.
        baseline = 10 * self.size
        max_cost = (self.speed + self.size) * 10
        movement_cost = baseline + round((max_cost - baseline) * (num_moves / self.speed))
        overall_change = total_food_collected - movement_cost
        self.energy += overall_change

    def reproduce(self):
        """
        Attempt reproduction if energy is high enough.
        If successful, create an offspring with 70% of its max energy and a mutated copy of the parent's RL policy.
        The parent's offspring_count is incremented.
        """
        reproduction_cost = 20 * (self.speed + self.size)
        if self.energy >= REPRODUCTION_THRESHOLD + (self.size - 1) * 50:
            new_speed = mutate_trait(self.speed)
            new_size = mutate_trait(self.size)
            new_seed = int.from_bytes(os.urandom(4), 'big')
            offset = self.rng.choice(DIRECTIONS + [(0, 0)])
            offspring_x = max(0, min(WORLD_SIZE - 1, self.x + offset[0]))
            offspring_y = max(0, min(WORLD_SIZE - 1, self.y + offset[1]))
            offspring = Creature(offspring_x, offspring_y, new_speed, new_size, seed=new_seed)
            # Offspring inherit a mutated copy of the parent's RL policy.
            offspring.policy = mutate_policy(self.policy)
            offspring.energy = round(0.7 * offspring.max_energy())
            self.energy -= reproduction_cost
            self.offspring_count += 1
            return offspring
        return None

class World:
    def __init__(self, population=None, seed=None):
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)
        self.rng = random.Random(seed)
        if population is None:
            Creature.next_id = 0
            self.creatures = [Creature(
                                self.rng.randint(0, WORLD_SIZE - 1),
                                self.rng.randint(0, WORLD_SIZE - 1),
                                self.rng.randint(1, 3),
                                self.rng.randint(1, 3),
                                seed=self.rng.randint(0, 2**32 - 1)
                              ) for _ in range(NUM_CREATURES)]
        else:
            self.creatures = population[:]  # Copy provided population.
        self.food_map = {}
        self.food_timer = FOOD_LIFESPAN
        self.spawn_food()

    def spawn_food(self):
        self.food_map.clear()
        for _ in range(NUM_FOOD):
            pos = (self.rng.randint(0, WORLD_SIZE - 1), self.rng.randint(0, WORLD_SIZE - 1))
            food_type = self.rng.choice([1, 2, 3])
            if pos in self.food_map:
                self.food_map[pos].append(food_type)
            else:
                self.food_map[pos] = [food_type]
        self.food_timer = FOOD_LIFESPAN

    def update(self):
        """
        For each generation:
          - Each creature moves (using the modified RL-based move).
          - Each creature attempts reproduction.
          - Creatures with energy <= 0 are removed.
        Returns reproduction and death counts.
        """
        initial_count = len(self.creatures)
        new_creatures = []
        reproduction_count = 0
        for creature in self.creatures:
            creature.move(self.food_map)
            offspring = creature.reproduce()
            if offspring:
                new_creatures.append(offspring)
                reproduction_count += 1
        survivors = [c for c in self.creatures if c.energy > 0]
        death_count = initial_count - len(survivors)
        self.creatures = survivors + new_creatures
        self.food_timer -= 1
        if self.food_timer <= 0:
            self.spawn_food()
        return reproduction_count, death_count

    def stats(self, epoch, generation, reproduction_count, death_count):
        """
        Compute overall statistics (population size, energy, trait counts, etc.) for logging.
        """
        speed_counts = {1: 0, 2: 0, 3: 0}
        size_counts = {1: 0, 2: 0, 3: 0}
        energies = [c.energy for c in self.creatures]
        offspring_counts = [c.offspring_count for c in self.creatures]
        total_creatures = len(self.creatures)
        total_energy = sum(energies)
        avg_energy = total_energy / total_creatures if total_creatures else 0
        min_energy = min(energies) if energies else 0
        max_energy = max(energies) if energies else 0
        std_energy = math.sqrt(sum((e - avg_energy) ** 2 for e in energies) / total_creatures) if total_creatures else 0
        avg_offspring = sum(offspring_counts) / total_creatures if total_creatures else 0
        for c in self.creatures:
            speed_counts[c.speed] += 1
            size_counts[c.size] += 1
        stats_text = (f"Epoch {epoch}, Gen {generation} | Creatures: {total_creatures} | "
                      f"Avg Energy: {avg_energy:.2f} | Min: {min_energy} | Max: {max_energy} | Std: {std_energy:.2f} | "
                      f"Avg Offspring: {avg_offspring:.2f} | Repro: {reproduction_count} | Death: {death_count} | "
                      f"Speed: {speed_counts[1]}/{speed_counts[2]}/{speed_counts[3]} | "
                      f"Size: {size_counts[1]}/{size_counts[2]}/{size_counts[3]}")
        print(stats_text)
        return [epoch, generation, total_creatures, avg_energy, min_energy, max_energy, std_energy,
                speed_counts[1], speed_counts[2], speed_counts[3],
                size_counts[1], size_counts[2], size_counts[3],
                avg_offspring, reproduction_count, death_count]

    def get_snapshot(self, epoch, generation):
        """Return a snapshot of the current state for animation."""
        snapshot = {
            "epoch": epoch,
            "generation": generation,
            "food_map": copy.deepcopy(self.food_map),
            "creatures": [{"id": c.id, "x": c.x, "y": c.y,
                           "speed": c.speed, "size": c.size, "energy": c.energy,
                           "offspring_count": c.offspring_count}
                          for c in self.creatures],
            "stats_title": f"Epoch {epoch} Gen {generation} | Creatures: {len(self.creatures)}"
        }
        return snapshot

    def display(self):
        """Display the current world state."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_xticks(range(WORLD_SIZE + 1))
        ax.set_yticks(range(WORLD_SIZE + 1))
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        for (x, y), food_list in self.food_map.items():
            for food_type in food_list:
                ax.scatter(x + 0.5, y + 0.5,
                           s=FOOD_BASE_SIZE + FOOD_SIZE_INCREMENT * food_type,
                           c=[FOOD_COLORS[food_type]], edgecolors='black')
        for creature in self.creatures:
            ax.scatter(creature.x + 0.5, creature.y + 0.5,
                       s=CREATURE_BASE_SIZE + CREATURE_SIZE_INCREMENT * creature.size,
                       c=[CREATURE_COLORS[creature.speed]], edgecolors='black')
        plt.show()

def run_epoch(population, epoch, generations=GENERATIONS_PER_EPOCH):
    """Run one epoch (a block of generations) and return the final population, stats, and snapshots."""
    results = []
    history = []
    world = World(population)
    for gen in range(1, generations + 1):
        repro_count, death_count = world.update()
        stats_line = world.stats(epoch, gen, repro_count, death_count)
        results.append(stats_line)
        snapshot = world.get_snapshot(epoch, gen)
        history.append(snapshot)
    return world.creatures, results, history

def generate_new_population(best_policy):
    """
    Create a new population of NUM_CREATURES where each creature inherits
    a mutated copy of the best policy from the previous epoch.
    """
    new_population = []
    for _ in range(NUM_CREATURES):
        x = random.randint(0, WORLD_SIZE - 1)
        y = random.randint(0, WORLD_SIZE - 1)
        speed = random.randint(1, 3)
        size = random.randint(1, 3)
        creature = Creature(x, y, speed, size)
        creature.policy = mutate_policy(best_policy)
        creature.energy = creature.max_energy()
        creature.offspring_count = 0
        new_population.append(creature)
    return new_population

def run_simulation():
    # Initialize the initial population randomly.
    population = [Creature(random.randint(0, WORLD_SIZE - 1),
                             random.randint(0, WORLD_SIZE - 1),
                             random.randint(1, 3),
                             random.randint(1, 3))
                  for _ in range(NUM_CREATURES)]
    overall_results = []
    overall_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Reset energy, offspring_count, and randomize positions for all creatures at the start of each epoch.
        for creature in population:
            creature.energy = creature.max_energy()
            creature.offspring_count = 0
            creature.x = random.randint(0, WORLD_SIZE - 1)
            creature.y = random.randint(0, WORLD_SIZE - 1)
        population, epoch_results, epoch_history = run_epoch(population, epoch)
        overall_results.extend(epoch_results)
        overall_history.extend(epoch_history)

        # Check if the population is empty.
        if population:
            best_creature = max(population, key=lambda c: c.offspring_count)
            print(f"Epoch {epoch} best creature: ID {best_creature.id}, Offspring Count: {best_creature.offspring_count}")
            best_policy = best_creature.policy
        else:
            # If extinct, reinitialize with a default (uniform) policy.
            print(f"Epoch {epoch}: Population extinct. Reinitializing population with default policy.")
            best_policy = [1/9] * 9

        # Generate a new population for the next epoch using the best (or default) policy.
        population = generate_new_population(best_policy)

    # Save overall statistics to a CSV file (code unchanged) ...
    with open("rl_overall_stats.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Generation", "Total Creatures", "Average Energy",
                         "Min Energy", "Max Energy", "Std Energy",
                         "Speed 1", "Speed 2", "Speed 3",
                         "Size 1", "Size 2", "Size 3",
                         "Avg Offspring Count", "Reproduction Count", "Death Count"])
        writer.writerows(overall_results)
    print("Overall statistics saved to rl_overall_stats.csv.")

    return population, overall_history, overall_results

def animate_simulation(history):
    """Animate the simulation snapshots."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_xticks(range(WORLD_SIZE + 1))
    ax.set_yticks(range(WORLD_SIZE + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    def update_frame(i):
        ax.clear()
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_xticks(range(WORLD_SIZE + 1))
        ax.set_yticks(range(WORLD_SIZE + 1))
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        snapshot = history[i]

        # Plot food.
        for (x, y), food_list in snapshot["food_map"].items():
            for food_type in food_list:
                ax.scatter(x + 0.5, y + 0.5,
                           s=FOOD_BASE_SIZE + FOOD_SIZE_INCREMENT * food_type,
                           c=[FOOD_COLORS[food_type]], edgecolors='black')
        # Plot creatures.
        for creature in snapshot["creatures"]:
            ax.scatter(creature["x"] + 0.5, creature["y"] + 0.5,
                       s=CREATURE_BASE_SIZE + CREATURE_SIZE_INCREMENT * creature["size"],
                       c=[CREATURE_COLORS[creature["speed"]]], edgecolors='black')
        # Annotate cell counts.
        cell_counts = {}
        for creature in snapshot["creatures"]:
            pos = (creature["x"], creature["y"])
            cell_counts[pos] = cell_counts.get(pos, 0) + 1
        for (x, y), count in cell_counts.items():
            ax.text(x + 0.3, y + 0.7, str(count),
                    color='blue', fontsize=8, weight='bold')
        ax.text(0.5, 1.02, snapshot["stats_title"],
                transform=ax.transAxes,
                fontsize=12, ha="center", va="bottom", color="black")

    anim = animation.FuncAnimation(fig, update_frame, frames=len(history), interval=500, repeat=False)
    plt.show()

if __name__ == "__main__":
    final_population, history, results = run_simulation()
    # Display the final state.
    world = World(final_population)
    world.display()
    # Optionally animate the simulation.
    animate_simulation(history)
