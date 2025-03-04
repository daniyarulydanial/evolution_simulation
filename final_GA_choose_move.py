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

# Food energy values by type
FOOD_TYPES = {1: 25, 2: 35, 3: 45}

# 8 Possible Movements (dx, dy)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1), (1, 0),  (1, 1)]

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
    """
    Mutate a trait value (e.g., speed or size) to one of the other possible values {1,2,3}.
    For example, if trait is 1, it can become either 2 or 3.
    """
    if random.random() < mutation_rate:
        possible_values = [t for t in [1, 2, 3] if t != trait]
        return random.choice(possible_values)
    return trait

class Creature:
    next_id = 0  # Unique ID counter for creatures

    def __init__(self, x, y, speed, size, seed=None):
        self.x = x
        self.y = y
        self.speed = speed    # 1, 2, or 3
        self.size = size      # 1, 2, or 3
        self.energy = self.max_energy()  # Start at maximum energy.
        self.id = Creature.next_id
        Creature.next_id += 1
        self.offspring_count = 0  # Track number of offspring produced
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)
        self.rng = random.Random(seed)

    def max_energy(self):
        return 150 + (self.size - 1) * 50

    def move(self, food_map):
        """
        For each step (up to self.speed):
          - Look at adjacent cells (and current cell) for food.
          - Move toward the cell with the highest food yield; if none, move randomly.
        Then, deduct an energy cost based on the number of moves.
        """
        num_moves = 0
        possible_moves = DIRECTIONS + [(0, 0)]
        for _ in range(self.speed):
            best_move = (0, 0)
            best_yield = 0
            for dx, dy in possible_moves:
                new_x = self.x + dx
                new_y = self.y + dy
                if 0 <= new_x < WORLD_SIZE and 0 <= new_y < WORLD_SIZE:
                    cell_food = food_map.get((new_x, new_y), [])
                    food_yield = sum(FOOD_TYPES[ft] for ft in cell_food)
                    if food_yield > best_yield:
                        best_yield = food_yield
                        best_move = (dx, dy)
            chosen_move = self.rng.choice(possible_moves) if best_yield == 0 else best_move
            if chosen_move != (0, 0):
                num_moves += 1
            self.x = max(0, min(WORLD_SIZE - 1, self.x + chosen_move[0]))
            self.y = max(0, min(WORLD_SIZE - 1, self.y + chosen_move[1]))
        baseline = 10 * self.size
        max_cost = (self.speed + self.size) * 10
        energy_cost = baseline + round((max_cost - baseline) * num_moves / self.speed)
        self.energy -= energy_cost

    def eat(self, food_map):
        if (self.x, self.y) in food_map:
            for food_type in food_map[(self.x, self.y)]:
                self.energy += FOOD_TYPES[food_type]
            del food_map[(self.x, self.y)]

    def reproduce(self):
        """
        Attempt reproduction if energy is high enough.
        Reproduction cost = 15*(speed+size). If successful, create an offspring with 70% of its max energy,
        reduce the parent's energy by the cost, and increment the parent's offspring_count.
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
          - Every creature moves and eats.
          - Each creature attempts reproduction.
          - Creatures with energy <= 0 are removed.
        Returns both reproduction_count and death_count.
        """
        initial_count = len(self.creatures)
        new_creatures = []
        reproduction_count = 0
        for creature in self.creatures:
            creature.move(self.food_map)
            creature.eat(self.food_map)
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
        Compute overall statistics:
          - Population size, energy metrics (avg, min, max, std),
          - Counts for each speed and size,
          - Average offspring count per creature,
          - Reproduction and death counts for that generation.
        Returns a list suitable for CSV output.
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
        """Return a snapshot of the current state for animation (if desired)."""
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

# ---------- GA OPERATORS ----------

def tournament_selection(population, tournament_size=3):
    """Select one individual using tournament selection based on offspring_count."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda c: c.offspring_count)

def crossover(parent1, parent2):
    """Uniform crossover: randomly choose each trait from one of the two parents."""
    child_speed = parent1.speed if random.random() < 0.5 else parent2.speed
    child_size = parent1.size if random.random() < 0.5 else parent2.size
    return child_speed, child_size

def mutate_genome(speed, size, mutation_rate=MUTATION_RATE):
    """Mutate each trait independently."""
    new_speed = mutate_trait(speed, mutation_rate)
    new_size = mutate_trait(size, mutation_rate)
    return new_speed, new_size

def reproduce_population(population, new_population_size=NUM_CREATURES):
    """
    Create a new population using GA operators:
      - Selection (tournament based on offspring_count)
      - Crossover (uniform)
      - Mutation (for traits)
    """
    new_population = []
    while len(new_population) < new_population_size:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child_speed, child_size = crossover(parent1, parent2)
        child_speed, child_size = mutate_genome(child_speed, child_size)
        x = random.randint(0, WORLD_SIZE - 1)
        y = random.randint(0, WORLD_SIZE - 1)
        child = Creature(x, y, child_speed, child_size)
        child.energy = child.max_energy()
        new_population.append(child)
    return new_population

# ---------- SIMULATION FUNCTIONS ----------

def run_epoch(population, epoch, generations=GENERATIONS_PER_EPOCH):
    """
    Run one epoch (a block of generations) with the given population.
    Returns the final population, a list of overall stats (for CSV), and snapshots history.
    """
    results = []   # Overall stats for each generation
    history = []   # Snapshots for optional animation
    world = World(population)
    for gen in range(1, generations + 1):
        reproduction_count, death_count = world.update()
        stats_line = world.stats(epoch, gen, reproduction_count, death_count)
        results.append(stats_line)
        snapshot = world.get_snapshot(epoch, gen)
        history.append(snapshot)
    return world.creatures, results, history

def run_simulation():
    # Initialize the first population randomly.
    population = [Creature(
                    random.randint(0, WORLD_SIZE - 1),
                    random.randint(0, WORLD_SIZE - 1),
                    random.randint(1, 3),
                    random.randint(1, 3)
                  ) for _ in range(NUM_CREATURES)]
    overall_results = []  # Overall statistics across epochs
    overall_history = []  # Snapshots history (for animation, if desired)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"--- Starting Epoch {epoch} with population size {len(population)} ---")
        # Reset each creature's energy, offspring_count, and assign new random positions.
        for creature in population:
            creature.energy = creature.max_energy()
            creature.offspring_count = 0
            creature.x = random.randint(0, WORLD_SIZE - 1)
            creature.y = random.randint(0, WORLD_SIZE - 1)
        population, epoch_results, epoch_history = run_epoch(population, epoch)
        overall_results.extend(epoch_results)
        overall_history.extend(epoch_history)
        print(f"Epoch {epoch} finished with {len(population)} creatures.")
        # Check for extinction before reproducing.
        if population:
            population = reproduce_population(population, NUM_CREATURES)
        else:
            print(f"Epoch {epoch}: Population extinct. Reinitializing population randomly.")
            population = [Creature(
                    random.randint(0, WORLD_SIZE - 1),
                    random.randint(0, WORLD_SIZE - 1),
                    random.randint(1, 3),
                    random.randint(1, 3)
                  ) for _ in range(NUM_CREATURES)]
    # Save overall statistics to CSV.
    with open("ga_overall_stats.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Generation", "Total Creatures", "Average Energy",
                         "Min Energy", "Max Energy", "Std Energy",
                         "Speed 1", "Speed 2", "Speed 3",
                         "Size 1", "Size 2", "Size 3",
                         "Avg Offspring Count", "Reproduction Count", "Death Count"])
        writer.writerows(overall_results)
    print("Overall statistics saved to ga_overall_stats.csv.")

    return population, overall_history, overall_results

def animate_simulation(history):
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

        # Plot food locations
        for (x, y), food_list in snapshot["food_map"].items():
            for food_type in food_list:
                ax.scatter(x + 0.5, y + 0.5,
                           s=FOOD_BASE_SIZE + FOOD_SIZE_INCREMENT * food_type,
                           c=[FOOD_COLORS[food_type]], edgecolors='black')
        # Plot creatures
        for creature in snapshot["creatures"]:
            ax.scatter(creature["x"] + 0.5, creature["y"] + 0.5,
                       s=CREATURE_BASE_SIZE + CREATURE_SIZE_INCREMENT * creature["size"],
                       c=[CREATURE_COLORS[creature["speed"]]], edgecolors='black')
        # Calculate and display creature count per cell
        cell_counts = {}
        for creature in snapshot["creatures"]:
            pos = (creature["x"], creature["y"])
            cell_counts[pos] = cell_counts.get(pos, 0) + 1
        for (x, y), count in cell_counts.items():
            ax.text(x + 0.3, y + 0.7, str(count),
                    color='blue', fontsize=8, weight='bold')
        # Place header text with overall statistics at the top of the plot.
        ax.text(0.5, 1.02, snapshot["stats_title"],
                transform=ax.transAxes,
                fontsize=12, ha="center", va="bottom", color="black")
    anim = animation.FuncAnimation(fig, update_frame, frames=len(history), interval=500, repeat=False)
    plt.show()


if __name__ == "__main__":
    final_population, history, results = run_simulation()
    # Optionally, display the final state.
    world = World(final_population)
    world.display()
    # Optionally animate the simulation.
    animate_simulation(history)
