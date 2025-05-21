import random
import numpy as np

# Calculate the "magic constant" that each row, column, and diagonal must sum to
def magic_constant(n):
    return n * (n**2 + 1) // 2

# Convert a flat list of n^2 numbers into an n x n matrix
def to_matrix(individual, n):
    return np.array(individual).reshape((n, n))

# Check if a given square is a valid magic square
def is_valid_magic_square(square, n):
    mtx = to_matrix(square, n)
    target = magic_constant(n)

    for i in range(n):
        if np.sum(mtx[i, :]) != target or np.sum(mtx[:, i]) != target:
            return False
    if np.sum(np.diag(mtx)) != target or np.sum(np.diag(np.fliplr(mtx))) != target:
        return False
    if sorted(mtx.flatten()) != list(range(1, n ** 2 + 1)):
        return False
    return True

# Evaluate how "good" a given individual is by computing its penalty (lower is better)
def fitness(individual, n):
    mtx = to_matrix(individual, n)         # Convert to matrix
    target = magic_constant(n)             # Expected sum for rows/columns/diagonals
    penalty = 0                            # Initialize penalty score

    # Penalize deviations from the target sum in rows and columns
    for i in range(n):
        penalty += abs(np.sum(mtx[i, :]) - target)    # Row i
        penalty += abs(np.sum(mtx[:, i]) - target)    # Column i

    # Penalize deviations in both diagonals
    penalty += abs(np.sum(np.diag(mtx)) - target)             # Main diagonal
    penalty += abs(np.sum(np.diag(np.fliplr(mtx))) - target)  # Anti-diagonal

    # Penalize if not all values from 1 to n^2 appear (i.e., duplicates or missing values)
    if len(set(individual)) != n ** 2:
        penalty += 1000
        print("Duplicate or missing values detected.")

    return -penalty  # Return negative penalty: higher fitness is better

# Generate a random individual (a shuffled list of numbers from 1 to n^2)
def create_individual(n):
    individual = list(range(1, n**2 + 1))
    random.shuffle(individual)
    # print(to_matrix(individual, n))
    # print("fitness:")
    # print(fitness(individual, n))
    return individual

# Randomly mutate an individual by swapping two or more elements
def mutate(individual, n):
    # Occasionally do a stronger mutation (3 swaps instead of 1)
    swaps = 3 if random.random() < 0.1 else 1
    for _ in range(swaps):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]


# Perform crossover between two parents using Order Crossover (OX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    # Copy a segment from parent1
    child[start:end] = parent1[start:end]

    # Fill the rest from parent2 in order
    fill_values = [x for x in parent2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[idx]
            idx += 1
    return child

# Select the best individual from a random sample of k individuals (tournament selection)
def select(population, fitnesses, k=3):
    selected = random.choices(list(zip(population, fitnesses)), k=k)
    return max(selected, key=lambda x: x[1])[0]

# Main genetic algorithm loop
def genetic_algorithm(n, population_size=100, generations=1000, stagnation_limit=500):
    best_gen1_solution = None
    eval_calls = 0
    best_scores = []
    avg_scores = []
    gen_found = None

   
    mutation_rate = 0.2 # Probability of mutation
    # Create initial random population
    population = []
    for i in range(population_size):
        #print("individual:", i)
        population.append(create_individual(n))

    best_solution = None
    best_score = float('-inf')
    stagnation_counter = 0

    for gen in range(generations):
        # Evaluate fitness for entire population for statistical data
        fitnesses = [fitness(ind, n) for ind in population]
        eval_calls += len(population) # Count evaluations
        
        # Find the best individual in current generation
        gen_best_score = max(fitnesses)
        gen_best_individual = population[fitnesses.index(gen_best_score)]
        avg_score = sum(fitnesses) / len(fitnesses)
        
        best_scores.append(-gen_best_score)  # Convert back to penalty for clearer plotting
        avg_scores.append(-avg_score)


        # store the best solution from the first generation
        if gen == 0:
            best_gen1_solution = gen_best_individual.copy()
        if gen_best_score == 0 and gen_found is None:
            gen_found = gen  # Record when perfect solution was found
        # Update best-ever solution
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_solution = gen_best_individual
            stagnation_counter = 0  # Reset stagnation counter
        else:
            stagnation_counter += 1

        # Stop early if perfect solution is found
        if best_score == 0 and is_valid_magic_square(best_solution, n):
            # print(f"✅ Valid magic square found at generation {gen} this printing from function")
            # #print the square
            # print(to_matrix(best_solution, n))
            # #print the correct statistics
            # print(f"Final Score: {-best_score}")
            # print(f"Final Evaluation Calls: {eval_calls}")
            
            return {
                'generation': gen,
                'best_solution': best_solution,
                'best_score': -best_score,
                'best_scores': best_scores,
                'avg_scores': avg_scores,
                'eval_calls': eval_calls,
                'gen_found': gen,
                'best_gen1': best_gen1_solution
            }
            break

        # Print progress every 100 generations
        if gen % 100 == 0:
            print(f"Generation {gen}: best score = {-best_score}")

        # return the best solution every 500 generations
        if gen % 500 == 0:
            yield {
                'generation': gen,
                'best_solution': best_solution,
                'best_score': -best_score,
                'best_scores': best_scores,
                'avg_scores': avg_scores,
                'eval_calls': eval_calls,
                'gen_found': gen_found,
                'best_gen1': best_gen1_solution
            }

        # If there's no improvement for many generations, reset the population
        if stagnation_counter >= stagnation_limit:
            print(f"Resetting population due to stagnation at generation {gen}")
            population = [create_individual(n) for _ in range(population_size)]
            stagnation_counter = 0
            continue

       # Create next generation with elitism: preserve the best individual
        next_gen = [best_solution.copy()]
        while len(next_gen) < population_size:
            p1 = select(population, fitnesses)
            p2 = select(population, fitnesses)
            while p1 == p2:
                p2 = select(population, fitnesses)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(child, n)
            next_gen.append(child)
        
      


        population = next_gen  # Move to next generation

      # Return best solution and its (positive) score
    return {
    'generation': generations,
    'best_solution': best_solution,
    'best_score': -best_score,
    'best_scores': best_scores,
    'avg_scores': avg_scores,
    'eval_calls': eval_calls,
    'gen_found': gen_found,
    'best_gen1': best_gen1_solution
}


# Run the genetic algorithm with a given N
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 4
    print(f"\n🎯 Running genetic algorithm for N = {N}...\n")

    result_gen = genetic_algorithm(N)
    snapshots = []

    try:
        while True:
            snapshot = next(result_gen)
            print(f"📊 Generation {snapshot['generation']}")
            print(f"Best Score: {snapshot['best_score']}")
            print(f"Evaluation Calls: {snapshot['eval_calls']}")
            print(f"Generation Found: {snapshot['gen_found']}")
            print("Current Best Magic Square:")
            print(to_matrix(snapshot['best_solution'], N))
            print("-" * 40)
            snapshots.append(snapshot)
    except StopIteration as stop:
        if stop.value:  # Early return happened
            snapshots.append(stop.value)

    # Final snapshot (either last yield or early return)
    final = snapshots[-1]

    print("\n✅ Final Results:")
    print(f"Final Score: {final['best_score']}")
    print(f"Final Evaluation Calls: {final['eval_calls']}")
    print(f"Best Magic Square Found:")
    print(to_matrix(final['best_solution'], N))

    # Plotting
    plt.plot(final['best_scores'], label='Best Score')
    plt.plot(final['avg_scores'], label='Average Score')
    plt.xlabel("Generation")
    plt.ylabel("Penalty")
    plt.title(f"Magic Square GA - N={N}")
    plt.legend()
    plt.grid(True)
    plt.show()
