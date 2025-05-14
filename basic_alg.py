import random
import numpy as np

# Calculate the "magic constant" that each row, column, and diagonal must sum to
def magic_constant(n):
    return n * (n**2 + 1) // 2

# Convert a flat list of n^2 numbers into an n x n matrix
def to_matrix(individual, n):
    return np.array(individual).reshape((n, n))

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
# def mutate(individual, n):
#     if random.random() < 0.1:
#         # Strong mutation: multiple swaps
#         for _ in range(n):
#             i, j = random.sample(range(len(individual)), 2)
#             individual[i], individual[j] = individual[j], individual[i]
#     else:
#         # Weak mutation: one swap
#         i, j = random.sample(range(len(individual)), 2)
#         individual[i], individual[j] = individual[j], individual[i]

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
def genetic_algorithm(n, population_size=100, generations=5000, stagnation_limit=500):
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
        if best_score == 0:
            print(f"Perfect magic square found at generation {gen}")
            break
        

        
        




        # Print progress every 100 generations
        # if gen % 100 == 0:
        #     print(f"Generation {gen}: best score = {-best_score}")

        # If there's no improvement for many generations, reset the population
        if stagnation_counter >= stagnation_limit:
            # print(f"Resetting population due to stagnation at generation {gen}")
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
        
        # Create next generation with elitism: preserve the best individual
    #     next_gen = [best_solution.copy()]
    #     print(f"\n--- Creating Generation {gen+1} ---")
    #     print(f"Best individual carried over:\n{to_matrix(best_solution, n)}\nScore: {-best_score}\n")
        
    # # print all the detailed stuff for gen 2

    #     while len(next_gen) < population_size:
    #         p1 = select(population, fitnesses)
    #         p2 = select(population, fitnesses)
    #         while p1 == p2:
    #             p2 = select(population, fitnesses)

    #         print("Selected Parents:")
    #         print(to_matrix(p1, n))
    #         print(to_matrix(p2, n))

    #         child = crossover(p1, p2)
    #         print("Child after crossover:")
    #         print(to_matrix(child, n))

    #         if random.random() < mutation_rate:
    #             mutate(child, n)
    #             print("Child after mutation:")
    #             print(to_matrix(child, n))

    #         print("Child fitness:", fitness(child, n), "\n")
    #         next_gen.append(child)


        population = next_gen  # Move to next generation

      # Return best solution and its (positive) score
    return best_solution, -best_score, best_scores, avg_scores, eval_calls, gen_found

# Run the genetic algorithm with a given N
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 4
    result = genetic_algorithm(N)
    solution, score, best_scores, avg_scores, eval_calls, gen_found = result

    print("Final Score:", score)
    print("Magic Square:")
    print(to_matrix(solution, N))
    print("Evaluation Calls:", eval_calls)
    print("Generation found (if any):", gen_found)

    plt.plot(best_scores, label='Best Score')
    plt.plot(avg_scores, label='Average Score')
    plt.xlabel("Generation")
    plt.ylabel("Penalty")
    plt.title(f"Magic Square GA - N={N}")
    plt.legend()
    plt.grid(True)
    plt.show()
