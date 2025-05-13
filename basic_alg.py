import random
import numpy as np

def magic_constant(n):
    return n * (n**2 + 1) // 2

def to_matrix(individual, n):
    return np.array(individual).reshape((n, n))

def fitness(individual, n):
    mtx = to_matrix(individual, n)
    target = magic_constant(n)
    penalty = 0

    # Row and column penalties
    for i in range(n):
        penalty += abs(np.sum(mtx[i, :]) - target)
        penalty += abs(np.sum(mtx[:, i]) - target)

    # Diagonals
    penalty += abs(np.sum(np.diag(mtx)) - target)
    penalty += abs(np.sum(np.diag(np.fliplr(mtx))) - target)

    # Penalize duplicates
    if len(set(individual)) != n ** 2:
        penalty += 1000

    return -penalty  # Higher is better

def create_individual(n):
    individual = list(range(1, n**2 + 1))
    random.shuffle(individual)
    return individual

def mutate(individual):
    # Occasionally do a stronger mutation (3 swaps)
    swaps = 3 if random.random() < 0.1 else 1
    for _ in range(swaps):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    fill_values = [x for x in parent2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[idx]
            idx += 1
    return child

def select(population, fitnesses, k=3):
    selected = random.choices(list(zip(population, fitnesses)), k=k)
    return max(selected, key=lambda x: x[1])[0]

def genetic_algorithm(n, population_size=100, generations=5000, mutation_rate=0.2, stagnation_limit=500):
    population = [create_individual(n) for _ in range(population_size)]
    best_solution = None
    best_score = float('-inf')
    stagnation_counter = 0

    for gen in range(generations):
        fitnesses = [fitness(ind, n) for ind in population]
        gen_best_score = max(fitnesses)
        gen_best_individual = population[fitnesses.index(gen_best_score)]

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_solution = gen_best_individual
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if best_score == 0:
            print(f"Perfect magic square found at generation {gen}")
            break

        if gen % 100 == 0:
            print(f"Generation {gen}: best score = {-best_score}")

        # Stagnation reset
        if stagnation_counter >= stagnation_limit:
            print(f"Resetting population due to stagnation at generation {gen}")
            population = [create_individual(n) for _ in range(population_size)]
            stagnation_counter = 0
            continue

        # New generation with elitism
        next_gen = [best_solution.copy()]
        while len(next_gen) < population_size:
            p1 = select(population, fitnesses)
            p2 = select(population, fitnesses)
            while p1 == p2:
                p2 = select(population, fitnesses)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(child)
            next_gen.append(child)

        population = next_gen

    return best_solution, -best_score

if __name__ == "__main__":
    N = 5 # Try N = 5 for more challenge
    solution, score = genetic_algorithm(N)
    print("Final Score:", score)
    print("Magic Square:")
    print(to_matrix(solution, N))
