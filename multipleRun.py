# this file is used to run multiple tests of the genetic algorithm for different sizes of magic squares
# and plot the success rate and other statistics

import numpy as np
import matplotlib.pyplot as plt
from basic_alg import genetic_algorithm, to_matrix, magic_constant

def is_valid_magic_square(square, n):
    mtx = to_matrix(square, n)
    target = magic_constant(n)

    for i in range(n):
        if np.sum(mtx[i, :]) != target or np.sum(mtx[:, i]) != target:
            return False
    if np.sum(np.diag(mtx)) != target or np.sum(np.diag(np.fliplr(mtx))) != target:
        return False
    flat = mtx.flatten()
    if sorted(flat) != list(range(1, n ** 2 + 1)):
        return False
    return True

def run_multiple_tests():
    runs = 10
    sizes = list(range(3, 6))
    results = {}
    avg_eval_calls = {}
    avg_solution_gen = {}

    for N in sizes:
        successes = 0
        total_eval_calls = 0
        gen_when_found_list = []

        print(f"\n🔍 Running tests for N={N}")
        for i in range(1, runs + 1):
            solution, score, _, _, eval_calls, gen_found = genetic_algorithm(N)
            total_eval_calls += eval_calls

            valid = is_valid_magic_square(solution, N)
            if valid:
                print(f"✅ Run {i}: Valid magic square found (Score: {score})")
                successes += 1
                gen_when_found_list.append(gen_found if gen_found is not None else 5000)
            else:
                print(f"❌ Run {i}: Invalid square (Score: {score})")

        results[N] = successes / runs
        avg_eval_calls[N] = total_eval_calls / runs
        avg_solution_gen[N] = (
            sum(gen_when_found_list) / len(gen_when_found_list) if gen_when_found_list else None
        )

    # --- Plot Success Rate ---
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("N (Size of Magic Square)")
    plt.ylabel("Success Rate")
    plt.title("Genetic Algorithm Success Rate per N")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xticks(sizes)
    plt.show()

    # --- Print Extra Stats ---
    print("\n📊 Additional Statistics:")
    for N in sizes:
        print(f"N={N}:")
        print(f"  - Success Rate: {results[N]*100:.1f}%")
        print(f"  - Avg Eval Calls: {avg_eval_calls[N]:.1f}")
        if avg_solution_gen[N] is not None:
            print(f"  - Avg Gen to Solution: {avg_solution_gen[N]:.1f}")
        else:
            print(f"  - Avg Gen to Solution: ❌ No valid solutions found")

run_multiple_tests()
