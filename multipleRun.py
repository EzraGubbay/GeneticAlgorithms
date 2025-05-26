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
    sizes = list(range(4, 6))  # N=3 to 5
    algorithms = ["darwinian", "lamarckian", "classic"]

    success_rates = {alg: {} for alg in algorithms}
    avg_eval_calls = {alg: {} for alg in algorithms}
    avg_solution_gen = {alg: {} for alg in algorithms}

    for alg in algorithms:
        print(f"\n🚀 Testing algorithm: {alg}")
        for N in sizes:
            successes = 0
            total_eval_calls = 0
            gen_when_found_list = []

            print(f"  🔍 N={N}")
            for run in range(1, runs + 1):
                print(f"    ▶ Run {run}/{runs}")
                result_gen = genetic_algorithm(N, strategy=alg)
                snapshots = []
                try:
                    while True:
                        snapshot = next(result_gen)
                        snapshots.append(snapshot)
                except StopIteration as stop:
                    if stop.value:  # final return
                        result = stop.value
                    elif snapshots:
                        result = snapshots[-1]
                    else:
                        print("❌ Generator yielded nothing.")
                        continue

                solution = result['best_solution']
                score = result['best_score']
                eval_calls = result['eval_calls']
                gen_found = result['gen_found']

                total_eval_calls += eval_calls
                valid = is_valid_magic_square(solution, N)
                if valid:
                    successes += 1
                    gen_when_found_list.append(gen_found if gen_found is not None else result['generation'])

            # Store stats
            success_rates[alg][N] = successes / runs
            avg_eval_calls[alg][N] = total_eval_calls / runs
            avg_solution_gen[alg][N] = (
                sum(gen_when_found_list) / len(gen_when_found_list) if gen_when_found_list else None
            )

    # --- Plot Success Rate ---
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        rates = [success_rates[alg].get(N, 0) for N in sizes]
        plt.plot(sizes, rates, marker='o', label=alg.capitalize())

    plt.xlabel("N (Size of Magic Square)")
    plt.ylabel("Success Rate")
    plt.title("Success Rate of Genetic Algorithms per Strategy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.xticks(sizes)
    plt.show()

    # --- Print Statistics ---
    print("\n📊 Summary:")
    for alg in algorithms:
        print(f"\n🔧 {alg.capitalize()}:")
        for N in sizes:
            print(f"  N={N}:")
            print(f"    - Success Rate: {success_rates[alg][N]*100:.1f}%")
            print(f"    - Avg Eval Calls: {avg_eval_calls[alg][N]:.1f}")
            if avg_solution_gen[alg][N] is not None:
                print(f"    - Avg Gen to Solution: {avg_solution_gen[alg][N]:.1f}")
            else:
                print(f"    - Avg Gen to Solution: ❌ No valid solutions")

if __name__ == "__main__":
    run_multiple_tests()
