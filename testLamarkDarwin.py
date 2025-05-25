import matplotlib.pyplot as plt
from basic_alg import genetic_algorithm, to_matrix  # Adjust the import as needed

def run_and_collect(strategy, N=4):
    print(f"🔬 Running strategy: {strategy}")
    result_gen = genetic_algorithm(N, strategy=strategy)
    snapshots = []
    try:
        while True:
            snapshot = next(result_gen)
            snapshots.append(snapshot)
    except StopIteration as stop:
        if stop.value:
            snapshots.append(stop.value)
    return snapshots[-1]  # final result

def plot_results(results, N):
    for strategy, result in results.items():
        generations = list(range(len(result['best_scores'])))
        plt.plot(generations, result['best_scores'], label=f"{strategy} - Best")
        plt.plot(generations, result['avg_scores'], linestyle='--', label=f"{strategy} - Avg")

    plt.title(f"Comparison of GA Strategies for N={N}")
    plt.xlabel("Generation")
    plt.ylabel("Penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"comparison_N{N}.png")
    plt.show()

if __name__ == "__main__":
    N = 4
    results = {}
    for strategy in ["classic", "darwinian", "lamarckian"]:
        results[strategy] = run_and_collect(strategy, N)

    plot_results(results, N)

    for strategy, result in results.items():
        print(f"\n✅ {strategy.upper()} Strategy Results")
        print(f"Final Score: {result['best_score']}")
        print(f"Eval Calls: {result['eval_calls']}")
        print(f"Generation Found: {result['gen_found']}")
        print("Best Magic Square:")
        print(to_matrix(result['best_solution'], N))
