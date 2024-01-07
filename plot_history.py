import matplotlib.pyplot as plt


def plot_score_vs_iterations(history):
    """
    Plots the progression of the best score against iterations in a optimization process.

    This function creates a line plot showing how the best fitness score (e.g., accuracy) changes over each iteration of a optimization algorithm. This visualization helps in understanding the algorithm's performance and convergence behavior over time.

    Parameters:
    - history: A list of tuples, each representing an entry in the optimization algorithm history. Each tuple contains the iteration number, the best parameters of that iteration, the best fitness score, and other details.

    The function extracts the iteration numbers and the corresponding best scores from the history. It then plots these values, with iterations on the x-axis and scores on the y-axis. The plot includes grid lines for better readability and markers for each data point.

    Note:
    This function requires the 'matplotlib' library. Ensure it is installed and imported as 'plt' before calling this function.

    Example of use:
    - history = [(0, best_params, max_fitness, ...), (1, best_params, max_fitness, ...), ...]
    - plot_score_vs_iterations(history)
    """
    iterations = [entry[0] for entry in history]
    scores = [entry[2] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, scores, marker='o')
    plt.title('Score vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()


def plot_diff_vs_iterations(history):
    """
    Plots the difference in fitness scores against iterations in a genetic algorithm optimization process.

    This function creates a line plot showing the change in the average fitness score of the population compared to the maximum fitness score at each iteration. This visualization is helpful for understanding the overall diversity and convergence behavior of the algorithm over time.

    Parameters:
    - history: A list of tuples, each representing an entry in the genetic algorithm history. Each tuple contains the iteration number, the best parameters, the best fitness score, and other metrics including the average fitness score.

    The function extracts iteration numbers and the differences between the average and maximum fitness scores from the history. It then plots these values, with iterations on the x-axis and the score differences on the y-axis. The plot is styled with markers for each data point and a red line to distinguish it easily.

    Note:
    This function requires the 'matplotlib' library. Ensure it is installed and imported as 'plt' before calling this function.

    Example of use:
    - history = [(0, best_params, max_fitness, avg_fitness, ...), (1, best_params, max_fitness, avg_fitness, ...), ...]
    - plot_diff_vs_iterations(history)
    """
    iterations = [entry[0] for entry in history]
    diffs = [entry[4] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, diffs, marker='o', color='red')
    plt.title('Difference in Score vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Difference in Score')
    plt.grid(True)
    plt.show()