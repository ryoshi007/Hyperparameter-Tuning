import numpy as np
import random
from sklearn.model_selection import cross_validate
from tqdm import tqdm
from sklearn.base import clone
import time


def objective_function(model, params, scoring, X, y, cv_times, n_jobs=-1):
    """
    Evaluates a machine learning model's performance using cross-validation.

    Parameters:
    - model: A scikit-learn compatible model object.
    - params: A dictionary of parameters to be set on the model.
    - scoring: A dictionary representing the scoring metrics to be used for evaluation.
    - X: The input data for the model.
    - y: The output data the model tries to predict.
    - cv_times: An integer specifying the number of cross-validation splits.
    - n_jobs: An optional integer specifying the number of jobs to run in parallel for cross_validate. Default is -1, which means using all processors.

    Returns:
    A dictionary containing the mean scores of the specified metrics across all cross-validation folds. These metrics include:
    - mean_accuracy: The mean accuracy across the cross-validation folds.
    - mean_precision: The mean precision across the cross-validation folds.
    - mean_recall: The mean recall across the cross-validation folds.
    - mean_f1: The mean F1 score across the cross-validation folds.
    - mean_roc_auc: The mean ROC AUC score across the cross-validation folds.

    This function utilizes `clone` to create a fresh copy of the model for each cross-validation fold, ensuring that each fold is evaluated independently. It then sets the specified parameters to the model using `set_params`. The `cross_validate` function from scikit-learn is used to perform the cross-validation, which evaluates the model on the specified scoring metrics.

    Example of use:
    - objective_function(model, {'solver': 'saga', 'penalty': 'l2'}, {'accuracy': make_scorer(accuracy_score),'f1_score': make_scorer(f1_score_score, average='macro')}, X_train, y_train, cv_times=5, n_jobs=4)
    """
    cloned_model = clone(model)
    cloned_model.set_params(**params)
    results = cross_validate(cloned_model, X, y, cv=cv_times, scoring=scoring, n_jobs=n_jobs)
    return {
        'mean_accuracy': np.mean(results['test_accuracy']),
        'mean_precision': np.mean(results['test_precision']),
        'mean_recall': np.mean(results['test_recall']),
        'mean_f1': np.mean(results['test_f1_score']),
        'mean_roc_auc': np.mean(results['test_roc_auc'])
    }



def get_random_params(param_space, current_params, model_type):
    """
    Generates a new set of parameters by randomly selecting values from a given parameter space.

    Parameters:
    - param_space: A dictionary representing the parameter space. Keys are parameter names, and values are lists of possible values for each parameter.
    - current_params: A dictionary of the current parameter values.
    - model_type: A string indicating the type of model ('logistic_regression' or 'svc').

    Returns:
    A dictionary containing a new set of parameters randomly chosen from the parameter space, while respecting model-specific constraints.

    Example of use:
    - param_space = {'solver': ['lbfgs', 'saga'], 'penalty': ['l2', 'elasticnet'], 'C': [1.0, 10.0]}
    - current_params = {'solver': 'lbfgs', 'penalty': 'l2', 'C': 1.0}
    - new_params = get_random_params(param_space, current_params, 'logistic_regression')
    """
    next_params = current_params.copy()

    if model_type == 'logistic_regression':
        # Logistic Regression specific logic
        next_params['solver'] = random.choice(param_space.get('solver', [None]))
        if 'penalty' in param_space:
            penalty_options = {
                'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'],
                'newton-cg': ['l2', None], 'newton-cholesky': ['l2', None],
                'sag': ['l2', None], 'saga': ['elasticnet', 'l1', 'l2', None]
            }
            next_params['penalty'] = random.choice(penalty_options[next_params['solver']])

            if next_params['penalty'] == 'elasticnet' and next_params['solver'] == 'saga':
                next_params['l1_ratio'] = random.choice(param_space.get('l1_ratio', [None]))
            else:
                next_params.pop('l1_ratio', None)
            if next_params['penalty'] is not None:
                next_params['C'] = random.choice(param_space.get('C', [1.0]))
            else:
                next_params.pop('C', None)

    elif model_type == 'svc':
        # SVC specific logic
        next_params['kernel'] = random.choice(param_space.get('kernel', [None]))
        if next_params['kernel'] == 'poly':
            next_params['degree'] = random.choice(param_space.get('degree', [None]))
        if next_params['kernel'] in ['rbf', 'poly', 'sigmoid']:
            next_params['gamma'] = random.choice(param_space.get('gamma', [None]))

    # Update other parameters common to all models
    for key in param_space:
        if model_type == 'logistic_regression' and key in ['solver', 'penalty', 'l1_ratio', 'C']:
            continue
        if model_type == 'svc' and key in ['kernel', 'degree', 'gamma']:
            continue
        next_params[key] = random.choice(param_space[key])

    return next_params


def validate_params(params, param_space, model_type):
    """
    Validates and adjusts the given parameters to ensure compatibility with the specified model type.

    This function checks the provided parameters against the parameter space and model constraints, particularly for logistic regression and SVC models. It adjusts the parameters to valid values if they are not compatible or not in the parameter space.

    Parameters:
    - params: A dictionary of parameters to be validated.
    - param_space: A dictionary representing the parameter space. Keys are parameter names, and values are lists of possible values for each parameter.
    - model_type: A string indicating the type of model ('logistic_regression' or 'svc').

    Returns:
    A dictionary containing validated and possibly adjusted parameters.

    The function ensures that selected parameters are valid for the specified model. If not, a new value will be chosen to correct it. The function returns a dictionary of valid parameters, ensuring that they are within the provided parameter space and comply with model-specific constraints.

    Example of use:
    - params = {'solver': 'lbfgs', 'penalty': 'l1', 'C': 10}
    - param_space = {'solver': ['lbfgs', 'saga'], 'penalty': ['l2', 'elasticnet'], 'C': [1.0, 10.0]}
    - validated_params = validate_params(params, param_space, 'logistic_regression')
    """
    valid_params = params.copy()

    if model_type == 'logistic_regression':
        # Adjust for logistic regression constraints
        solver = valid_params.get('solver', None)
        penalty_options = {
            'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'],
            'newton-cg': ['l2', None], 'newton-cholesky': ['l2', None],
            'sag': ['l2', None], 'saga': ['elasticnet', 'l1', 'l2', None]
        }

        if solver in penalty_options:
            valid_penalty = penalty_options[solver]
            if valid_params.get('penalty') not in valid_penalty:
                valid_params['penalty'] = random.choice(valid_penalty)

            if valid_params['penalty'] == 'elasticnet' and solver == 'saga':
                # Update l1_ratio only if it's not already set or if it's invalid
                if 'l1_ratio' not in valid_params or valid_params['l1_ratio'] not in param_space.get('l1_ratio', []):
                    valid_params['l1_ratio'] = random.choice(param_space.get('l1_ratio', [None]))
            else:
                valid_params.pop('l1_ratio', None)

            if valid_params['penalty'] is not None:
                # Update C only if it's not already set or if it's invalid
                if 'C' not in valid_params or valid_params['C'] not in param_space.get('C', []):
                    valid_params['C'] = random.choice(param_space.get('C', [1.0]))
            else:
                valid_params.pop('C', None)

    # SVC adjustments
    elif model_type == 'svc':
        kernel = valid_params.get('kernel', None)

        if kernel == 'poly':
            # Update 'degree' only if not set or invalid
            if 'degree' not in valid_params or valid_params['degree'] not in param_space.get('degree', []):
                valid_params['degree'] = random.choice(param_space.get('degree', [None]))
        else:
            valid_params.pop('degree', None)

        if kernel in ['rbf', 'poly', 'sigmoid']:
            # Update 'gamma' only if not set or invalid
            if 'gamma' not in valid_params or valid_params['gamma'] not in param_space.get('gamma', []):
                valid_params['gamma'] = random.choice(param_space.get('gamma', [None]))
        else:
            valid_params.pop('gamma', None)

    return valid_params


def crossover(parent1, parent2):
    """
    Performs a crossover operation between two parent parameter sets.

    This function creates a new child parameter set by combining parameters from two parent sets. The crossover process selects each parameter value from either parent based on a random chance, effectively mixing the parameters from both parents.

    Parameters:
    - parent1: A dictionary representing the first set of parameters.
    - parent2: A dictionary representing the second set of parameters.

    Returns:
    A new dictionary representing the child parameter set. This set is a combination of parameters from both parents.

    The function identifies the base parent as the one with the greater number of parameters. It then iterates through each parameter in the base parent. For each parameter, there is a 50% chance to select the value from 'parent1'. If the parameter is not in 'parent1' or is not selected, the function takes the value from 'parent2', or retains the value from the base parent if it's not present in 'parent2'.

    Example of use:
    - parent1 = {'solver': 'sag', 'C': 1}
    - parent2 = {'solver': 'saga', 'C': 2}
    - child = crossover(parent1, parent2)
    """
    base_parent = parent1 if len(parent1) >= len(parent2) else parent2
    
    child = {}
    for param in base_parent:
        child[param] = parent1[param] if param in parent1 and random.random() < 0.5 else parent2.get(param, base_parent[param])
    return child


def mutate(chromosome, mutation_rate, param_space):
    """
    Performs a mutation operation on a given set of parameters (chromosome).

    This function iterates through each parameter in the chromosome and, with a probability defined by the mutation rate, randomly alters the parameter's value based on the provided parameter space.

    Parameters:
    - chromosome: A dictionary representing a set of parameters (chromosome) to be mutated.
    - mutation_rate: A float representing the probability of mutation for each parameter.
    - param_space: A dictionary representing the parameter space. Keys are parameter names, and values are lists of possible values for each parameter.

    Returns:
    The mutated chromosome as a dictionary. This chromosome will have some parameters altered randomly based on the mutation rate.

    The function goes through each parameter in the chromosome. For each parameter, it checks against a randomly generated number to see if it falls below the mutation rate. If it does, the parameter's value is randomly selected from the corresponding list in the parameter space.

    Example of use:
    - chromosome = {'C': 1.0, 'kernel': 'rbf'}
    - mutation_rate = 0.1
    - param_space = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    - mutated_chromosome = mutate(chromosome, mutation_rate, param_space)
    """
    for param in chromosome.keys():
        if random.random() < mutation_rate:
            chromosome[param] = random.choice(param_space[param])
    return chromosome


def create_initial_population(pop_size, param_space, model_type):
    """
    Generates an initial population of parameter sets for genetic algorithm-based optimization.

    This function creates a list of random parameter sets (chromosomes) based on the provided parameter space and model type. Each parameter set in the population is generated to be potentially suitable for the specified model type.

    Parameters:
    - pop_size: An integer specifying the size of the population to generate.
    - param_space: A dictionary representing the parameter space. Keys are parameter names, and values are lists of possible values for each parameter.
    - model_type: A string indicating the type of model (e.g., 'logistic_regression', 'svc') for which the parameters are being optimized.

    Returns:
    A list of dictionaries, where each dictionary represents a unique set of model parameters (a chromosome). The size of the list is equal to `pop_size`.

    The function iterates `pop_size` times, each time calling `get_random_params` to generate a random set of parameters that are valid for the specified model type. These parameter sets form the initial population for further genetic algorithm-based optimization processes.

    Example of use:
    - param_space = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    - initial_population = create_initial_population(10, param_space, 'svc')
    """
    population = []
    for _ in range(pop_size):
        params = get_random_params(param_space, {}, model_type)
        population.append(params)
    return population


def update_rates(iteration, fit_avg, fit_max, sel_rate, mut_rate):
    """
    Dynamically updates the selection and mutation rates based on the current iteration and fitness metrics.

    This function adjusts the selection and mutation rates during the genetic algorithm process. The rates are updated based on the iteration number and the ratio of average fitness to maximum fitness of the population.

    Parameters:
    - iteration: An integer representing the current iteration number in the genetic algorithm process.
    - fit_avg: A float representing the average fitness score of the current population.
    - fit_max: A float representing the maximum fitness score in the current population.
    - sel_rate: A float representing the current selection rate.
    - mut_rate: A float representing the current mutation rate.

    Returns:
    A tuple containing the updated selection rate and mutation rate.

    Every 4 iterations, the function increases the mutation rate and decreases the selection rate by a factor of 25%. For other iterations, the function adjusts the rates based on the ratio of average to maximum fitness. The selection rate is increased, and the mutation rate is decreased proportionally to this ratio. The updated rates are constrained to be between 0 and 1.

    Example of use:
    - iteration = 5
    - fit_avg = 0.6
    - fit_max = 0.8
    - sel_rate = 0.5
    - mut_rate = 0.1
    - updated_sel_rate, updated_mut_rate = update_rates(iteration, fit_avg, fit_max, sel_rate, mut_rate)
    """
    if iteration % 4 == 0:
        update_factor = 0.25
        mut_rate *= (1 + update_factor)
        sel_rate *= (1 - update_factor)
    else:
        ratio = fit_avg / fit_max
        sel_rate += 0.1 * ratio
        mut_rate -= 0.1 * ratio
        
    mut_rate = min(max(mut_rate, 0), 1)
    sel_rate = min(max(sel_rate, 0), 1)
    
    return sel_rate, mut_rate


def genetic_algorithm(X, y, param_space, model, model_type, scoring, pop_size, max_iterations, no_improve_limit, sel_rate,
                      mut_rate, cv_times=3, n_jobs=-1, seed=42):
    """
    Executes a genetic algorithm for hyperparameter optimization of a machine learning model.

    This function simulates the process of natural selection to find the most effective hyperparameters for a given model. It involves creating an initial population of random parameters, evaluating their fitness, and iteratively selecting, crossing over, and mutating them over several generations.

    Parameters:
    - X: Input features for the model.
    - y: Target variable.
    - param_space: A dictionary representing the parameter space for the model.
    - model: A scikit-learn compatible model object.
    - model_type: A string indicating the type of model (e.g., 'logistic_regression', 'svc').
    - scoring: A dictionary representing the scoring metric to be used for model evaluation.
    - pop_size: The size of the population in each generation.
    - max_iterations: The maximum number of iterations (generations) for the algorithm.
    - no_improve_limit: The number of iterations to continue without improvement before stopping.
    - sel_rate: The rate of selection for breeding in each generation.
    - mut_rate: The mutation rate.
    - cv_times: The number of cross-validation splits. Default is 3.
    - n_jobs: The number of jobs to run in parallel. Default is -1.
    - seed: An integer for random seed initialization. Default is 42.

    Returns:
    - best_params: The best set of parameters found.
    - best_metrics: The evaluation metrics of the best model.
    - history: A list containing the history of the algorithm, including iteration number, best parameters, best score, best metrics, average fitness, selection rate, mutation rate, and population with fitness scores.
    - elapsed_time: The total time taken for the optimization process.

    The algorithm evaluates the fitness of each parameter set in the population using cross-validation. It then selects the best performing sets for breeding, creates new offspring through crossover and mutation, and forms a new generation. The process is repeated for a specified number of iterations or until there is no improvement in the best score for a given number of iterations.

    Example of use:
    - genetic_algorithm(X_train, y_train, {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, SVC(), 'svc', {'accuracy': make_scorer(accuracy_score),'f1_score': make_scorer(f1_score_score, average='macro')}, 10, 50, 5, 0.5, 0.1)
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    start_time = time.time()
    
    # Initialize population
    population = create_initial_population(pop_size, param_space, model_type)
    history = []

    best_score = float('-inf')
    best_params = None
    best_metrics = None
    no_improve_rounds = 0

    for iteration in tqdm(range(max_iterations), desc="GA Progress"):
        # Evaluate population (params at here also referred as chromosome)
        fitness_results = [objective_function(model, params, scoring, X, y, cv_times, n_jobs) for params in population]
        fitness = [result['mean_accuracy'] for result in fitness_results] 

        # Stopping criteria
        max_fitness = max(fitness)
        if max_fitness > best_score:
            best_score = max_fitness
            best_params = population[fitness.index(max_fitness)]
            best_metrics = fitness_results[fitness.index(max_fitness)]
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= no_improve_limit:
                break

        # Update adaptive rates
        avg_fitness = np.mean(fitness)
        sel_rate, mut_rate = update_rates(iteration + 1, avg_fitness, max_fitness, sel_rate, mut_rate)

        # Selection (random selection of parents)
        selection_count = int(sel_rate * pop_size)
        selection_count = max(2, min(selection_count, len(population)))
        parents = random.sample(population, k=selection_count)

        # Crossover and mutation
        offspring = []
        for _ in range(pop_size - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child_params = crossover(parent1, parent2)
            child_params = mutate(child_params, mut_rate, param_space)
            child_params = validate_params(child_params, param_space, model_type)
            offspring.append(child_params)

        # Create new population
        population = parents + offspring
        history.append((iteration, best_params, best_score, best_metrics, avg_fitness, sel_rate, mut_rate, list(zip(population, fitness))))
    
    elapsed_time = time.time() - start_time
                       
    return best_params, best_metrics, history, elapsed_time


def format_ga_history(history):
    """
    Formats the history of a genetic algorithm optimization process into a readable string format.

    This function processes the history data from a genetic algorithm optimization and formats each entry into a readable string. It includes details about the iteration, best parameters found, maximum and average fitness scores, and the selection and mutation rates for that iteration.

    Parameters:
    - history: A list of tuples, each representing an entry in the genetic algorithm history. Each tuple contains the iteration number, the best parameters found, the maximum fitness score, the average fitness score, the selection rate, the mutation rate, and other details.

    Returns:
    A string where each line represents a summary of an iteration in the genetic algorithm history. The string includes formatted details such as the iteration number, best parameters, maximum and average fitness scores, and the rates of selection and mutation.

    Each entry in the formatted string includes:
    - Iteration: The iteration number in the genetic algorithm process.
    - Best Parameters: The best set of parameters found in that iteration.
    - Max Fitness Score: The highest fitness score achieved, formatted to six decimal places.
    - Average Fitness Score: The average fitness score across the population, formatted to six decimal places.
    - Selection Rate: The selection rate used in that iteration, formatted to six decimal places.
    - Mutation Rate: The mutation rate used in that iteration, formatted to six decimal places.

    The function iterates over each entry in the history, formats it for readability, and then concatenates all formatted entries with newlines for clear presentation.

    Example of use:
    - history = [(0, {'C': 1.0}, 0.85, 0.65, 0.5, 0.1, [...]), ...]
    - formatted_history = format_ga_history(history)
    """
    formatted_history = []
    for entry in history:
        iteration, best_params, max_fitness, best_metrics, avg_fitness, sel_rate, mut_rate, _ = entry

        formatted_entry = (
            f"Iteration: {iteration}\n"
            f"Best Parameters: {best_params}\n"
            f"Max Fitness Score: {max_fitness:.6f}\n"
            f"Average Fitness Score: {avg_fitness:.6f}\n"
            f"Selection Rate: {sel_rate:.6f}\n"
            f"Mutation Rate: {mut_rate:.6f}\n"
        )
        formatted_history.append(formatted_entry)
    return '\n'.join(formatted_history)
                  
                       
def access_population_info(history):
    """
    Prints detailed information about the population at each iteration of the genetic algorithm.

    This function iterates through the history of a genetic algorithm process and displays detailed information about the parameter sets (chromosomes) in the population and their corresponding fitness scores for each iteration.

    Parameters:
    - history: A list of tuples, each representing an entry in the genetic algorithm history. Each tuple contains the iteration number, various metrics, and the population information, including parameter sets and their fitness scores.

    This function does not return a value. Instead, it prints the population information for each iteration directly to the console. The output includes the iteration number, and for each member of the population, the parameter set and its fitness score.

    The information printed for each iteration includes:
    - Iteration number: The current iteration of the genetic algorithm.
    - Parameters of each chromosome in the population: The parameter sets being evaluated in that iteration.
    - Fitness score of each chromosome: The fitness score (e.g., accuracy) achieved by each parameter set.

    This function is useful for understanding the evolution of the population over time and observing how the genetic algorithm explores the parameter space.

    Example of use:
    - history = [(0, best_params, max_fitness, avg_fitness, sel_rate, mut_rate, population_info), ...]
    - access_population_info(history)
    """
    for iteration, _, _, _, _, _, _, population_info in history:
        print(f"Iteration {iteration}:")
        for params, score in population_info:
            formatted_params = ', '.join([f'{k}: {v}' for k, v in params.items()])
            print(f"  Params: {formatted_params}, Score: {score:.6f}")
        print()