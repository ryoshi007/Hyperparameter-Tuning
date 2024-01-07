from sklearn.model_selection import cross_validate, ParameterGrid
from sklearn.utils import resample
from joblib import Parallel, delayed
from sklearn.base import clone
import numpy as np
import itertools
from tqdm import tqdm 
from math import ceil
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
    
    if params.get('penalty') == None and ('C' in params or 'l1_ratio' in params):
        print("Warning: Setting penalty=None with C and l1_ratio parameters:", params)
        
        
    try:
        results = cross_validate(cloned_model, X, y, cv=cv_times, scoring=scoring, n_jobs=n_jobs)
        return {
            'mean_accuracy': np.mean(results['test_accuracy']),
            'mean_precision': np.mean(results['test_precision']),
            'mean_recall': np.mean(results['test_recall']),
            'mean_f1': np.mean(results['test_f1_score']),
            'mean_roc_auc': np.mean(results['test_roc_auc'])
        }
    except ValueError as e:
        print("Error occurred during model fitting:", e)
        # Return default values for each metric
        return {
            'mean_accuracy': 0,
            'mean_precision': 0,
            'mean_recall': 0,
            'mean_f1': 0,
            'mean_roc_auc': 0
     }


def generate_param_grid(param_space, model_type=None):
    """
    Generates a grid of parameter combinations for specific machine learning models.

    Parameters:
    - param_space: A dictionary where keys are parameter names and values are lists of parameter settings to try as values.
    - model_type: An optional string specifying the type of model. Accepted values are 'logistic_regression' and 'svc'. If not specified or set to any other value, a general parameter grid is generated without model-specific constraints.

    Returns:
    A list of dictionaries, where each dictionary represents a unique combination of parameters.

    The function first checks the model type. If it's logistic regression or support vector classifier, it will create a grid of parameter combinations based on the defined constraints. In cases where no specific model type is provided, it generates a general parameter grid based on the provided parameter space.

    Examples of use:
    - generate_param_grid({'solver': ['lbfgs', 'saga'], 'penalty': ['l2', 'elasticnet']}, 'logistic_regression')
    - generate_param_grid({'kernel': ['rbf', 'poly'], 'C': [1.0, 10.0]}, 'svc')
    """
    grid = []

    if model_type == 'logistic_regression':
        valid_combinations = {
            'lbfgs': ['l2', None],
            'liblinear': ['l1', 'l2'],
            'newton-cg': ['l2', None],
            'newton-cholesky': ['l2', None],
            'sag': ['l2', None],
            'saga': ['elasticnet', 'l1', 'l2', None]
        }

        for solver, penalty in itertools.product(param_space.get('solver', ['lbfgs']),
                                                 param_space.get('penalty', ['l2'])):
            if penalty in valid_combinations.get(solver, []):
                if penalty == 'elasticnet' and solver == 'saga':
                    for l1_ratio, C in itertools.product(param_space.get('l1_ratio', [None]),
                                                         param_space.get('C', [1.0])):
                        grid.append({'solver': solver, 'penalty': penalty, 'l1_ratio': l1_ratio, 'C': C})
                else:
                    for C in param_space.get('C', [1.0]):
                        grid.append({'solver': solver, 'penalty': penalty, 'C': C})

    elif model_type == 'svc':
        for kernel, C in itertools.product(param_space.get('kernel', ['rbf']),
                                           param_space.get('C', [1.0])):
            if kernel == 'poly':
                for degree, gamma in itertools.product(param_space.get('degree', [3]),
                                                       param_space.get('gamma', ['scale'])):
                    grid.append({'kernel': kernel, 'C': C, 'degree': degree, 'gamma': gamma})
            else:
                for gamma in param_space.get('gamma', ['scale']):
                    grid.append({'kernel': kernel, 'C': C, 'gamma': gamma})
    else:
        grid = list(ParameterGrid(param_space))

    return grid


def halving_grid_search(model, param_space, model_type, X, y, scoring, cv_times=3, n_iterations=3, n_jobs=-1, factor=2,
                       dynamic_reduction=False, seed=42):
    """
    Performs a halving grid search to optimize model parameters efficiently.

    This function implements a grid search with successive halving, a technique that reduces the computational cost by iteratively culling the number of parameter combinations. It evaluates a large number of combinations initially and then reduces them in each iteration based on their performance.

    Parameters:
    - model: A scikit-learn compatible model object.
    - param_space: A dictionary defining the parameter space.
    - model_type: A string indicating the type of model (e.g., 'logistic_regression', 'svc').
    - X: The input features for the model.
    - y: The target variable.
    - scoring: A dictionary representing the scoring metric to be used for model evaluation.
    - cv_times: The number of cross-validation splits. Default is 3.
    - n_iterations: The number of halving iterations to perform. Default is 3.
    - n_jobs: The number of jobs to run in parallel. Default is -1, which uses all processors.
    - factor: The reduction factor for culling parameter combinations. Default is 2.
    - dynamic_reduction: A boolean indicating whether to dynamically adjust the reduction factor. Default is False.
    - seed: An integer for random seed initialization. Default is 42.

    The function first initializes the random seed and generates a parameter grid. It then iteratively evaluates subsets of the parameter grid, reducing the number of parameters based on their performance in each iteration. If `dynamic_reduction` is True, the reduction factor is dynamically calculated. The function records the history of evaluations and returns the best parameters and metrics found, along with the complete history and the elapsed time.

    Example of use:
    - halving_grid_search(model, {'C': [0.1, 1, 10]}, 'svc', X_train, y_train, {'accuracy': make_scorer(accuracy_score),'f1_score': make_scorer(f1_score_score, average='macro')}, n_iterations=4, factor=3)
    """
    
    np.random.seed(seed)

    param_grid = generate_param_grid(param_space, model_type)
    n_candidates = len(param_grid)
    n_samples = len(X)
    history = []
    
    start_time = time.time()

    for iteration in tqdm(range(n_iterations), desc="Halving Grid Search"):
        iter_best_metrics = None
        iter_best_params = None
        
        # Determine the reduction factor
        if dynamic_reduction:
            remaining_iterations = n_iterations - iteration
            if remaining_iterations > 1:
                dynamic_factor = ceil(n_candidates / (2 ** (remaining_iterations - 1)))
            else:
                dynamic_factor = n_candidates
        else:
            dynamic_factor = factor

        iter_n_samples = min(n_samples, n_samples * (2 ** iteration) // (2 ** (n_iterations - 1)))
        X_iter, y_iter = X[:iter_n_samples], y[:iter_n_samples]

        # Evaluate each combination of parameters
        eval_results = Parallel(n_jobs=n_jobs)(
            delayed(objective_function)(clone(model), params, scoring, X_iter, y_iter, cv_times=cv_times, n_jobs=n_jobs)
            for params in param_grid
        )

        # Process and store the results
        for params, metrics in zip(param_grid, eval_results):
            history.append((iteration, params, metrics['mean_accuracy'], iter_n_samples, metrics))
            if iter_best_metrics is None or metrics['mean_accuracy'] > iter_best_metrics['mean_accuracy']:
                iter_best_metrics = metrics
                iter_best_params = params

        # Select the top candidates based on the factor
        mean_scores = [metrics['mean_accuracy'] for metrics in eval_results]
        num_top_candidates = max(n_candidates // dynamic_factor, 1)
        top_candidates = np.argsort(mean_scores)[-num_top_candidates:]
        param_grid = [param_grid[i] for i in top_candidates]

        # Reduce the number of candidates for the next iteration
        n_candidates = len(top_candidates)
    
    elapsed_time = time.time() - start_time

    return iter_best_params, iter_best_metrics, history, elapsed_time


def hs_format_history(history):
    """
    Formats the history of a halving grid search into a readable string format.

    This function processes the history data from a halving grid search and formats it into a readable string. Each entry includes details about the iteration, the parameters evaluated, the achieved score, and the resource count used in that iteration.

    Parameters:
    - history: A list of tuples, each representing an entry in the halving grid search history. Each tuple contains the iteration number, evaluated parameters, the mean score achieved, the number of resources used (e.g., sample size), and additional metrics.

    Returns:
    A string with each line representing a summary of an iteration in the halving grid search history. The string includes formatted details about the iteration number, parameters used, score, and resource count.

    Each entry in the formatted string includes:
    - Iteration: The iteration number in the halving grid search.
    - Parameters: The parameters evaluated in that iteration.
    - Score: The mean score achieved, formatted to six decimal places.
    - Resource Count: The number of resources (e.g., sample size) used in that iteration.

    The function iterates over each entry in the history, formats it for readability, and then joins all formatted entries with newlines.

    Example of use:
    - history = [(0, {'C': 1.0}, 0.85, 100, {...}), (1, {'C': 10.0}, 0.87, 50, {...})]
    - formatted_history = hs_format_history(history)
    """
    formatted_history = []
    for entry in history:
        iteration, params, score, resource_count, _ = entry
        formatted_entry = f"Iteration: {iteration}\nParameters: {params}\nScore: {score:.6f}\nResource Count: {resource_count}\n"
        formatted_history.append(formatted_entry)

    return '\n'.join(formatted_history)