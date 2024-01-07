from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.base import clone
from tqdm import tqdm
import numpy as np
import time
import itertools
from joblib import Parallel, delayed


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


def calculate_grid_size(model_type=None, param_grid=None):
    """
    Calculates the number of parameter combinations in a parameter grid for specific machine learning models.

    Parameters:
    - model_type: An optional string specifying the type of model. Accepted values are 'logistic_regression' and 'svc'. If not specified or set to any other value, the function calculates the size for a general parameter grid.
    - param_grid: The parameter grid whose size is to be calculated. It should be a list of dictionaries for logistic regression and SVC, or an iterable (like a ParameterGrid object) for other model types.

    Returns:
    An integer representing the total number of parameter combinations in the provided parameter grid.

    The function checks if the model type is either logistic regression or SVC. If so, it returns the length of the `param_grid` list directly, assuming that the grid is already in the form of a list of dictionaries. For other model types, it converts the `param_grid` into a list and then returns its length.

    Example of use:
    - calculate_grid_size('logistic_regression', [{'solver': 'lbfgs', 'C': 1.0}, {'solver': 'saga', 'C': 10.0}])
    - calculate_grid_size(None, ParameterGrid({'kernel': ['rbf'], 'C': [1.0, 10.0]}))
    """
    if model_type in ['logistic_regression', 'svc']:
        return len(param_grid)
    else:
        return len(list(param_grid))


def grid_search(model, param_grid, X, y, scoring, seed, cv_times=3, n_jobs=-1, parallelize=False):
    """
    Performs a grid search to find the best parameters for a given machine learning model.

    This function systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. The function offers both parallel and sequential execution modes.

    Parameters:
    - model: A scikit-learn compatible model object.
    - param_grid: A list of dictionaries, each representing a different combination of parameters to try.
    - X: The input features for the model.
    - y: The target variable.
    - scoring: A dictionary representing the scoring metric to be used for model evaluation.
    - seed: An integer used to seed the random number generator for reproducibility.
    - cv_times: An optional integer specifying the number of cross-validation splits. Default is 3.
    - n_jobs: An optional integer specifying the number of jobs to run in parallel for cross_validate. Default is -1, which means using all processors.
    - parallelize: A boolean indicating whether to run the grid search in parallel. Default is False.

    Returns:
    - best_params: A dictionary of the parameters that yielded the best score.
    - best_metrics: A dictionary containing the evaluation metrics of the best model.
    - history: A list of tuples, each containing the iteration number, parameters, mean cross-validation score, and metrics of each parameter combination.
    - elapsed_time: The total time taken to perform the grid search.

    The function first sets the seed for reproducibility. It then iterates over each combination of parameters in the `param_grid`. For each combination, it uses the `objective_function` to evaluate the model's performance. The function keeps track of the best performing parameters and their scores. If `parallelize` is True, the function uses parallel computing to speed up the grid search. The function returns the best parameters, their corresponding metrics, a history of all evaluations, and the total elapsed time.

    Example of use:
    - grid_search(model, [{'C': 1.0}, {'C': 10.0}], X_train, y_train, {'accuracy': make_scorer(accuracy_score),'f1_score': make_scorer(f1_score_score, average='macro')}, 42, cv_times=5, n_jobs=4, parallelize=True)
    """
    best_score = -np.inf
    best_params = None
    best_metrics = None
    history = []
    
    
    np.random.seed(seed)
    start_time = time.time()

    if parallelize:
        # Parallel execution
        with tqdm(total=len(param_grid), desc='Grid Search Progress') as progress_bar:
            def update(*args):
                progress_bar.update()

            results = Parallel(n_jobs=n_jobs)(delayed(objective_function)(model, params, scoring, X, y, cv_times, n_jobs) for params in param_grid)

            for i, metrics in enumerate(results):
                mean_cv_score = metrics['mean_accuracy']

                history.append((i, param_grid[i], mean_cv_score, metrics))

                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = param_grid[i]
                    best_metrics = metrics
                update()
    else:
        # Sequential execution
        for i, params in enumerate(tqdm(param_grid, desc='Grid Search Progress')):
            metrics = objective_function(model, params, scoring, X, y, cv_times, n_jobs)
            mean_cv_score = metrics['mean_accuracy']

            history.append((i, params, mean_cv_score, metrics))

            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = params
                best_metrics = metrics

    elapsed_time = time.time() - start_time

    return best_params, best_metrics, history, elapsed_time


def gs_format_history(history):
    """
    Formats the history of a grid search into a readable string format.

    This function takes the history of parameter evaluations from a grid search and formats each entry into a readable string. This includes the iteration index, the parameters used in that iteration, and the mean cross-validation score achieved.

    Parameters:
    - history: A list of tuples generated by a grid search function. Each tuple contains the iteration index, the parameters used, the mean cross-validation score, and additional metrics.

    Returns:
    A single string where each line represents one iteration's summary in the grid search history. The string includes formatted details like the iteration index, used parameters, and the score.

    Each entry in the formatted string includes:
    - Iteration index: The index of the iteration in the grid search.
    - Parameters: The parameters used in that iteration.
    - Score: The mean cross-validation score achieved with those parameters, formatted to six decimal places.

    The function iterates over each entry in the history, formats it, and then joins all formatted entries with newlines for clear readability.

    Example of use:
    - history = [(0, {'C': 1.0}, 0.85, {...}), (1, {'C': 10.0}, 0.87, {...})]
    - formatted_history = gs_format_history(history)
    """
    formatted_history = []
    for i, (iteration_index, params, mean_cv_score, metrics) in enumerate(history):
        formatted_entry = f"Iteration: {iteration_index}\nParameters: {params}\nScore: {mean_cv_score:.6f}\n"
        formatted_history.append(formatted_entry)

    return '\n'.join(formatted_history)