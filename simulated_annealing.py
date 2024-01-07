import numpy as np
import random
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone
from tqdm import tqdm
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


def simulated_annealing(model, param_space, X, y, model_type, n_jobs, scoring, max_iter=100, initial_temp=100, cooling_rate=0.95,
                        cv_times=3, seed=42):
    """
    Performs parameter optimization using the Simulated Annealing technique.

    This function applies the Simulated Annealing algorithm for optimizing hyperparameters of a given machine learning model. It starts with an initial configuration of parameters and iteratively explores the parameter space, guided by the principles of simulated annealing, to find an optimal set of parameters.

    Parameters:
    - model: A scikit-learn compatible model object.
    - param_space: A dictionary defining the parameter space.
    - X: The input features for the model.
    - y: The target variable.
    - model_type: A string indicating the type of model ('logistic_regression', 'svc', etc.).
    - n_jobs: The number of jobs to run in parallel during model evaluation.
    - scoring: A dictionary representing the scoring metric to be used for model evaluation.
    - max_iter: The maximum number of iterations for the annealing process. Default is 100.
    - initial_temp: The initial temperature for the annealing process. Default is 100.
    - cooling_rate: The rate at which the temperature decreases. Default is 0.95.
    - cv_times: The number of cross-validation splits. Default is 3.
    - seed: An integer for random seed initialization. Default is 42.

    Returns:
    - best_params: The best set of parameters found.
    - best_metrics: The evaluation metrics of the best model.
    - history: A list containing the history of evaluations, including iteration number, parameters, mean accuracy, reason for selection or rejection, temperature difference, temperature, and metrics.
    - elapsed_time: The total time taken for the optimization process.

    The function initializes with random parameters, then iteratively generates new parameter sets and evaluates them. If a new set of parameters performs better, or if it performs worse but is accepted by the algorithm's probabilistic criterion, it becomes the current parameter set. The history of iterations, including the reasons for accepting or rejecting each new set of parameters, is recorded. The temperature is gradually decreased according to the cooling rate.

    Example of use:
    - simulated_annealing(model, {'C': [0.1, 1, 10]}, X_train, y_train, 'logistic_regression', 4, {'accuracy': make_scorer(accuracy_score),'f1_score': make_scorer(f1_score_score, average='macro')})
    """
    np.random.seed(seed)
    random.seed(seed)
    initial_params = {k: random.choice(v) for k, v in param_space.items()}
    current_params = get_random_params(param_space, initial_params, model_type)
    current_metrics = objective_function(model, current_params, scoring, X, y, cv_times, n_jobs)

    best_params = current_params
    best_metrics = current_metrics

    history = [(0, current_params, current_metrics['mean_accuracy'], 'Initial configuration', 0, initial_temp, current_metrics)]
    temp = initial_temp

    start_time = time.time()
    for i in tqdm(range(1, max_iter + 1), desc='Simulated Annealing Progress'):
        next_params = get_random_params(param_space, current_params, model_type)
        metrics = objective_function(model, next_params, scoring, X, y, cv_times, n_jobs)

        diff = metrics['mean_accuracy'] - best_metrics['mean_accuracy']
        prob = np.exp(diff / temp)

        if diff > 0 or random.uniform(0, 1) < prob:
            current_params = next_params
            current_metrics = metrics
            if metrics['mean_accuracy'] > best_metrics['mean_accuracy']:
                best_params = current_params
                best_metrics = metrics
                history.append((i, next_params, metrics['mean_accuracy'], 'Improvement and accepted', diff, temp, metrics))
            else:
                history.append((i, next_params, metrics['mean_accuracy'], 'No improvement and accepted', diff, temp, metrics))
        else:
            history.append((i, next_params, metrics['mean_accuracy'], 'No improvement and rejected', diff, temp, metrics))

        temp *= cooling_rate

    elapsed_time = time.time() - start_time

    return best_params, best_metrics, history, elapsed_time


def sa_format_history(history):
    """
    Formats the history of a simulated annealing optimization process into a readable string format.

    This function takes the history of parameter evaluations from a simulated annealing process and formats each entry into a readable string. It includes details about the iteration, the parameters evaluated, the score achieved, the status of the iteration, the score difference, and the temperature at that iteration.

    Parameters:
    - history: A list of tuples, each representing an entry in the simulated annealing history. Each tuple contains the iteration number, evaluated parameters, the mean score achieved, the status (improvement, acceptance, or rejection), the score difference from the previous iteration, and the temperature.

    Returns:
    A string where each line represents a summary of an iteration in the simulated annealing history. The string includes formatted details such as the iteration number, parameters used, score, status, score difference, and temperature.

    Each entry in the formatted string includes:
    - Iteration: The iteration number in the simulated annealing process.
    - Parameters: The parameters evaluated in that iteration.
    - Score: The mean score achieved, formatted to six decimal places.
    - Status: The status of the iteration, indicating whether there was an improvement, and if the new parameters were accepted or rejected.
    - Score Difference: The difference in score compared to the previous iteration, formatted to six decimal places.
    - Temp: The temperature at that iteration.

    The function iterates over each entry in the history, formats it for readability, and then concatenates all formatted entries with newlines for clear presentation.

    Example of use:
    - history = [(0, {'C': 1.0}, 0.85, 'Initial configuration', 0, 100, {...}), (1, {'C': 10.0}, 0.87, 'Improvement and accepted', 0.02, 95, {...})]
    - formatted_history = sa_format_history(history)
    """
    formatted_history = []
    for entry in history:
        iteration, params, score, status, diff, temp, _ = entry
        formatted_entry = f"Iteration: {iteration}\nParameters: {params}\nScore: {score:.6f}\nStatus: {status}\nScore Difference: {diff:.6f}\nTemp: {temp}\n"
        formatted_history.append(formatted_entry)
    return '\n'.join(formatted_history)
