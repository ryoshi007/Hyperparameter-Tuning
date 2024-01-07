# Machine Learning Optimization Toolkit 🤖
## Overview
This repository contains a collection of optimization algorithms and tools designed to enhance the hyperparameter tuning process for machine learning models. The toolkit includes implementations of various optimization strategies such as Grid Search, Half Grid Search, Simulated Annealing, and Genetic Algorithm. Additionally, it features functions to visualize the optimization process, aiding in analysis and decision-making.  

## Contents 📕
1. **Grid Search**: A comprehensive search over specified parameter values for a given algorithm.
2. **Halving Grid Search**: A more efficient version of Grid Search that progressively halves the search space.
3. **Simulated Annealing**: A probabilistic technique for approximating the global optimum of a given function.
4. **Genetic Algorithm**: An evolutionary algorithm inspired by natural selection processes, used for solving both constrained and unconstrained optimization problems.
5. **Plotting Functions**: Tools to plot the progress of these algorithms to visualize their performance and convergence.  

## Getting Started 🔨
### Prerequisites
Python 3.x  
Scikit-learn (for model building and evaluation)  
NumPy (for numerical operations)  
Matplotlib (for plotting results)   

### Installation
Clone the repository to your local machine:

`git clone https://github.com/ryoshi007/Hyperparameter-Tuning.git`  

### Usage
Import the desired module and use the functions to optimize your machine learning models. Each module contains functions that can be directly applied to a dataset and a model. For example, to use the Grid Search:

`from grid_search import grid_search`

`best_params = grid_search(model, param_grid, X_train, y_train, scoring_metric)`

Refer to the streamlit website below for detailed usage instructions.  

## Contributing 🧑‍🤝‍🧑
Contributions to enhance this toolkit are welcome. Please feel free to fork the repository and submit pull requests.


