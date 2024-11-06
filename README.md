# Flowing Basin

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>

🔗 [[Paper - MILP & PSO](https://doi.org/10.1007/s10100-024-00934-z)] [Paper - RL] [[Dataset]()] [[MILP & PSO solution files]()] [[RL model weights]()]

**Flowing Basin** is a research project focused on the
optimization of hydropower stations having multiple reservoirs.
Unlike previous studies, we address a short-term intraday
optimization problem with the objective of maximizing revenue.
This variant of the problem is called the
Hydropower Reservoirs Intraday Economic Optimization (HRIEO) problem.

We used a variety of methods to tackle the HRIEO problem:
- **Mixed-Integer Linear Programming (MILP)**: an exact linearized optimization
model for the HRIEO problem implemented in [PuLP](https://github.com/coin-or/pulp).
- **Particle Swarm Optimization (PSO)**: a metaheuristic algorithm
adapted to the HRIEO problem and based on [PySwarms](https://github.com/ljvmiranda921/pyswarms).
- **Reinforcement Learning (RL)**: a machine learning approach
involving an agent that learns by interacting with an environment.
In this repository, the environment for the HRIEO problem is based on [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
and the agent is trained in this environment using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

The HRIEO problem, the MILP model and the PSO algorithm
are detailed in [this paper](https://doi.org/10.1007/s10100-024-00934-z).
The RL approach is explained in this paper.

This document explains how to install the dependencies
of this repository, where to find the data and input instances,
how to run the MILP and PSO methods for a specific problem instance,
and how to train and run a RL agent.

## Installation

The project is written in Python 3.12.
You can install all the dependencies of the project using the following command:
```python
pip install -r requirements.txt
```

## Available data and instances

...

## Mixed-Integer Linear Programming (MILP)

...

## Particle Swarm Optimization (PSO)

### Description of our PSO implementation

..

### Run the algorithm

..

### Solution files

..

## Reinforcement Learning (RL)

### Description of our RL implementation

..

### Train a model

..

### Install model weights

..

### Run a trained model

..

## 📚 Citation
If you find our work useful, please consider citing our papers:
- Paper detailing the HRIEO problem, the MILP model and the PSO algorithm (found [here](https://doi.org/10.1007/s10100-024-00934-z)):
    ```
    @article{Castro-Freibott2024,
      author = {Rodrigo Castro-Freibott and Carlos Garc{\'i}a-Castellano Gerbol{\'e}s and Alvaro Garc{\'i}a-S{\'a}nchez and Miguel Ortega-Mier},
      title = {MILP and PSO approaches for solving a hydropower reservoirs intraday economic optimization problem},
      journal = {Central European Journal of Operations Research},
      year = {2024},
      month = {sep},
      volume = {},
      number = {},
      pages = {},
      doi = {10.1007/s10100-024-00934-z},
      url = {https://doi.org/10.1007/s10100-024-00934-z},
      issn = {1613-9178}
    }
    ```
- Paper detailing the RL approach (found here):
    ...