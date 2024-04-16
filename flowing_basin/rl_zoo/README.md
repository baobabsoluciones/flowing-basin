Hyperparameter tuning with RL Zoo
---

Relevant documentation:
- Custom environments: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/custom_env.html
- Hyperparameter tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html

The scripts in this directory (`rl_zoo`) use RL Zoo's hyperparameter optimization function
to find the best hyperparameters of StableBaselines3's RL algorithms.

Not all hyperparameters are tuned; some remain with a fixed value.
These values are read from the default_hyperparams directory,
which were set to match StableBaselines3 default values.

The original RL Zoo library is not used.
Some modifications were required, so the whole code
was copy-pasted and modified inside this directory (see `rl_zoo/rl_zoo3`).
Please refer to the README.md in here.
