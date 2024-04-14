This is a fork of RL Zoo for personal use within this project.

Original repository: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master

Modifications done to the code:
- `import rl_zoo3` was replaced with `import flowing_basin.rl_zoo.rl_zoo3` in all files.
- The `train.py` file was modified to allow registering a river-basin RLEnvironment on-the-fly.
- The `exp_manager.py` file was modified to allow any RLEnvironment to use
the same default parameters for every algorithm.
- The `hyperparameters_opt.py` file was modified to automatically enable gSDE when 
an additional argument 'continuous_actions' is True (option `--continuous-actions`).
In addition, `ortho_init = True` was set to match StableBaselines3's default value.
