"""
Hyperparameter tuning with RL Zoo with one-step discrete actions

Algorithms to tune:
- A2C
- PPO
(SAC does not allow discrete actions)

Environments in which to tune these algorithms:
- The real (G0) and simplified (G1) environments.
- The one-step discrete actions (A31, A32, and A33).
- Only the normal observation (O231), normal reward (R1), and normal training configuration (T0),
since they should not influence on the performance of the hyperparameters.
"""

import os
from itertools import product

NUM_TRIALS = 100
NUM_TIMESTEPS_PER_TRIAL = 9_900  # 1/10th of training T0
NUM_PARALLEL_JOBS = -1  # This will set it to the system's CPU count
EVAL_EPISODES = 10  # Same as training T0
CONTINUOUS_ACTIONS = {"A1", "A113"}

# Possible values: --algo {a2c,ddpg,dqn,ppo,sac,td3,ars,qrdqn,tqc,trpo,ppo_lstm}
algorithms = ["a2c", "ppo"]
generals = ["G0", "G1"]
actions = ["A31", "A32", "A33"]
observation = "O231"
reward = "R1"
training = "T0"

for algorithm, general, action in product(algorithms, generals, actions):

    env_id = f"{action}{general}{observation}{reward}{training}"
    print(f"Tuning {algorithm} in environment {env_id}...")
    instruction = (
        f"python rl_zoo3/train.py --algo {algorithm} --env RLEnvironment-{env_id}"
        f" -n {NUM_TIMESTEPS_PER_TRIAL} -optimize --n-trials {NUM_TRIALS} --eval-episodes {EVAL_EPISODES}"
        f" --n-jobs {NUM_PARALLEL_JOBS}"
        f" --conf-file default_hyperparams/{algorithm}.yml"
    )
    if action in CONTINUOUS_ACTIONS:
        instruction += " --continuous-actions"
    os.system(instruction)
