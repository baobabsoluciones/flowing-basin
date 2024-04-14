"""
Parse the results from RL Zoo's hyperparameter tuning
"""

import os

# Analyze the generated .pkl file
FILENAME = "logs/a2c/report_RLEnvironment-A1G0O231R1T0_2-trials-5-tpe-median_1713051371.pkl"
components = FILENAME.split("/")
algorithm = components[1]
env_id = components[2].split("_")[1].split("-")[1]
os.system(
    f"python parse_study.py -i {FILENAME} "
    f"--print-n-best-trials 1 --save-n-best-hyperparameters 1 -f logs/best_hyperparams_{algorithm}_{env_id}"
)
