from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLConfiguration, RLRun
import time

NUM_STEPS_LOOKAHEAD = 16

model_datetime = "2023-10-26 01.10"
filepath_agent = f"../solutions/rl_models/RL_model_{model_datetime}.zip"
filepath_config = f"../solutions/rl_models/RL_model_{model_datetime}_config.json"
filepath_instance = f"../instances/instances_rl/instance1_expanded{NUM_STEPS_LOOKAHEAD}steps_backforth.json"
filepath_sol = f"../solutions/instance1_RLmodel_2dams_1days_time{model_datetime}.json"

config = RLConfiguration.from_json(filepath_config)
run = RLRun(
    config=config,
    instance=Instance.from_json(filepath_instance)
)

start_time = time.perf_counter()
run.solve(policy=filepath_agent)
run.solution.to_json(filepath_sol)
exec_time = time.perf_counter() - start_time
print(f"{exec_time=}")
