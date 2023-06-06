from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLConfiguration, RLRun
import time

NUM_STEPS_LOOKAHEAD = 16

config = RLConfiguration(
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    flow_smoothing=2,
    num_prices=NUM_STEPS_LOOKAHEAD,
    num_unreg_flows=NUM_STEPS_LOOKAHEAD,
    num_incoming_flows=NUM_STEPS_LOOKAHEAD,
    length_episodes=24 * 4 + 3,
)
run = RLRun(
    config=config,
    instance=Instance.from_json(f"../data/input_example1_expanded{NUM_STEPS_LOOKAHEAD}steps.json")
)

start_time = time.perf_counter()
run.solve(path_agent="../data/RL_model_2023-06-05 14.03.zip")
run.solution.to_json("../data/RL_model_2023-06-05 14.03_sol_example1.json")
exec_time = time.perf_counter() - start_time
print(f"{exec_time=}")
