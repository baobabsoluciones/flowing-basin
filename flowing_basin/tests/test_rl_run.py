from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLConfiguration, RLRun
import time

NUM_STEPS_LOOKAHEAD = 16

model_datetime = "2023-10-11 01.18"
filepath_agent = f"../solutions/rl_models/RL_model_{model_datetime}.zip"
filepath_instance = f"../instances/instances_rl/instance1_expanded{NUM_STEPS_LOOKAHEAD}steps_backforth.json"
filepath_sol = f"../solutions/instance1_RLmodel{model_datetime}_sol.json"

config = RLConfiguration(
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    flow_smoothing=2,
    flow_smoothing_penalty=25,
    flow_smoothing_clip=False,
    action_type="exiting_flows",
    features=[
        "past_vols", "past_flows", "past_variations", "past_prices", "future_prices", "past_inflows",
        "future_inflows", "past_turbined", "past_groups", "past_powers", "past_clipped", "past_periods"
    ],
    num_steps_sight=16,
    length_episodes=24 * 4 + 3,
    do_history_updates=False,
)
run = RLRun(
    config=config,
    instance=Instance.from_json(filepath_instance)
)

start_time = time.perf_counter()
run.solve(path_agent=filepath_agent)
run.solution.to_json(filepath_sol)
exec_time = time.perf_counter() - start_time
print(f"{exec_time=}")
