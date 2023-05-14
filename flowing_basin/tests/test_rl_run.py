from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLConfiguration, RLRun

config = RLConfiguration(
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
    length_episodes=24 * 4 + 3,
)
run = RLRun(
    config=config,
    instance=Instance.from_json("../data/input_example1_expanded.json")
)

# run.solve(path_agent="../data/RL_model_2023-05-14 19.28.zip")
# run.solution.to_json("../data/RL_model_2023-05-14 19.28_sol_example1.json")
# run.solve(path_agent="../data/RL_model_2023-05-14 20.10.zip")
# run.solution.to_json("../data/RL_model_2023-05-14 20.10_sol_example1.json")
run.solve(path_agent="../data/RL_model_2023-05-14 20.25.zip")
run.solution.to_json("../data/RL_model_2023-05-14 20.25_sol_example1.json")
