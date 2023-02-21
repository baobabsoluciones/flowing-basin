from flowing_basin.solvers.rl import Training
# from datetime import datetime

paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
training = Training(
    length_episodes=24*4,
    paths_power_models=paths_power_models,
    path_constants="../data/rl_training_data/constants.json",
    path_training_data="../data/rl_training_data/training_data.pickle",
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
    # initial_row=datetime.strptime("2021-04-03 00:00", "%Y-%m-%d %H:%M")
)
print(training.constants)
print(training.env.instance.check())
print(training.env.instance.data)
print(training.env.get_observation())
# training.env.instance.to_json("../data/input_example3.json")
