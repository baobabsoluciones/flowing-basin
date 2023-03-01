from flowing_basin.core import Instance
from flowing_basin.solvers import PSO
from flowing_basin.tools import RiverBasin
import numpy as np
import pandas as pd


# Instance we want to solve
instance = Instance.from_json("../data/input_example1.json")
path_solution = "../data/output_example1.json"

# PSO object to find the solution
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
pso = PSO(
    instance=instance,
    paths_power_models=paths_power_models,
    num_particles=20,
)

# River basin object to study the solutions
river_basin = RiverBasin(
    instance=instance,
    paths_power_models=paths_power_models
)

# Test instance and river basin are correctly saved ---- #
# print(pso.instance.data)
# print(pso.solution.data)
# print(pso.river_basin.get_state())
# print(pso.num_particles)
# print(pso.num_dimensions)

# Test particle and relvars equivalence ---- #
# decisionsNVABC = np.array(
#     [
#         [
#             [0.5, 0.75, 1],
#             [0.5, 0.75, 1],
#         ],
#         [
#             [0.25, 0.5, 1],
#             [0.25, 0.5, 1],
#         ],
#     ]
# )
# padding = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_num_time_steps() - decisionsNVABC.shape[0])])
# decisionsNVABC_all_periods = np.concatenate([decisionsNVABC, padding])
# print("relvars:", decisionsNVABC_all_periods)
# swarm = pso.relvars_to_swarm(decisionsNVABC_all_periods)
# print("relvars -> swarm:", swarm)
# print("swarm -> relvars:", pso.swarm_to_relvars(swarm))
# print("first particle:", swarm[0])
# print("first particle -> relvars:", pso.particle_to_relvar(swarm[0]))

# Test objective function ---- #
# print(pso.objective_function(swarm))

# Original solution taken in data ---- #
# path_training_data = "../data/rl_training_data/training_data.pickle"
# df = pd.read_pickle(path_training_data)
# start, _ = instance.get_start_end_datetimes()
# initial_row = df.index[df["datetime"] == start].tolist()[0]
# last_row = initial_row + instance.get_num_time_steps() - 1
# decisions = [
#     [dam1_flow, dam2_flow]
#     for dam1_flow, dam2_flow in zip(
#         df["dam1_flow"].loc[initial_row:last_row],
#         df["dam2_flow"].loc[initial_row:last_row],
#     )
# ]
# print("original solution:", decisions)
# income = river_basin.deep_update_flows(decisions)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print("optimal solution history:", river_basin.history)
# river_basin.plot_history()
# print("original solution income:", income)

# Optimal solution found by PSO ---- #
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
status = pso.solve(options, num_iters=100)
print("status:", status)
solution_inconsistencies = pso.solution.check_schema()
if solution_inconsistencies:
    print("inconsistencies with schema:", solution_inconsistencies)
print("optimal solution:", pso.solution.data)
print("optimal income:", pso.get_objective())
pso.solution.to_json(path_solution)
