from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import PSOConfiguration, PSO
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os

# Instance we want to solve
instance = Instance.from_json("../data/input_example3.json")

# PSO object to find the solution
config = PSOConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
    startups_penalty=50,
    limit_zones_penalty=50,
    volume_objectives={
        "dam1": 59627.42324,
        "dam2": 31010.43613642857
    },
    use_relvars=False,
    max_relvar=1,
    flow_smoothing=0,
    mode="linear"
)
pso = PSO(
    instance=instance,
    config=config,
)

# Test particle and flows equivalence ---- #
# decisionsABC = np.array(
#         [[[6.79, 8, 1], [6.58, 7, 1]], [[7.49, 9, 1], [6.73, 8.5, 1]]]
#     )
# paddingABC = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_largest_impact_horizon() - decisionsABC.shape[0])])
# decisionsABC_all_periods = np.concatenate([decisionsABC, paddingABC])
# print("ABC flows:", decisionsABC_all_periods)
# swarmABC = pso.reshape_as_swarm(decisionsABC_all_periods)
# print("ABC flows -> swarm:", swarmABC)
# print("ABC swarm -> flows:", pso.reshape_as_flows_or_relvars(swarmABC))
# print("ABC first particle:", swarmABC[0])
# print("ABC first particle -> flows:", pso.turn_into_flows(swarmABC[0].reshape(1, -1), relvars=False))
# print("ABC first particle history", pso.river_basin.history.to_string())

# Test objective function
# pso.river_basin.deep_update(pso.reshape_as_flows_or_relvars(swarmABC), relvars=False)
# print(pso.objective_function_env())
# print(pso.river_basin.get_state())

# Test particle and relvars equivalence ---- #
# decisionsVABC = np.array(
#     [
#         [
#             [0.5, -0.25, 0.25],
#             [0.5, -0.25, 0.5],
#         ],
#         [
#             [-0.25, 0.5, 0.5],
#             [-0.25, 0.5, 0.3],
#         ],
#         [
#             [-0.25, 0.5, 0.5],
#             [-0.25, 0.5, 0.3],
#         ],
#     ]
# )
# paddingVABC = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_largest_impact_horizon() - decisionsVABC.shape[0])])
# decisionsNVABC_all_periods = np.concatenate([decisionsVABC, paddingVABC])
# print("VABC relvars:", decisionsNVABC_all_periods)
# swarmVABC = pso.reshape_as_swarm(decisionsNVABC_all_periods)
# print("VABC relvars -> swarm:", swarmVABC)
# print("VABC swarm -> relvars:", pso.reshape_as_flows_or_relvars(swarmVABC))
# for i in range(swarmVABC.shape[0]):
#     print(f"VABC particle {i}:", swarmVABC[i])
#     print(f"VABC particle {i} -> relvars:", pso.reshape_as_flows_or_relvars(swarmVABC[i].reshape(1, -1)))
#     pso.river_basin.deep_update(pso.reshape_as_flows_or_relvars(swarmVABC[i].reshape(1, -1)), relvars=True)
#     print(f"VABC particle {i} -> flows:", pso.river_basin.all_past_clipped_flows)
#     print(f"VABC particle {i} history", pso.river_basin.history.to_string())

# Test objective function
# print(pso.objective_function_env(swarmVABC, relvars=True))
# print(pso.river_basin.get_state())

# Original solution taken in data ---- #
# path_training_data = "../data/rl_training_data/training_data.pickle"
# df = pd.read_pickle(path_training_data)
# start, _ = instance.get_start_end_datetimes()
# initial_row = df.index[df["datetime"] == start].tolist()[0]
# last_row = initial_row + instance.get_largest_impact_horizon() - 1
# decisions = np.array(
#     [
#         [[dam1_flow], [dam2_flow]]
#         for dam1_flow, dam2_flow in zip(
#             df["dam1_flow"].loc[initial_row:last_row],
#             df["dam2_flow"].loc[initial_row:last_row],
#         )
#     ]
# )
# sol_original = Solution.from_flows(decisions, dam_ids=instance.get_ids_of_dams())
# print(sol_original.check())
# sol_original.to_json("../data/output_example1_original-real-decisions_solution.json")
sol_original = Solution.from_json("../data/output_example1_original-real-decisions_solution.json")

# Study LP model solution ---- #
# sol_lp = Solution.from_dict(
#     {
#         "dams": [
#             {
#                 "id": "dam1",
#                 "flows": [13.66, 13.66, 13.66, 13.66, 13.66, 13.66, 13.66, 13.66, 13.66, 13.66, 13.66, 9.61, 9.61, 9.61, 9.61, 9.61, 9.61, 9.61, 9.67, 9.87, 10.07, 10.27, 10.47, 13.66, 13.66, 13.66, 13.66, 13.66, 9.99, 9.79, 9.59, 9.39, 9.37, 9.37, 9.57, 9.65, 9.61, 9.61, 9.61, 9.61, 9.61, 9.61, 9.61, 9.6, 9.4, 5.32, 5.18, 4.98, 5.01, 5.21, 5.25, 5.1, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 4.63, 4.83, 4.98, 5.08, 4.98, 5.0, 5.02, 5.04, 5.04, 5.24, 5.44, 5.64, 5.84, 5.95, 5.88, 5.68, 5.48, 5.28, 5.08, 0.0, 0.0, 0.0, 0.0, 0.0, 15.24, 0.0, 0.0, 0.0, 0.0],
#             },
#             {
#                 "id": "dam2",
#                 "flows": [6.19, 6.28, 7.46, 8.12, 9.28, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.84, 9.85, 9.85, 9.85, 9.84, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.84, 9.81, 8.89, 8.69, 4.28, 7.46, 7.26, 7.06, 6.86, 6.66, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 6.74, 8.58, 6.54, 6.74, 6.94, 7.14, 7.21, 7.25, 7.41, 7.21, 7.25, 7.41, 7.21, 7.25, 7.41, 7.21, 7.01, 6.81, 0.27, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             },
#         ]
#      }
# )
# print(sol_lp.check())
# print(sol_lp.to_flows())
# sol_lp.to_json("../data/output_example1_LPmodel_gap0_solution.json")

sol_lp = Solution.from_json("../data/output_instance3_LPmodel.json")

# Optimal solution found by PSO ---- #

# path_parent = "../data"
# dir_name = f"output_example1_PSO_{datetime.now().strftime('%Y-%m-%d %H.%M')}_mode={pso.config.mode}_k={pso.config.flow_smoothing}"
# options = {'c1': 2.905405139888455, 'c2': 0.4232260541405988, 'w': 0.4424113459034113}
# status = pso.solve(options, num_particles=200, num_iters=1000)
# print("status:", status)
# print("solver info:", pso.solver_info)

# path_parent = "../data"
# dir_name = f"output_example1_original-real-decisions_mode={pso.config.mode}_k={pso.config.flow_smoothing}"
# pso.solution = sol_original

path_parent = "../data"
dir_name = "output_example_instance3_LPModel_Linear"
pso.solution = sol_lp

print(pso.solution.check())
print("optimal solution:", pso.solution.data)
pso.river_basin.deep_update(pso.solution.to_flows(), is_relvars=False)
print("optimal solution's objective function values:", pso.objective_function_values_env())
print("optimal solution's full objective function value:", pso.objective_function_env())
print("optimal solution's full objective function value (cornflow method):", pso.get_objective())
dam1_income = pso.river_basin.history["dam1_income"].loc[0:96].sum()
dam2_income = pso.river_basin.history["dam2_income"].loc[0:98].sum()
print("optimal solution's income (from history):", dam1_income + dam2_income)
print(pso.river_basin.history.to_string())
pso.save_solution_info(path_parent=path_parent, dir_name=dir_name)
pso.river_basin.history.to_excel(os.path.join(path_parent, dir_name, "history.xlsx"))

# Search for best PSO parameters ---- #
# options_search = {"c1": [0.1, 5], "c2": [0.1, 5], "w": [0.3, 0.9], "k": [1, 2], "p": 1}
# # We need to put "k": [1, 2], "p": 1 for the method to work,
# # even though "k" and "p" are not used in GlobalBestPSO
# print(pso.search_best_options(options_search, num_iters_selection=30, num_iters_each_test=20))
# Result: 'c1': 2.905405139888455, 'c2': 0.4232260541405988, 'w': 0.4424113459034113 | cost=-10114.100850959443

# Study configuration ---- #
# print(pso.study_configuration(options))
