from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import PSOConfiguration, PSOFlowVariations, PSOFlows
from flowing_basin.tools import RiverBasin
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Instance we want to solve
instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

# PSO object to find the solution
path_solution = "../data/output_example1_PSO.json"
path_history_plot = "../data/output_example1_PSO.png"
path_obj_history_plot = "../data/output_example1_PSO.jpeg"
use_variations = True
config = PSOConfiguration(
    num_particles=200,
    max_relvar=0.5,
    keep_direction=2,
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
    volume_objectives=[59627.42324, 31010.43613642857],
)
if use_variations:
    print("---- SOLVING PROBLEM WITH PSO FLOW VARIATIONS ----")
    pso = PSOFlowVariations(
        instance=instance,
        paths_power_models=paths_power_models,
        config=config,
    )
else:
    print("---- SOLVING PROBLEM WITH PSO FLOWS ----")
    pso = PSOFlows(
        instance=instance,
        paths_power_models=paths_power_models,
        config=config,
    )

# River basin object to study the solutions
river_basin = RiverBasin(
    instance=instance,
    paths_power_models=paths_power_models
)

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
# sol = Solution.from_nestedlist(decisions, dam_ids=instance.get_ids_of_dams())
# sol.to_json("../data/output_example1_original-real-decisions.json")

# Study original solution ---- #
# sol = Solution.from_json("../data/output_example1_original-real-decisions.json")
# print("original solution's objective function values:", pso.get_objective_values(sol))
# print("original solution's full objective function value:", pso.get_objective(sol))
# print("original solution (repeat):", sol.to_nestedlist())
# decisions = sol.to_nestedlist()
# income = river_basin.deep_update_flows(decisions)
# river_basin.plot_history()
# plt.savefig("../data/output_example1_original-real-decisions.png")
# plt.show()
# print("original solution's income:", income)

# Search for best PSO parameters ---- #
# options_search = {"c1": [0.1, 5], "c2": [0.1, 5], "w": [0.3, 0.9], "k": [1, 2], "p": 1}
# # We need to put "k": [1, 2], "p": 1 for the method to work,
# # even though "k" and "p" are not used in GlobalBestPSO
# print(pso.search_best_options(options_search, num_iters_selection=30, num_iters_each_test=20))
# Result: 'c1': 2.905405139888455, 'c2': 0.4232260541405988, 'w': 0.4424113459034113 | cost=-10114.100850959443

# Optimal solution found by PSO ---- #
options = {'c1': 2.905405139888455, 'c2': 0.4232260541405988, 'w': 0.4424113459034113}
status = pso.solve(options, num_iters=2)
print("status:", status)
solution_inconsistencies = pso.solution.check_schema()
if solution_inconsistencies:
    print("inconsistencies with schema:", solution_inconsistencies)
print("optimal solution:", pso.solution.data)
print("optimal solution's objective function values:", pso.get_objective_values())
print("optimal solution's full objective function value:", pso.get_objective())
print(pso.river_basin.log)
pso.save_solution(path_solution)
pso.save_plot_history(path_history_plot)
pso.save_plot_objective_function_history(path_obj_history_plot)

# Study configuration ---- #
# print(pso.study_configuration(options))

# Study LP model solution ---- #
# flows = [[15.15, 14.95, 14.75, 13.66, 13.57, 13.37, 13.17, 12.97, 12.77, 12.57, 12.37, 12.17, 11.97, 11.77, 11.57, 11.37, 11.17, 10.97, 10.77, 10.57, 10.41, 10.61, 10.81, 11.01, 10.81, 10.61, 10.41, 10.21, 10.01, 9.81, 9.61, 9.52, 9.52, 9.52, 9.61, 9.61, 9.61, 9.44, 9.24, 9.04, 8.84, 8.64, 8.8, 9, 9.2, 9.4, 5.12, 4.98, 4.98, 4.98, 4.98, 4.98, 4.98, 4.98, 4.98, 4.78, 4.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.78, 4.98, 5.05, 5.25, 5.45, 5.65, 5.58, 5.38, 5.18, 4.98, 4.98, 5.18, 4.98, 5.18, 4.98, 5.18, 5.18, 4.98, 4.98, 4.98, 5.18, 5.38, 5.18, 4.98, 5.1, 5.3, 13.66, 0], [6.19, 6.28, 7.69, 7.88, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.85, 9.82, 9.77, 9.7, 9.61, 9.5, 9.43, 9.4, 9.41, 9.44, 7.46, 7.26, 7.13, 7.33, 7.13, 6.93, 7.03, 0.22, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.06, 5.21, 5.28, 5.48, 5.68, 5.88, 5.81, 5.89, 6.09, 6.29, 6.49, 6.69, 6.89, 7.09, 7.29, 7.39, 7.19, 7.07, 6.87, 6.67, 6.47, 6.67, 0.4, 0.2, 0, 0, 0]]
# flows = np.array(flows)
# flows = np.transpose(flows)
# flows = flows.tolist()
# print("LP model solution:", flows)
# print(len(flows))  # noqa (suppres PyCharm inspection)
# sol_lp = Solution.from_nestedlist(flows, dam_ids=instance.get_ids_of_dams())  # noqa
# print("LP model solution's objective function values:", pso.get_objective_values(sol_lp))
# print("LP model solution's full objective function value:", pso.get_objective(sol_lp))
# sol_lp.to_json("../data/output_example1_LPmodel_gap2.json")

# sol_lp = Solution.from_json("../data/output_example1_LPmodel_gap2.json")
# river_basin.reset(num_scenarios=1)
# income_lp = river_basin.deep_update_flows(sol_lp.to_nestedlist())  # noqa
# river_basin.plot_history()
# plt.savefig("../data/output_example1_LPmodel_gap2.png")
# plt.show()
# print("LP model solution's income:", income_lp)
# print(river_basin.log)

