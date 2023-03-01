from flowing_basin.core import Instance
from flowing_basin.solvers import PSOFlowVariations, PSOFlows
from flowing_basin.tools import RiverBasin
import pandas as pd

# Instance we want to solve
instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

# PSO object to find the solution
use_variations = False
if use_variations:
    print("---- SOLVING PROBLEM WITH PSO FLOW VARIATIONS ----")
    path_solution = "../data/output_example1_variations.json"
    pso = PSOFlowVariations(
        instance=instance,
        paths_power_models=paths_power_models,
        num_particles=20,
    )
else:
    print("---- SOLVING PROBLEM WITH PSO FLOWS ----")
    path_solution = "../data/output_example1_flows.json"
    pso = PSOFlows(
        instance=instance,
        paths_power_models=paths_power_models,
        num_particles=20,
    )

# River basin object to study the solutions
river_basin = RiverBasin(
    instance=instance,
    paths_power_models=paths_power_models
)

# Original solution taken in data ---- #
path_training_data = "../data/rl_training_data/training_data.pickle"
df = pd.read_pickle(path_training_data)
start, _ = instance.get_start_end_datetimes()
initial_row = df.index[df["datetime"] == start].tolist()[0]
last_row = initial_row + instance.get_num_time_steps() - 1
decisions = [
    [dam1_flow, dam2_flow]
    for dam1_flow, dam2_flow in zip(
        df["dam1_flow"].loc[initial_row:last_row],
        df["dam2_flow"].loc[initial_row:last_row],
    )
]

# Study original solution ---- #
print("original solution:", decisions)
income = river_basin.deep_update_flows(decisions)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("optimal solution history:", river_basin.history)
river_basin.plot_history()
print("original solution income:", income)

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

# Study optimal solution ---- #
opt_flows = pso.solution.to_nestedlist()
print("optimal solution equivalent flows:", opt_flows)
river_basin.reset()
opt_income = river_basin.deep_update_flows(opt_flows)
print("original income calculated with equivalent flows:", opt_income)
print("optimal solution history:\n", river_basin.history)
river_basin.plot_history()
