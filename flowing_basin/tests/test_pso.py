from flowing_basin.core import Instance
from flowing_basin.solvers import PSOConfiguration, PSOFlowVariations, PSOFlows
from flowing_basin.tools import RiverBasin
import pandas as pd

# Instance we want to solve
instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

# PSO object to find the solution
path_solution = "../data/output_example1.json"
path_history_plot = "../data/output_example1.png"
use_variations = True
config = PSOConfiguration(
    num_particles=20,
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
# river_basin.plot_history()
print("original solution income:", income)

# Optimal solution found by PSO ---- #
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
status = pso.solve(options, num_iters=100)
print("status:", status)
solution_inconsistencies = pso.solution.check_schema()
if solution_inconsistencies:
    print("inconsistencies with schema:", solution_inconsistencies)
print("optimal solution:", pso.solution.data)
print("optimal objective function value:", pso.get_objective())
pso.save_solution(path_solution)
pso.save_history_plot(path_history_plot)
