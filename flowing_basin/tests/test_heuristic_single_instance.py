from flowing_basin.core import Instance
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
from datetime import datetime
import matplotlib.pyplot as plt

EXAMPLE = '1'
NUM_DAMS = 2
NUM_DAYS = 1
K_PARAMETER = 2
PLOT_SOL = True
RANDOM_BIASED_FLOWS = False
PROB_BELOW_HALF = 0.15
RANDOM_BIASED_SORTING = False
COMMON_RATIO = 0.6
MAXIMIZE_FINAL_VOL = False

path_instance = f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json"
path_sol = f"../solutions/instance{EXAMPLE}_Heuristic_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

# Instance we want to solve
instance = Instance.from_json(path_instance)

# Configuration
config = HeuristicConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0,
    startups_penalty=50,
    limit_zones_penalty=0,
    volume_objectives={
        "dam1": 59627.42324,
        "dam2": 31010.43613642857,
        "dam3_dam2copy": 31010.43613642857,
        "dam4_dam2copy": 31010.43613642857,
        "dam5_dam1copy": 59627.42324,
        "dam6_dam1copy": 59627.42324,
        "dam7_dam2copy": 31010.43613642857,
        "dam8_dam1copy": 59627.42324,
    },
    flow_smoothing=K_PARAMETER,
    mode="linear",
    maximize_final_vol=MAXIMIZE_FINAL_VOL,
    random_biased_flows=RANDOM_BIASED_FLOWS,
    prob_below_half=PROB_BELOW_HALF,
    random_biased_sorting=RANDOM_BIASED_SORTING,
    common_ratio=COMMON_RATIO,
)

# Solution
heuristic = Heuristic(config=config, instance=instance)
heuristic.solve()
print("Total income:", heuristic.solution.get_objective_function())
heuristic.solution.to_json(path_sol)

# Check solution
sol_inconsistencies = heuristic.solution.check()
if sol_inconsistencies:
    raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
print("Optimal solution:", heuristic.solution.data)


# Plot simple solution graph for each dam
if PLOT_SOL:
    for dam_id in instance.get_ids_of_dams():
        fig, ax = plt.subplots()
        heuristic.solution.plot_solution_for_dam(dam_id, ax)
        plt.show()
