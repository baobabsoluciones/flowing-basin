from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration
from datetime import datetime
import matplotlib.pyplot as plt

EXAMPLE = 1
NUM_DAMS = 3
NUM_DAYS = 1
PLOT_SOL = True
SAVE_SOLUTION = True
TIME_LIMIT_MINUTES = 0.25

path_instance = f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json"
path_sol = f"../solutions/instance{EXAMPLE}_LPmodel_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

config = LPConfiguration(
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
    step_min=4,
    MIPGap=0.01,
    time_limit_seconds=TIME_LIMIT_MINUTES * 60
)

instance = Instance.from_json(path_instance)
lp = LPModel(config=config, instance=instance)
lp.LPModel_print()
lp.solve()

# Check solution
sol_inconsistencies = lp.solution.check()
if sol_inconsistencies:
    raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
print("Optimal solution:", lp.solution.data)

# Save solution
if SAVE_SOLUTION:
    lp.solution.to_json(path_sol)

# Plot simple solution graph for each dam
if PLOT_SOL:
    for dam_id in instance.get_ids_of_dams():
        fig, ax = plt.subplots()
        lp.solution.plot_solution_for_dam(dam_id, ax)
        plt.show()
