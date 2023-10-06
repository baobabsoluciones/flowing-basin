from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration
from datetime import datetime
import matplotlib.pyplot as plt

EXAMPLE = 1
NUM_DAMS = 2
NUM_DAYS = 1
PLOT_SOL = False
SAVE_SOLUTION = True
TIME_LIMIT_MINUTES = 1

path_instance = f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json"
path_sol = f"../solutions/instance{EXAMPLE}_LPmodel_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

instance = Instance.from_json(path_instance)
config = LPConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0,
    startups_penalty=50,
    limit_zones_penalty=1000,
    volume_objectives={
        dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
    },
    MIPGap=0.01,
    time_limit_seconds=TIME_LIMIT_MINUTES * 60,
    flow_smoothing=2,
)

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
