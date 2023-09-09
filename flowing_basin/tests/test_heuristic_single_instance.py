from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
from datetime import datetime

EXAMPLE = '3'
NUM_DAMS = 1
NUM_DAYS = 1
K_PARAMETER = 2
PLOT_SOL = True
RANDOM_BIASED_FLOWS = True
PROB_BELOW_HALF = 0.15
MAXIMIZE_FINAL_VOL = False

# Instance we want to solve
instance = Instance.from_json(f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json")

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
)

# Solution
heuristic = Heuristic(config=config, instance=instance)
heuristic.solve()
print("Total income:", heuristic.solution.get_objective_function())
path_sol = f"../solutions/instance{EXAMPLE}_Heuristic_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
# heuristic.solution.to_json(path_sol)

# Plot simple solution graph for each dam
if PLOT_SOL:
    import matplotlib.pyplot as plt
    for dam_id in instance.get_ids_of_dams():
        assigned_flows = heuristic.solution.get_exiting_flows_of_dam(dam_id)
        predicted_volumes = heuristic.solution.get_volumes_of_dam(dam_id)
        fig, ax = plt.subplots(1, 1)
        twinax = ax.twinx()
        ax.plot(predicted_volumes, color='b', label="Predicted volume")
        ax.set_xlabel("Time (15min)")
        ax.set_ylabel("Volume (m3)")
        ax.legend()
        twinax.plot(instance.get_all_prices(), color='r', label="Price")
        twinax.plot(assigned_flows, color='g', label="Flow")
        twinax.set_ylabel("Flow (m3/s), Price (€)")
        twinax.legend()
        plt.show()
