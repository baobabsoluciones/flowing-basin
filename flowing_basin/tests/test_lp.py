from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration
from datetime import datetime
import matplotlib.pyplot as plt

EXAMPLE = 3
NUM_DAMS = 8
NUM_DAYS = 1

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
    time_limit_seconds=15*60
)

instance = Instance.from_json(f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json")
lp = LPModel(config=config, instance=instance)
lp.LPModel_print()

lp.solve()
path_sol = f"../solutions/instance{EXAMPLE}_LPmodel_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
lp.solution.to_json(path_sol)

# Plot simple solution graph for each dam
for dam_id in instance.get_ids_of_dams():

    assigned_flows = lp.solution.get_exiting_flows_of_dam(dam_id)
    predicted_volumes = lp.solution.get_volumes_of_dam(dam_id)

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
