from flowing_basin.core import Instance
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
import matplotlib.pyplot as plt

EXAMPLE = 1
NUM_DAMS = 2
NUM_DAYS = 1
K_PARAMETER = 0

# Instance we want to solve
instance = Instance.from_json(f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json")

# Configuration
config = HeuristicConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
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
    mode="linear"
)

# Solver
heuristic = Heuristic(config=config, instance=instance)

# Sorted prices
# prices = instance.get_all_prices()
# prices_sorted = [prices[time_step] for time_step in heuristic.sort_time_steps()]
# plt.plot(prices_sorted, color='r')
# plt.show()

# Available volume in dam1
# available_volumes = heuristic.calculate_available_volumes("dam1")
# plt.plot(available_volumes, color='b')
# plt.show()

# Plot solution for dam1
assigned_flows, predicted_volumes = heuristic.solve_for_dam("dam1")
fig, ax = plt.subplots(1, 1)
twinax = ax.twinx()
ax.plot(predicted_volumes, color='b', label="Predicted volume")
ax.set_xlabel("Time (15min)")
ax.set_ylabel("Volume (m3)")
ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
twinax.plot(instance.get_all_prices(), color='r', label="Price")
twinax.plot(assigned_flows, color='g', label="Flow")
twinax.set_ylabel("Flow (m3/s), Price (€)")
twinax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.9))
plt.show()

# Check solution
turbined_flows, actual_flows, actual_volumes = heuristic.turbined_flows_from_assigned_flows("dam1", assigned_flows)
# Compare volumes:
plt.plot(predicted_volumes, color='b', label="Predicted volume")
plt.plot(actual_volumes, color='pink', label="Actual volume")
plt.show()
# Compare flows:
plt.plot(assigned_flows, color='g', label="Predicted volume")
plt.plot(actual_flows, color='pink', label="Actual volume")
plt.show()
