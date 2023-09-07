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

# Plot solution for dam1
assigned_flows, predicted_volumes = heuristic.single_dam_solvers['dam1'].solve()
fig1, ax = plt.subplots(1, 1)
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

# Check solution
actual_volumes, _, actual_flows, dam_income, dam_net_income = heuristic.single_dam_solvers['dam1'].simulate()
fig2, axs = plt.subplots(1, 2)
# Compare volumes:
axs[0].set_xlabel("Time (15min)")
axs[0].plot(predicted_volumes, color='b', label="Predicted volume")
axs[0].plot(actual_volumes, color='c', label="Actual volume")
axs[0].set_ylabel("Volume (m3)")
axs[0].legend()
# Compare flows:
axs[1].set_xlabel("Time (15min)")
axs[1].plot(assigned_flows, color='g', label="Assigned flows")
axs[1].plot(actual_flows, color='lime', label="Actual exiting flows")
axs[1].set_ylabel("Flow (m3/s)")
axs[1].legend()
plt.show()

# Evaluate solution
print("TOTAL INCOME:", dam_income)
print("TOTAL INCOME (w/ startup costs and obj final volumes):", dam_net_income)

# print([time_step for time_step in range(instance.get_largest_impact_horizon()) if abs(predicted_volumes[time_step] - actual_volumes[time_step]) > 1000])
# for time_step in range(instance.get_largest_impact_horizon()):
#     print(
#         "heuristic", time_step, heuristic.single_dam_solvers['dam1'].added_volumes[time_step],
#         assigned_flows[time_step] * instance.get_time_step_seconds(), assigned_flows[time_step],
#         predicted_volumes[time_step]
#     )
