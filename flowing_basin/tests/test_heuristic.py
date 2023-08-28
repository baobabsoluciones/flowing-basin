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
    flow_smoothing=K_PARAMETER
)

# Solver
heuristic = Heuristic(config=config, instance=instance)

# Sorted prices
prices = instance.get_all_prices()
prices_sorted = [prices[time_step] for time_step in heuristic.time_steps_sorted]
plt.plot(prices_sorted, color='r')
plt.show()

# Available volume in dam1
available_volumes = heuristic.calculate_available_volume_for_dam("dam1")
plt.plot(available_volumes, color='b')
plt.show()
