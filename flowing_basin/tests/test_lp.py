from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration
from datetime import datetime

config = LPConfiguration(
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
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
    startups_penalty=50,
    limit_zones_penalty=0,
    step_min=4,
    MIPGap=0.01,
    time_limit_seconds=8*60
)

EXAMPLE = 3
NUM_DAMS = 2
NUM_DAYS = 1

instance = Instance.from_json(f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json")
lp = LPModel(config=config, instance=instance)
lp.LPModel_print()

lp.solve()
path_sol = f"../solutions/instance{EXAMPLE}_LPmodel_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
lp.solution.to_json(path_sol)
