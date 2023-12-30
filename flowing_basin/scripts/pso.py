from flowing_basin.core import Instance
from flowing_basin.solvers import PSOConfiguration, PSO
from datetime import datetime

SAVE_SOLUTION = True
EXAMPLE = 1
NUM_DAMS = 2
NUM_DAYS = 1
K_PARAMETER = 2
USE_RELVARS = True
TIME_LIMIT_MINUTES = 1

path_instance = f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json"
path_sol = f"../solutions/instance{EXAMPLE}_PSO_{NUM_DAMS}dams_{NUM_DAYS}days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

instance = Instance.from_json(path_instance)
config = PSOConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0,
    startups_penalty=50,
    limit_zones_penalty=0,
    volume_objectives={
        dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
    },
    use_relvars=USE_RELVARS,
    max_relvar=1,
    flow_smoothing=K_PARAMETER,
    mode="linear",
    num_particles=200,
    max_time=TIME_LIMIT_MINUTES * 60,
    cognitive_coefficient=2.905405139888455,
    social_coefficient=0.4232260541405988,
    inertia_weight=0.4424113459034113
)

pso = PSO(instance=instance, config=config)

# Find solution
status = pso.solve()
print("Status:", status)

# Check solution
sol_inconsistencies = pso.solution.check()
if sol_inconsistencies:
    raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
print("Optimal solution:", pso.solution.data)

# Save solution
if SAVE_SOLUTION:
    pso.solution.to_json(path_sol)