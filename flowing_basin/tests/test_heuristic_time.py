from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
from time import perf_counter

EXAMPLES = [f'_intermediate{i}' for i in range(10)]
NUM_REPLICATIONS = 20
NUM_DAMS = 2
NUM_DAYS = 1
K_PARAMETER = 2
MAXIMIZE_FINAL_VOL = False

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
    maximize_final_vol=MAXIMIZE_FINAL_VOL
)

num_instances = 0
start = perf_counter()

for example in EXAMPLES:
    for replication in range(NUM_REPLICATIONS):

        instance = Instance.from_json(f"../instances/instances_big/instance{example}_{NUM_DAMS}dams_{NUM_DAYS}days.json")
        num_instances += 1

        heuristic = Heuristic(config=config, instance=instance, do_tests=True)
        heuristic.solve()
        # print(heuristic.solution.data)

exec_time = perf_counter() - start
print(f"Solved {num_instances} instances in {exec_time}s.")
# Solved 200 instances in 32.902009000012185s. <-- without prints and WITH TESTS
# Solved 200 instances in 33.10179009998683s.
# Solved 200 instances in 30.569451900024433s.
# Solved 200 instances in 27.123247299983632s.
# Solved 200 instances in 24.85201999999117s. <-- without prints and WITHOUT TESTS
# Solved 200 instances in 24.950323499972s.
# Solved 200 instances in 36.96347959997365s.
# Solved 200 instances in 29.957258799986448s.
# I got inconsistent results when running code with and without tests; I am not sure which is faster
# I think running the code w/o tests is faster, but running it w/ tests may be equivalent because of assert disabling
