from flowing_basin.core import Instance
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
from time import perf_counter
from itertools import product
from math import sqrt

# EXAMPLES = [f'_intermediate{i}' for i in range(11)]
# EXAMPLES = ['1', '3']
# NUMS_DAMS = [i for i in range(1, 9)]
# NUM_REPLICATIONS = 200
EXAMPLES = ['1', '3']
NUMS_DAMS = [2]
NUM_REPLICATIONS = 2
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

obj_function_values = {(example, num_dams): [] for example, num_dams in product(EXAMPLES, NUMS_DAMS)}
exec_times = {(example, num_dams): 0. for example, num_dams in product(EXAMPLES, NUMS_DAMS)}

for example, num_dams in product(EXAMPLES, NUMS_DAMS):

    start = perf_counter()

    for replication in range(NUM_REPLICATIONS):

        instance = Instance.from_json(f"../instances/instances_big/instance{example}_{num_dams}dams_{NUM_DAYS}days.json")

        heuristic = Heuristic(config=config, instance=instance, do_tests=False)
        heuristic.solve()
        obj_function_values[example, num_dams].append(heuristic.solution.get_objective_function())
        # print(heuristic.solution.data)

    exec_time = perf_counter() - start
    exec_times[example, num_dams] = exec_time
    print(f"For instance {example} with {num_dams} dams: generated {NUM_REPLICATIONS} solutions in {exec_time}s.")

# Solved 200 instances in 20.582084900001064s. <-- without prints and WITH TESTS
# Solved 200 instances in 20.029083499975968s.
# Solved 200 instances in 21.50905569997849s. <-- without prints and WITHOUT TESTS
# Solved 200 instances in 20.016695700003766s.
# I got inconsistent results when running code with and without tests; I am not sure which is faster
# I think running the code w/o tests is faster, but running it w/ tests may be equivalent because of assert disabling

print(obj_function_values)
print(exec_times)

for example, num_dams in product(EXAMPLES, NUMS_DAMS):

    print(f"Objective function values for instance {example} with {num_dams} dams:")
    values = obj_function_values[example, num_dams]

    print("Minimum:", min(values))
    print("Maximum:", max(values))

    num_values = len(values)
    mean = sum(values) / num_values
    print("Mean:", mean)
    print("Standard deviation:", sqrt(sum([(val - mean) ** 2 for val in values]) / num_values))
