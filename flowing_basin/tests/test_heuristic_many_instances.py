from flowing_basin.core import Instance
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
from time import perf_counter
from itertools import product
from math import sqrt
from dataclasses import asdict
import csv
import json

# EXAMPLES = [f'_intermediate{i}' for i in range(11)]
EXAMPLES = ['1', '3']
# EXAMPLES = ['1']
# NUMS_DAMS = [i for i in range(1, 9)]
NUMS_DAMS = [6, 7, 8]
NUM_REPLICATIONS = 200
NUM_DAYS = 1
K_PARAMETER = 2
RANDOM_BIASED_FLOWS = True
PROB_BELOW_HALF = 0.15
MAXIMIZE_FINAL_VOL = False
SAVE_REPORT = False
REPORT_NAME = "random_biased_flows"
DECIMAL_PLACES = 2

report_filepath = f"reports/{REPORT_NAME}.csv"
config_filepath = f"reports/{REPORT_NAME}_config.json"

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

report = [
    ["instance", "no_dams", "num_trials", "exec_time_s", "min_val_eur", "max_val_eur", "avg_val_eur", "std_eur"]
]

for example, num_dams in product(EXAMPLES, NUMS_DAMS):

    start = perf_counter()
    obj_function_values = []

    for replication in range(NUM_REPLICATIONS):

        instance = Instance.from_json(
            f"../instances/instances_big/instance{example}_{num_dams}dams_{NUM_DAYS}days.json"
        )

        heuristic = Heuristic(config=config, instance=instance, do_tests=False)
        heuristic.solve()
        obj_function_values.append(heuristic.solution.get_objective_function())
        # print(heuristic.solution.data)

    print(f"For instance {example} with {num_dams} dams:")

    exec_time = perf_counter() - start
    print(f"\tgenerated {NUM_REPLICATIONS} solutions in {exec_time}s.")

    min_obj_val = min(obj_function_values)
    max_obj_val = max(obj_function_values)
    num_obj_val = len(obj_function_values)
    avg_obj_val = sum(obj_function_values) / num_obj_val
    std_obj_val = sqrt(sum([(obj_val - avg_obj_val) ** 2 for obj_val in obj_function_values]) / num_obj_val)
    print(f"\tthe obj fun values are {obj_function_values}.")
    print(f"\t - minimum: {min_obj_val}")
    print(f"\t - maximum: {max_obj_val}")
    print(f"\t - mean: {avg_obj_val}")
    print(f"\t - standard deviation: {std_obj_val}")

    report.append([
        example, num_dams, NUM_REPLICATIONS, round(exec_time, DECIMAL_PLACES), round(min_obj_val, DECIMAL_PLACES),
        round(max_obj_val, DECIMAL_PLACES), round(avg_obj_val, DECIMAL_PLACES), round(std_obj_val, DECIMAL_PLACES)
    ])

# Print results
print(
    "--------",
    "Results:",
    *[
        ''.join([f"{el:<15.2f}" if isinstance(el, float) else f"{el:<15}" for el in row]) for row in report
    ],
    "--------",
    sep='\n'
)

if SAVE_REPORT:

    # Create report file
    with open(report_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(report)
    print(f"Created CSV file '{report_filepath}'.")

    # Store configuration used
    with open(config_filepath, 'w') as file:
        json.dump(asdict(config), file, indent=2)
    print(f"Created JSON file '{config_filepath}'.")

# Solved 200 instances in 20.582084900001064s. <-- without prints and WITH TESTS
# Solved 200 instances in 20.029083499975968s.
# Solved 200 instances in 21.50905569997849s. <-- without prints and WITHOUT TESTS
# Solved 200 instances in 20.016695700003766s.
# I got inconsistent results when running code with and without tests; I am not sure which is faster
# I think running the code w/o tests is faster, but running it w/ tests may be equivalent because of assert disabling
