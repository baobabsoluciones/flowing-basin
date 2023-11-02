from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import HeuristicConfiguration, Heuristic
from time import perf_counter
from itertools import product
from math import sqrt
from dataclasses import asdict
import csv
import json

EXAMPLES = ['Percentile25', 'Percentile75']
NUMS_DAMS = [2, 4, 6, 8, 10]
VOL_BONUS = True
POWER_PENALTY = True

NUM_REPLICATIONS = 10
RANDOM_BIASED_FLOWS = True
RANDOM_BIASED_SORTING = True

SAVE_REPORT = False
REPORT_NAME = (
    f"test_heuristic_old_instances25and75"
    f"{'_VolExceed' if VOL_BONUS else ''}"
    f"{'_NoPowerPenalty' if not POWER_PENALTY else ''}"
)
DECIMAL_PLACES = 2
COMPARE_MILP = True

report_filepath = f"reports/{REPORT_NAME}.csv"
config_filepath = f"reports/{REPORT_NAME}_config.json"

first_row = [
    "instance", "no_dams", "greedy_val_eur",
    "num_trials", "exec_time_s", "min_val_eur", "max_val_eur", "avg_val_eur", "std_eur"
]
if COMPARE_MILP:
    first_row += ["milp_val_eur", "milp_final_gap", "diff_milp_vs_best"]
report = [first_row]

for example, num_dams in product(EXAMPLES, NUMS_DAMS):

    start = perf_counter()
    obj_function_values = []

    # Instance and configuration
    instance = Instance.from_json(
        f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json"
    )
    config = HeuristicConfiguration(
        volume_shortage_penalty=3,
        volume_exceedance_bonus=0.035 if VOL_BONUS else 0.,
        startups_penalty=50. if POWER_PENALTY else 0.,
        limit_zones_penalty=50. if POWER_PENALTY else 0.,
        volume_objectives={
            dam_id: (
                instance.get_min_vol_of_dam(dam_id) + (
                    instance.get_max_vol_of_dam(dam_id) - instance.get_min_vol_of_dam(dam_id)
                ) / 2
            ) for dam_id in instance.get_ids_of_dams()
        },
        flow_smoothing=2,
        mode="linear",
        maximize_final_vol=False,
        random_biased_flows=False,
        prob_below_half=0.15,
        random_biased_sorting=False,
        common_ratio=0.6,
    )

    # Greedy solution (heuristic)
    heuristic = Heuristic(config=config, instance=instance, greedy=True, do_tests=True)
    heuristic.solve()
    obj_function_greedy = heuristic.solution.get_objective_function()

    # MILP solution
    obj_function_milp = None
    milp_final_gap = None
    if COMPARE_MILP:
        milp_sol = Solution.from_json(
            f"../solutions/test_milp/instance{example}_MILP_{num_dams}dams_1days"
            f"{'_VolExceed' if VOL_BONUS else ''}{'_NoPowerPenalty' if not POWER_PENALTY else ''}.json"
        )
        obj_function_milp = milp_sol.get_objective_function()
        milp_final_gap = milp_sol.get_final_gap_value()

    # RBO solutions
    config.random_biased_sorting = RANDOM_BIASED_SORTING
    config.random_biased_flows = RANDOM_BIASED_FLOWS
    for replication in range(NUM_REPLICATIONS):
        heuristic = Heuristic(config=config, instance=instance, do_tests=True)
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

    # Fraction over MILP
    milp_vs_best = None
    if COMPARE_MILP:
        rbo_best = max(obj_function_greedy, max_obj_val)
        milp_vs_best = (obj_function_milp - rbo_best) / obj_function_milp

    new_row = [
        example, num_dams, round(obj_function_greedy, DECIMAL_PLACES),
        NUM_REPLICATIONS, round(exec_time, DECIMAL_PLACES), round(min_obj_val, DECIMAL_PLACES),
        round(max_obj_val, DECIMAL_PLACES), round(avg_obj_val, DECIMAL_PLACES), round(std_obj_val, DECIMAL_PLACES)
    ]
    if COMPARE_MILP:
        new_row += [
            round(obj_function_milp, DECIMAL_PLACES), round(milp_final_gap, DECIMAL_PLACES),
            round(milp_vs_best, DECIMAL_PLACES)
        ]
    report.append(new_row)

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
