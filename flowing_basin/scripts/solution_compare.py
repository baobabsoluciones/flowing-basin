from flowing_basin.core import Solution
from matplotlib import pyplot as plt

INSTANCE = 1
NUM_DAMS = 2
SOLUTIONS = [
    ("PSO", "2023-09-25_00-51"),
    ("PSO", "2023-09-25_13-20"),
    ("LPmodel", "2023-09-26_07-39"),
    ("PSO-RBO", "2023-09-28_18-31")
]

# Load solutions
solutions = {
    (solver, sol_datetime):
        Solution.from_json(f"../solutions/instance{INSTANCE}_{solver}_{NUM_DAMS}dams_1days_time{sol_datetime}.json")
    for solver, sol_datetime in SOLUTIONS
}

# Plot obj fun history for every solution
fig, ax = plt.subplots()
for sol_info, solution in solutions.items():
    solution.plot_objective_values(ax, label=f"{sol_info}")
ax.set_ylim(bottom=0)  # Call AFTER plotting to automatically have the top adjusted
ax.legend()
plt.show()
