"""
This script runs the MILP solver for a single instance and saves the solution in the current folder.
It is useful for debugging the MILP model.
"""

from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration
from datetime import datetime

TIME_LIMIT_SECONDS = 15
INSTANCE_NAME = "Percentile25"
NUM_DAMS = 2
PATH_SOLUTION = f"lp_run_{INSTANCE_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

if __name__ == "__main__":

    config = LPConfiguration(
        volume_shortage_penalty=0.,
        volume_exceedance_bonus=0.,
        startups_penalty=50,
        limit_zones_penalty=50,
        volume_objectives={},
        MIPGap=0.01,
        max_time=TIME_LIMIT_SECONDS,
        flow_smoothing=0,
        max_relvar=0.2
    )
    instance = Instance.from_name(INSTANCE_NAME, num_dams=NUM_DAMS)
    lp = LPModel(config=config, instance=instance)
    lp.LPModel_print()

    lp.solve()
    lp.solution.to_json(PATH_SOLUTION)
