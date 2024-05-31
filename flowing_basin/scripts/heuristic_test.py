"""
heuristic_tess.py
This script tests the Heuristic period evaluation function
"""

from flowing_basin.core import Instance
from flowing_basin.solvers import Baseline
from flowing_basin.solvers.heuristic import HeuristicSingleDam
from math import log


if __name__ == "__main__":
    instance = Instance.from_name("Percentile50", num_dams=6)
    config = Baseline(solver="Heuristic", general_config='G2').config
    for dam_id in instance.get_ids_of_dams():
        heuristic = HeuristicSingleDam(
            instance=instance, config=config, dam_id=dam_id, flow_contribution=instance.get_all_incoming_flows(),
            bias_weight=log(config.prob_below_half) / log(0.5)
        )
        print(dam_id, heuristic.get_all_period_values())


