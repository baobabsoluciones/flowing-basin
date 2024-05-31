from flowing_basin.core import Instance
from flowing_basin.solvers import Heuristic, Baseline


if __name__ == "__main__":
    instance = Instance.from_name("Percentile50", num_dams=6)
    config = Baseline(solver="Heuristic", general_config='G2').config
    heuristic = Heuristic(instance=instance, config=config)
    for period in range(instance.get_largest_impact_horizon()):
        print(period, heuristic.get_period_value(period))


