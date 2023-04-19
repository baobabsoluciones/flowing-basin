from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration

config = LPConfiguration(
    volume_objectives={
        "dam1": 55000,
        "dam2": 30000
        },
    step_min=4
)
instance = Instance.from_json("../data/input_example1.json")
lp = LPModel(config=config, instance=instance)
lp.LPModel_print()

