from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin

INSTANCE = '_intermediate0'
NUM_DAMS = 2
# TIME = '2023-10-04_10-12'  # '1' 2dams
# TIME = '2023-10-05_22-39'  # '3' 2dams
# TIME = '2023-10-06_01-14'  # '_intermediate0' 1dam
TIME = '2023-10-06_01-43'  # '_intermediate0' 2dams

instance = Instance.from_json(f"../instances/instances_big/instance{INSTANCE}_{NUM_DAMS}dams_1days.json")
river_basin = RiverBasin(instance=instance, mode="linear")
solution = Solution.from_json(f"../solutions/instance{INSTANCE}_LPmodel_{NUM_DAMS}dams_1days_time{TIME}.json")

# Check solution
assert solution.complies_with_flow_smoothing(
    flow_smoothing=2,
    initial_flows={
        dam_id: instance.get_initial_lags_of_channel(dam_id)[0]
        for dam_id in instance.get_ids_of_dams()
    }
)
print(solution.check())

# Print simulation historic values with solution
river_basin.deep_update_flows(solution.get_exiting_flows_array())
print(river_basin.history.to_string())
for dam_id in instance.get_ids_of_dams():
    print(f"{dam_id}'s income from energy: {river_basin.history[f'{dam_id}_income'].sum()}")
    print(f"{dam_id}'s number of startups: {river_basin.history[f'{dam_id}_startups'].sum()}")
    print(f"{dam_id}'s number of limit zones: {river_basin.history[f'{dam_id}_limits'].sum()}")
