from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin

instance = Instance.from_json("../instances/instances_base/instance1.json")
river_basin = RiverBasin(instance=instance, mode="linear")
solution = Solution.from_json("../solutions/instance1_LPmodel_2dams_1days_time2023-10-04_10-12.json")

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
    print(f"{dam_id}'s number of startups: {river_basin.history.loc[0:95, f'{dam_id}_startups'].sum()}")
    print(f"{dam_id}'s number of limit zones: {river_basin.history.loc[0:95, f'{dam_id}_limits'].sum()}")
