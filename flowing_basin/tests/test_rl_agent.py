from flowing_basin.core import Instance
from flowing_basin.solvers.rl import Agent
import torch

# Create instance and get limits of actions from it
instance = Instance.from_json("../data/input_example1.json")
action_lower_limits = torch.tensor([0 for _ in instance.get_ids_of_dams()])
action_upper_limits = torch.tensor([instance.get_max_flow_of_channel(dam_id) for dam_id in instance.get_ids_of_dams()])
print(action_lower_limits, action_upper_limits)

# Create agent with the given limits
agent = Agent(action_lower_limits=action_lower_limits, action_upper_limits=action_upper_limits)
print(agent.action_size)

# Random actions
print(agent.policy())
print(agent.policy())
print(agent.policy())
print(agent.policy())
