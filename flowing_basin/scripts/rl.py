from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("G0O221A1R1T01")
rl.collect_obs()
rl.train()
