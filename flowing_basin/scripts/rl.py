from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("G0O221A1R1T0", verbose=2)
rl.collect_obs()
rl.plot_histograms_projector()
rl.train()
