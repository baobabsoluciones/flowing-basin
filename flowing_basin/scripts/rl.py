from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("G0O211A1R1T01")
rl.collect_obs()
rl.plot_histograms_projector()
rl.train()
