from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("G0O22A1R1T01", verbose=2)
rl.collect_obs()
# rl.plot_histograms_projector()
rl.train()
rl.plot_histograms_observations()
