from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("A1G0O2R1T02", verbose=2)
rl.collect_obs()
rl.plot_histograms_projector()
rl.train()
rl.plot_histograms_observations()
rl.plot_training_curve()
rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])

