from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("A1G1O2R1T2", verbose=2)
rl.check_train_env(max_timestep=6, obs_types=['raw'])
# rl.collect_obs()
# rl.plot_histograms_projector()
# rl.train()
# rl.plot_histograms_observations()
# rl.plot_training_curve()
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])

# print(ReinforcementLearning.get_all_agents(".*G0.*T1"))
# ReinforcementLearning.plot_all_training_curves(".*G0.*T1")

