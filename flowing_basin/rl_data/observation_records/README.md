- "O11": Observations obtained from the training of a PCA agent with observation O1 in 1000 episodes
- "O12": Observations obtained from a random agent with observation O1 in 1000 episodes
- "O21": Observations obtained from the training of a PCA agent with observation O2 in 1000 episodes
- "O22": Observations obtained from a random agent with observation O2 in 1000 episodes

The folder name may additionally contain a string "_stepsN" indicating the number of timesteps
run when collecting the observations (if it is different from 99,000), and a string "_blockN" indicating
the block size (if it is different from 1).

Note the observations themselves, saved in the `observations.npy` file of every folder,
are gitignored because they occupy too much space.
