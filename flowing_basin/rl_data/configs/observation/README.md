- O1: all information from the past, with the price repeated for each dam
(required for the CNN, but can also be used with the MLP).
- O2: only one period from the past, with the price not repeated.
- O11: observation O1 with a PCA projector (trained with SAC agent observations).
- O21: observation O2 with a PCA projector (trained with SAC agent observations).
- O111: observation O1 with a QuantilePseudoDiscretizer followed by a PCA projector (trained with SAC agent observations).
- O211: observation O2 with a QuantilePseudoDiscretizer followed by a PCA projector (trained with SAC agent observations).
- O121: observation O1 with a QuantilePseudoDiscretizer followed by a PCA projector (trained with random agent observations).
- O221: observation O2 with a QuantilePseudoDiscretizer followed by a PCA projector (trained with random agent observations).