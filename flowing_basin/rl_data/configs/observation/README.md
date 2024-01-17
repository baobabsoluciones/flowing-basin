- `O1`: all information from the past, with the price repeated for each dam
(required for the CNN, but can also be used with the MLP).
- `O1X`: observation O1 with a PCA projector.
- `O1X1`: observation O1 with a QuantilePseudoDiscretizer followed by a PCA projector.
- `O1X2`: observation O1 with a QuantilePseudoDiscretizer.
- `O2`: only one period from the past, with the price not repeated.
- `O2X`: observation O2 with a PCA projector.
- `O2X1`: observation O2 with a QuantilePseudoDiscretizer followed by a PCA projector.
- `O2X2`: observation O2 with a QuantilePseudoDiscretizer.

In all of these, `X` can be either `1` or `2`:
- If `X` is `1`, the PCA projector is trained with observations obtained from a SAC agent.
- If `X` is `2`, the PCA projector is trained with observations obtained from a random agent.

Observations with randomized features (for testing purposes):
- `O10`: observation O1 with all features randomized.
- `O101`: observation O1 with all features randomized, except the future prices.
- `O20`: observation O2 with all features randomized.
- `O201`: observation O2 with all features randomized, except the future prices.
