- `O1`: all information from the past, with the price repeated for each dam
(required for the CNN, but can also be used with the MLP).
- `O1X`: observation O1 with a PCA projector.
- `O1X1`: observation O1 with a QuantilePseudoDiscretizer followed by a PCA projector.
- `O2`: only one period from the past, with the price not repeated.
- `O2X`: observation O2 with a PCA projector.
- `O2X1`: observation O2 with a QuantilePseudoDiscretizer followed by a PCA projector.

In all of these, `X` can be either `1` or `2`:
- If `X` is `1`, the PCA projector is trained with observations obtained from a SAC agent.
- If `X` is `2`, the PCA projector is trained with observations obtained from a random agent.