- `G0`: linear simulator, no bonus or penalty for objective final volumes,
flow smoothing of 2, startup and limit zone penalty of 50€.
- `G01`: same as G0 but with a maximum flow variation of 20%
instead of the flow smoothing constraint.
- `G1`: simplified environment - same as G0, but with no
flow smoothing, startup or limit zone penalty.
- `G2`: same as G0 but with 6 dams instead of just 2.
- `G21`: same as G2 but with a maximum flow variation of 20%
instead of the flow smoothing constraint.
- `G3`: same as G1 but with 6 dams instead of just 2.
- `G8`: same as G0 but with 1 dam instead of 2.
- `G9`: same as G1 but with 1 dam instead of 2.