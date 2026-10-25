# Assignment 04 (Variant 9) - short notes

Source of all numbers: `outputs/assignment04/results.json` and `outputs/assignment04/metrics/som_comparison_table.csv`.
Data: 4468 train / 957 test images, 1024-d DenseNet121 embeddings -> StandardScaler + PCA(32) fitted on train
(73.3 % variance kept). Same budget for every map: 20 x 4468 = 89,360 sample presentations; 5 seeds (42-46).

## Numerical example (one BMU selection and one update)

Input x = (0.8, 0.2). Three neurons on a line lattice: w1 = (0.1, 0.9), w2 = (0.6, 0.4), w3 = (0.9, 0.9).

1. Distances: d1 = sqrt(0.7^2 + 0.7^2) = 0.990; d2 = sqrt(0.2^2 + 0.2^2) = 0.283; d3 = sqrt(0.1^2 + 0.7^2) = 0.707.
   BMU = neuron 2 (smallest distance).
2. Learning rate 0.5, sigma 1. Neighbourhood h = exp(-d_grid^2 / 2): h2 = 1, h1 = h3 = exp(-0.5) = 0.607.
3. Update w_j <- w_j + 0.5 * h_j * (x - w_j):
   w2 = (0.6 + 0.5*(0.2), 0.4 + 0.5*(-0.2)) = (0.700, 0.300);
   w1 = (0.1 + 0.303*0.7, 0.9 - 0.303*0.7) = (0.312, 0.688);
   w3 = (0.9 - 0.303*0.1, 0.9 - 0.303*0.7) = (0.870, 0.688).
   (Same values are computed by the script in `results.json` -> `numerical_example`.)

## Which lattice size is the best balance?

* QE falls steadily with more units (21.48 -> 19.89 -> 18.64 -> 17.99 on train) because each unit covers fewer samples.
* TE is lowest for 3x3 (0.13), but this is trivial: in a 3x3 grid most units are neighbours. From 5x5 upward TE is
  stable on train (0.22-0.23) and slightly higher on test for 8x8/10x10 (0.26), a sign of mild over-fitting of the lattice.
* No empty units on train for any size; 10x10 starts to leave units empty on test (0.2 %).
* Training time grows slowly (about 1.3 s -> 2.8 s in the reference run; times depend on the computer).
* Normal/Abnormal separation (labels used only afterwards) improves only slightly: 0.56 -> 0.58 -> 0.58 -> 0.59.
* 8x8 is the best compromise: clearly lower QE than 5x5, the same test TE as 10x10, no empty units, 64 units are still
  readable on the U-matrix and hit map, and its regions with high abnormal share are visible.

## AI recommendation verification (brief)

| # | AI recommendation | Check | Evidence | Verdict |
|---|---|---|---|---|
| 1 | Standardize and fit PCA(32) on train only | Code review | scaler/PCA `.fit` called on train only; 73.3 % variance kept | Supported |
| 2 | Same presentation budget for all maps | `results.json` settings | 89,360 presentations for each size | Supported |
| 3 | Larger lattices lower QE | 5 seeds per size | QE 21.48 -> 17.99, std <= 0.03 | Supported |
| 4 | TE must be read together with lattice size | Compare 3x3 with larger maps | 3x3 TE 0.13 vs 0.22-0.26 for larger maps | Supported |
| 5 | Larger maps leave units empty | Hit maps, empty % | 0 % train for all sizes; 0.2 % test only for 10x10 | Partly supported (effect small at <= 100 units) |
| 6 | SOM clusters separate Normal from Abnormal | Label agreement on test | 0.56-0.59 vs 0.55 majority baseline | Weakly supported |
| 7 | 8x8 is the best balance | Combined comparison above | see section above | Supported (judgement) |
| 8 | The own SOM implementation is correct | Same data and budget with the independent MiniSom library (seed 42, test set) | MiniSom QE 21.12 / 21.02 / 20.95 and TE 0.16 / 0.13 / 0.10 for 5x5 / 8x8 / 10x10: higher QE, lower TE than ours, because MiniSom's neighbourhood only shrinks to sigma/3 instead of 0.5 (smoother map) | Supported (difference explained by the schedule) |
| 9 | 20 epochs is enough training | Our SOM with double budget (40 epochs, seed 42, test set) | QE 8x8 18.85 -> 18.65, 10x10 18.32 -> 18.23 (small change); TE 10x10 rises 0.26 -> 0.30 | Supported |
