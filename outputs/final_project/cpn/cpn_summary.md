# Counterpropagation Network (CPN) - summary of findings

Source: `outputs/final_project/cpn/results.json`, `summary_test.csv`, `config_search_val.csv`.

* Architecture: 1024-d DenseNet121 embeddings -> StandardScaler + PCA(32) (fitted on the training subset,
  73.3 % variance) -> Kohonen layer (winner-take-all, instar rule) -> Grossberg layer (one outstar weight per
  Kohonen unit = estimated probability of Abnormal for that cluster). Decision threshold 0.5.
* Configuration search (seed 42, validation F1): 16 units 0.604, 64 units 0.616, 144 units 0.620.
  The largest layer (144 units) was selected; no unit was dead in any configuration.
* Test set, 144 units, 3 seeds: Accuracy 0.568 ± 0.007, Precision 0.607 ± 0.014, Recall 0.612 ± 0.044,
  F1 0.609 ± 0.015, ROC-AUC 0.620 ± 0.002.
* Interpretation: the CPN is clearly weaker than the MLP on the same embeddings (test F1 0.720-0.727,
  ROC-AUC about 0.77). Its output is a look-up table: every image receives the abnormal rate of the cluster it
  falls into. The Kohonen layer forms clusters without labels, and, as in the Assignment 04 SOM analysis
  (label agreement 0.56-0.59), these clusters follow general image appearance more than disease status.
  More units give slightly finer clusters and better validation F1, but small clusters also produce extreme
  outputs (0 or 1) estimated from very few samples. The advantage of the CPN is speed (about 2 s training, no
  gradients) and transparency (each prediction can be traced to one prototype and its label statistics).
