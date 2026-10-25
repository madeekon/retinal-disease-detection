# AI Recommendation Table

Recommendations made by the AI assistant (Claude) while designing this project.
The columns "Evidence" and "Verdict" must be filled in by the student after checking
the code and the experimental results. Nothing in those columns is pre-filled.

Verdict options: Supported / Partly supported / Not supported / Not yet tested.

| ID | Stage | AI recommendation | Reason given by the AI | How to check it | Evidence (student) | Verdict (student) |
|---|---|---|---|---|---|---|
| R1 | Data | Use the image-level label from `labels`/`target`, not the patient-level N..O flags | Patient flags combine both eyes; a healthy eye of a patient with one diseased eye has patient flag N = 0 | Compare both label sources per image in `labeled_images_audit.csv`; count normal eyes with patient N = 0 in `audit_report.json` | | |
| R2 | Data | Binary target: N -> 0, all other codes -> 1 | Matches the assignment goal "normal vs any disease or abnormality" | Check `binary_target` against `label_code` in `clean_manifest.csv` | | |
| R3 | Data | Match images by filename and check that the patient ID in the filename equals `ID` | Array position is not a safe key; the folder holds more files than labeled rows | `image_checks` and `image_dir_scan` in `audit_report.json`; `unlabeled_image_files.csv` | | |
| R4 | Data | Exclude records whose eye keywords contradict the binary label | A label that contradicts its own clinical description is unreliable | Inspect the rows in `excluded_records.csv` manually | | |
| R5 | Data | Split 70/15/15 by patient, stratified by group composition, seed 42; merge patients that share an identical image | Prevents both eyes of one patient (and copied images) from reaching different subsets | `verification` block in `split_summary.json`; recompute overlaps independently | | |
| R5b | Data | Detect repeated retinas under different patient IDs with DenseNet121 descriptors: merge patients into one split group at cosine >= 0.97 (same eye side); treat >= 0.98 as the same retina and exclude such pairs with conflicting normal/abnormal labels | MD5 misses re-encoded copies and repeat visits; such pairs across subsets leak test information | Inspect `near_duplicates_merged.png` and `near_duplicates_review_band.png`; check `near_duplicate_pairs.csv`; confirm the threshold separates same-retina pairs from different eyes | | |
| R6 | Data | Save the split once and verify checksums before every experiment | All models must be compared on exactly the same images | `load_split()` raises on modified files; compare sha256 values | | |
| R7 | Data | Do not reuse the legacy embeddings blindly; re-extract embeddings keyed by filename | Legacy files store no filenames, use patient-level labels and skip the final ReLU | `legacy_embeddings` block in `audit_report.json` (row-order checks and re-encoding check) | | |
| R8 | Models | Logistic Regression on DenseNet121 embeddings as baseline; scaler fitted on train only | Simple, strong baseline; avoids leakage of validation/test statistics | Code review of the training script (stage 2) | | |
| R9 | Models | PyTorch MLP on the same embeddings with validation-based early stopping (optional dropout) | Tests whether a nonlinear head improves on the linear baseline under identical inputs | Validation curves and test metrics (stage 2) | | |
| R10 | Models | DenseNet121 with frozen backbone and trainable binary head; BCEWithLogitsLoss with one logit; AdamW | Standard transfer-learning setup for a small medical dataset on a 4 GB GPU | Code review; training logs (stage 3) | | |
| R11 | Models | Compare no augmentation vs conservative augmentation under identical conditions | Isolates the effect of augmentation | Same split, seed, epochs budget, selection rule (stage 3) | | |
| R12 | Evaluation | Select hyperparameters and stopping points on validation only; test once at the end | Test set must remain an unbiased estimate | Code review: test set loaded only in the final evaluation | | |
| R13 | Evaluation | Report false negatives separately | Missing a diseased eye is the clinically most costly error | Confusion matrices and FN table (stage 4) | | |
