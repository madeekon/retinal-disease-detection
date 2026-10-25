# Verification Checklist for AI-Assisted Work

Purpose: document how the AI-generated code and experimental plan were checked.
Every box is unchecked on purpose. Tick a box only after doing the check yourself and
write what you observed in the "Notes" line. Do not copy numbers from the AI chat;
take them from the files produced by the code.

## A. Data and labels (stage 1)

- [ ] A1. Opened `full_df.csv` and confirmed that one row corresponds to one image (`filename`).
      Notes:
- [ ] A2. Checked several rows by hand: `labels`, `target` and the eye-specific keyword agree.
      Notes:
- [ ] A3. Found at least one patient where the patient-level flags differ from the image-level label.
      Notes:
- [ ] A4. Confirmed that `clean_manifest.csv` and `data/splits/*.csv` contain no diagnostic keyword columns.
      Notes:
- [ ] A5. Opened a few images from `image_path` and confirmed they show the expected eye of the expected patient.
      Notes:
- [ ] A6. Reviewed every row of `excluded_records.csv` and agreed or disagreed with the exclusion.
      Notes:
- [ ] A7. Reviewed `unlabeled_image_files.csv` and confirmed these files are not used.
      Notes:
- [ ] A8. Reviewed `exact_duplicate_groups.csv` and `near_duplicate_pairs.csv`.
      Notes:
- [ ] A9. Looked at `near_duplicates_merged.png`: every merged pair shows the same retina.
      Notes:
- [ ] A10. Looked at `near_duplicates_review_band.png`: pairs below the merge threshold are different eyes
      (if not, lower `merge_threshold` and rebuild the split before any training).
      Notes:

## B. Split (stage 1)

- [ ] B1. Recomputed independently that no `patient_id` appears in more than one of train/val/test.
      Notes:
- [ ] B2. Checked that the image shares are close to 70/15/15 and the abnormal fraction is similar in all subsets.
      Notes:
- [ ] B3. Re-ran `02_make_splits.py` and confirmed that it refuses to overwrite the existing split.
      Notes:
- [ ] B4. Confirmed that later scripts load the split through `load_split()` (checksum verified).
      Notes:

## C. Code review (stages 2-4)

- [ ] C1. Scalers / normalisation statistics are fitted on the training subset only.
      Notes:
- [ ] C2. The test subset is not used for model selection, early stopping or threshold tuning.
      Notes:
- [ ] C3. The CNN uses 224x224 RGB input and ImageNet normalisation; augmentations are conservative.
      Notes:
- [ ] C4. Loss is `BCEWithLogitsLoss` with one raw logit; optimizer is AdamW.
      Notes:
- [ ] C5. The two CNN runs (with / without augmentation) share split, seed, budget and selection rule.
      Notes:
- [ ] C6. Metric implementations checked against a small hand-computed confusion matrix.
      Notes:

## D. Results versus recommendations (stage 4)

- [ ] D1. For each row of `ai_recommendation_table.md`, filled "Evidence" from saved files.
      Notes:
- [ ] D2. Stated where the results do NOT support an AI recommendation.
      Notes:
- [ ] D3. Reported false negatives and their consequences.
      Notes:
- [ ] D4. All numbers in the report can be traced to a CSV/JSON file in `outputs/`.
      Notes:
