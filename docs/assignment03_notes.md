# Assignment 03 - short notes

## Small CNN vs pretrained CNN

A small CNN trained from scratch (a few convolution + pooling layers) must learn every filter from
about 4,500 training images. That is enough for coarse cues (brightness, colour, large lesions) but
not for a robust hierarchy of retinal features, so it overfits quickly and is sensitive to
initialisation and augmentation. It is fast and has few parameters.

A pretrained CNN (DenseNet121, ImageNet weights) already contains general edge, texture and shape
detectors learned from 1.2 million images. With the backbone frozen only the 1,025 parameters of the
head are trained, so training is fast, stable and needs little GPU memory, and the risk of
overfitting is small. Its limitation is domain shift: ImageNet features are not specific to fundus
pathology (micro-aneurysms, drusen, optic-disc changes), which caps accuracy unless deeper layers
are fine-tuned. For a small medical dataset the pretrained model is the stronger starting point.

## AI recommendation verification (brief)

| # | AI recommendation | How it was checked | Evidence | Verdict |
|---|---|---|---|---|
| 1 | Use image-level labels (`labels`/`target`), not patient flags N..O | Audit compared both sources per image | 772 normal eyes have patient flag N = 0 (would be mislabeled) | Supported |
| 2 | Match images by filename + patient ID | Audit matched all rows | 6392/6392 matched, 1608 unlabeled files ignored | Supported |
| 3 | Patient-grouped 70/15/15 split, seed 42 | Independent overlap check | 0 shared patients; abnormal share 0.550 / 0.553 / 0.551 | Supported |
| 4 | Keep repeated retinas (other patient ID) in one subset | Visual check of pairs, split verification | 33 pairs merged, 0 split across subsets | Supported (threshold is a judgement) |
| 5 | Reuse legacy embeddings only after verification | Re-encoding 20 images | cosine 1.0000 with stored rows | Supported |
| 6 | LogReg baseline, scaler fit on train, C chosen on validation | Code review, `results.json` | C = 0.1 chosen on validation (val F1 0.722); test F1 0.710 | Supported |
| 7 | Frozen DenseNet121 + early stopping on validation loss | Loss curves, `results.json` | best epochs 14 (no aug) and 5 (aug); test F1 0.717 vs 0.710 for LogReg | Partly supported (gain is small) |
| 8 | Moderate augmentation improves the frozen CNN | Same split, seed, budget and selection rule | val F1 0.725 vs 0.732, test F1 0.713 vs 0.717 (with vs without augmentation) | Not supported |
