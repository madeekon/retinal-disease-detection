# Grad-CAM (XAI) - summary of findings

Source: `outputs/final_project/xai/gradcam_examples.png`, `gradcam_cases.csv`, `results.json`.
Model: Assignment 03 DenseNet121 (ImageNet weights, frozen backbone, trained head, no augmentation); no retraining.
Grad-CAM on the last dense block (7 x 7 feature maps), explaining the score of the predicted class.
12 test images: the 3 most confident true positives, true negatives, false positives and false negatives.

## Sanity check
The re-computed probabilities match the probabilities saved in Assignment 03 (maximum difference 0.0005), so the
correct checkpoint and the correct images were used.

## Observations
* Resolution: the maps come from a 7 x 7 grid, so each heat blob covers about 1/7 of the image width. Small lesions
  such as microaneurysms or the optic-disc cup cannot be localised at this scale.
* True positives (cataract 2282_right, AMD 1857_left, pathological myopia 1574_right; P = 0.99-1.00): the heat is
  concentrated in the central retina (24 % of the heat in the central ninth of the image on average, about twice the
  share expected from uniform heat). For 1857_left and 1574_right it lies on the large, visibly abnormal central
  areas (macular pigment changes, atrophic patches) and not on the bright optic disc. For the cataract image the heat
  is broad and diffuse, consistent with a global loss of sharpness rather than a focal finding.
* True negatives (P = 0.16): the evidence for "Normal" lies mostly in the upper retina along the vessel arcades;
  the optic disc and the macula are usually cold.
* False positives (images labelled Normal, P = 0.88-0.93): 2197_left is dark, unevenly lit and shows several round
  bright spots (likely dust or reflections); the heat sits on one of these spots. 3218_right contains many small
  yellow dots and a bright area next to the disc, and the heat covers the dotted region; such an image is hard to
  call normal from appearance alone, so label noise cannot be excluded. 552_right looks like a well-exposed normal
  fundus; the heat covers the bright central region, suggesting that colour and brightness also drive the decision.
* False negatives (glaucoma 1246_right, other 140_right, diabetic retinopathy 4251_left; P = 0.15-0.18): 1246_right
  is very dark and the heat lies at the lower border, not on the optic disc where glaucomatous changes would appear.
  4251_left is a low-contrast image with heat above and below the fovea; diabetic lesions are too small to be seen
  at this resolution. In 140_right the heat follows the optic disc and the lower vessels.
* Heat outside the retinal disc (black border): 5-16 % per image, mean 11 %, below the roughly 20 % expected for
  evenly spread heat; most evidence therefore comes from the retina itself, but part of it comes from the image
  border and corners.

## Interpretation and limits
The frozen ImageNet backbone responds to large, high-contrast patterns and to global colour and illumination. It
recognises obvious, large abnormalities, but it misses small or subtle disease signs, especially in dark or
low-contrast photographs, and it can be misled by artefacts such as lens dust. Grad-CAM shows which image regions
increased the model's score; it does not confirm any medical pathology, and these 12 cases (the most confident of
each group) are illustrative, not a quantitative evaluation. Fine-tuning deeper layers or using a higher input
resolution would be needed before the model's attention could be expected to match clinically relevant structures.
