# IDRiD Diabetic Retinopathy Grading — Three-Day Plan

## 1. Goal and scope

Build a small, reproducible PyTorch system that accepts one retinal fundus photograph and predicts one of the five **ordered** diabetic retinopathy (DR) grades:

| Grade | Meaning |
|---:|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR (PDR) |

The first model will be an ImageNet-pretrained **EfficientNet-B0** created through `timm`. The project will classify disease severity only. Lesion segmentation and diabetic macular edema (DME) grading are outside the three-day scope.

The final demo is a research/hackathon prototype, not a diagnostic medical device. Its output should include the predicted grade and class probabilities, plus an explicit clinical-use disclaimer.

## 2. Inspection status and important limitation

Inspection on 2026-08-21 found that the assigned workspace `C:\Users\doodl\OneDrive\Desktop\Dr`:

- is empty except for this plan after planning;
- is not currently a Git repository;
- contains no IDRiD images, CSV files, archives, notebooks, source code, or hidden project configuration;
- has no IDRiD-named dataset or grading archive in the usual nearby Desktop, Downloads, Documents, or OneDrive locations searched.

Therefore, local file counts, decoded pixel dimensions, corrupt-image checks, filename/CSV joins, duplicate hashes, and the exact on-disk directory names **could not be verified from local data**. The dataset facts below are the expected official IDRiD grading-task facts and must be confirmed by an audit script as the first implementation gate once the data is placed in the workspace.

This distinction matters: published statistics help us plan, but they do not prove that a particular local download is complete or correctly arranged.

## 3. Expected IDRiD grading data

The official IDRiD description reports 516 grading images, captured with a 50-degree field of view and stored as JPEG at 4288 × 2848 pixels. The official disease-grading split is 413 training images and 103 testing images.

Expected relevant layout (folder punctuation and spaces can vary slightly between distributions):

```text
IDRiD/
└── B. Disease Grading/
    ├── 1. Original Images/
    │   ├── a. Training Set/       # 413 .jpg files
    │   └── b. Testing Set/        # 103 .jpg files
    └── 2. Groundtruths/
        ├── a. IDRiD_Disease Grading_Training Labels.csv
        └── b. IDRiD_Disease Grading_Testing Labels.csv
```

The CSVs are expected to contain an image identifier, a DR grade, and a DME risk grade. This project will read only the image identifier and DR grade while preserving the other column without using it as an input feature.

The dataset also contains segmentation and optic-disc/fovea localization material. Those folders should be inventoried by the audit, but they should not be fed to this classifier. Using lesion masks or annotations unavailable for every inference image would change the task and can create an unfair evaluation.

Sources used for planning:

- [Official IDRiD data description](https://idrid.grand-challenge.org/Data/)
- [Peer-reviewed IDRiD data descriptor](https://www.mdpi.com/2306-5729/3/3/25)
- [Published IDRiD-DR split and class table](https://pmc.ncbi.nlm.nih.gov/articles/PMC12472159/)

### Expected image and label statistics

| Split | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Official training | 134 (32.45%) | 20 (4.84%) | 136 (32.93%) | 74 (17.92%) | 49 (11.86%) | 413 |
| Official test | 34 (33.01%) | 5 (4.85%) | 32 (31.07%) | 19 (18.45%) | 13 (12.62%) | 103 |
| Total | 168 (32.56%) | 25 (4.84%) | 168 (32.56%) | 93 (18.02%) | 62 (12.02%) | 516 |

### Class-imbalance finding

The dataset is imbalanced. In the training set, the largest class has 136 images and the smallest has 20, a **6.8:1 ratio**. Mild DR is especially difficult: it represents only 4.84% of training data, and the official test set contains only five mild cases.

This has three practical consequences:

1. A model can achieve deceptively reasonable accuracy while almost never recognizing Mild DR.
2. Per-class precision/recall and macro F1 are necessary because macro averaging gives every grade equal importance.
3. The validation split will contain only about four Mild DR images, so small metric changes are noisy and must not be overinterpreted.

## 4. Day-one data audit gate

Before training, create and run one read-only audit command that produces a JSON and Markdown report. Training must stop if any required check fails.

The audit will:

1. Recursively inventory all files and identify the two grading image directories and two grading CSVs.
2. Normalize only filename extensions when joining CSV IDs to files; never guess missing identifiers silently.
3. Verify expected row and image counts: 413 train, 103 test, 516 total.
4. Verify one-to-one joins: every label has exactly one image and every image has exactly one label.
5. Check that DR grades are integers in `{0, 1, 2, 3, 4}` and report missing values or duplicate CSV rows.
6. Decode every image with Pillow, record width, height, mode, and file size, and fail on corrupt files.
7. Confirm the expected 4288 × 2848 RGB/JPEG format while reporting any exceptions rather than forcibly hiding them.
8. Recompute class counts independently for train and test and compare them with the table above.
9. Calculate SHA-256 hashes and a perceptual hash to detect exact and near duplicates within and across splits.
10. Save a class-distribution chart and a small contact sheet for visual orientation only; do not modify source images.

The raw dataset should remain immutable and excluded from Git. Any cached, cropped images must go into a separate generated directory with preprocessing parameters recorded.

## 5. Dataset splitting policy

Keep the official 103-image test set sealed until the final evaluation.

From the official 413-image training set, create one fixed, stratified 80/20 development split:

- about 330 images for fitting;
- about 83 images for validation;
- a fixed seed stored in configuration;
- the generated manifest committed to the repository so every run uses the same samples.

Stratification approximately preserves the five-grade distribution. It cannot make the validation estimate stable when a class has only 20 total training examples, but it is the simplest defensible approach for a three-day build.

If a patient identifier is available, grouping by patient takes priority over ordinary stratification so that images from the same patient never appear in both subsets. If no patient identifier is supplied, document that limitation and use image-level splitting. Do not infer a patient relationship from filename patterns without evidence.

Model selection and threshold-free tuning happen only on validation data. The official test set is run once for the final frozen model. If a final model is retrained on all 413 training images, use the epoch count and settings chosen during development; do not use test results to decide when to stop or which checkpoint to keep.

Five-fold stratified cross-validation would make better use of this small dataset, but it multiplies training time and complexity. It is a stretch goal only after the end-to-end baseline and demo work reliably.

## 6. Preprocessing

Use one deterministic preprocessing function for training, validation, testing, and inference before applying training-only augmentation:

1. Open with Pillow and convert to RGB.
2. Detect and crop only the surrounding near-black border, retaining the complete retinal field of view.
3. Pad to a square if necessary instead of stretching the retina.
4. Resize to 224 × 224 for the initial EfficientNet-B0 baseline.
5. Convert to a tensor and normalize with the ImageNet mean and standard deviation expected by the pretrained `timm` weights.

Why this choice:

- Removing the black frame prevents the network from spending capacity on irrelevant pixels.
- Preserving the full field avoids losing small peripheral lesions that help distinguish severe grades.
- Square padding avoids geometric distortion.
- ImageNet normalization matches the statistics used to pretrain the backbone.
- A 224-pixel baseline trains quickly. If time and GPU memory permit, 320 or 384 pixels is the single most useful resolution experiment because DR lesions can be very small.

Do not add CLAHE, heavy denoising, vessel extraction, or lesion segmentation to the initial baseline. Each adds a new source of bugs and can shift colors away from the pretrained model's expectations. CLAHE can be one controlled experiment only if the plain baseline is complete.

The border detector must be tested visually. A bad crop that clips retinal edges is worse than leaving some black border.

## 7. Training-only augmentation

Use conservative transformations that simulate camera and positioning variation without erasing pathology:

- random horizontal flip, probability 0.5;
- small rotation, approximately ±15 degrees;
- small translation and scale variation, for example up to 5% translation and 5% scale change;
- mild brightness, contrast, saturation, and hue jitter;
- optional very light blur or noise at low probability if visual QA shows lesions remain visible.

Avoid aggressive random crops, CutMix/MixUp, large erasing regions, or strong blur in the first baseline. Microaneurysms and small hemorrhages are tiny; an augmentation can accidentally delete the evidence that defines a grade. Validation, test, and inference transforms must be deterministic and contain no random augmentation.

Any transform change requires a contact sheet showing before/after examples from every grade.

## 8. Model and transfer-learning pipeline

### Baseline

- Library: PyTorch plus `timm`.
- Backbone: `efficientnet_b0`, ImageNet pretrained.
- Head: replace the classifier with a five-logit linear layer.
- Output: softmax probabilities for grades 0–4; training consumes raw logits.
- Task formulation: ordinary five-class classification for the three-day baseline.

Although grades are ordered, standard multiclass classification is easier to implement and debug. Quadratic weighted kappa will still penalize distant mistakes more than adjacent mistakes. Ordinal-regression heads are a post-baseline experiment, not a dependency for delivery.

### Imbalance handling

Use class-weighted cross-entropy, with weights computed **only from the current fitting subset** and saved with the run. Prefer normalized inverse-square-root frequency weights over raw inverse frequency so the 20 Mild DR samples receive extra attention without dominating training.

Do not initially combine a weighted loss and a weighted sampler; applying both can overcorrect the imbalance. If recall for Mild DR remains near zero, compare one controlled alternative using a weighted sampler or focal loss, changing only that component.

### Training recipe

- Freeze the pretrained backbone and train the new head for 1–2 warm-up epochs.
- Unfreeze the network and fine-tune all layers.
- Optimizer: AdamW.
- Starting learning rates: roughly `3e-4` for the head and `1e-4` for the backbone; confirm with a short run rather than a large sweep.
- Weight decay: about `1e-4`.
- Schedule: cosine decay, optionally with a short warm-up.
- Epoch budget: 15–25 with early stopping patience around five epochs.
- Checkpoint criterion: highest validation quadratic weighted kappa; use macro F1 as a tie-breaker and inspect per-class recall.
- Use automatic mixed precision on CUDA, gradient clipping, deterministic seeds where practical, and a batch size that fits the available GPU (start with 16 or 32 at 224 pixels).
- Store the resolved configuration, package versions, seed, split manifest, metrics history, and best checkpoint for reproducibility.

Only a small experiment matrix is realistic:

1. EfficientNet-B0 at 224 pixels with class-weighted cross-entropy.
2. If time permits, the same run at 320 or 384 pixels.
3. Only if Mild DR recall is still unusable, compare one imbalance-treatment alternative.

Change one factor at a time. Do not launch a broad hyperparameter search on 413 images.

## 9. Evaluation design

Calculate metrics from the same fixed predictions and store both machine-readable JSON and human-readable tables/figures.

Required metrics:

- **Accuracy:** fraction of exactly correct grades. Easy to understand, but can be dominated by common classes.
- **Macro F1:** calculate F1 independently for each grade and average equally. This exposes failure on rare classes.
- **Per-class precision:** among images predicted as a grade, how many really have that grade.
- **Per-class recall:** among images truly in a grade, how many the model found.
- **Per-class support:** number of test images for each grade, always displayed beside precision and recall.
- **Confusion matrix:** raw counts and row-normalized percentages, with axes and grade names clearly labeled.
- **Quadratic weighted Cohen's kappa (QWK):** measures agreement while assigning a larger penalty to predictions farther from the true ordered grade; predicting 4 for a true 0 is worse than predicting 1.

Also report the validation loss and optionally ordinal mean absolute error as a helpful diagnostic, but do not replace the required metrics.

Because the test set has only five Mild DR images, report exact support and avoid claims of clinical reliability. If time permits, add bootstrap 95% confidence intervals for accuracy, macro F1, and QWK, clearly noting that the sample is small.

Evaluation output should include:

```text
outputs/<run_name>/
├── config.yaml
├── history.csv
├── best.pt
├── metrics.json
├── classification_report.csv
├── confusion_matrix_counts.png
├── confusion_matrix_normalized.png
└── predictions.csv   # image_id, true_grade, predicted_grade, five probabilities
```

## 10. Simple repository architecture

Keep modules small and avoid a framework inside the project:

```text
dr-retinopathy-grading/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── baseline.yaml
├── data/
│   ├── README.md                 # where IDRiD must be placed; no images in Git
│   ├── raw/                      # immutable, gitignored IDRiD download
│   ├── manifests/                # audited train/val/test CSVs
│   └── cache/                    # optional generated crops, gitignored
├── src/
│   └── dr_grading/
│       ├── __init__.py
│       ├── data.py               # audit-aware dataset and transforms
│       ├── model.py              # timm model factory
│       ├── engine.py             # train/validation loops and checkpoints
│       ├── metrics.py            # reports, QWK, confusion matrices
│       └── utils.py              # seeds, config, logging, device helpers
├── scripts/
│   ├── audit_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py                        # minimal Streamlit upload demo
├── tests/
│   └── test_smoke.py             # joins, transforms, forward pass, metric sanity
└── outputs/                      # checkpoints/results, mostly gitignored
```

Command-line scripts and one YAML configuration are enough. Hydra, experiment databases, distributed training, containers, APIs, and a separate frontend/backend are unnecessary for this deadline.

## 11. Leakage and evaluation mistakes to avoid

1. **Using official test data during tuning.** Do not choose preprocessing, loss, resolution, epoch, or checkpoint after looking at test metrics.
2. **Splitting after augmentation.** Split original images first; augmented versions of one image must remain in the same subset.
3. **Patient or paired-eye leakage.** Group by patient when metadata exists. Two eyes or repeated captures from one patient can be more alike than unrelated images.
4. **Exact or near duplicates across splits.** Check cryptographic and perceptual hashes before training.
5. **Fitting preprocessing on all data.** Any learned mean/std, threshold, or class weight comes from the fitting subset only. Fixed ImageNet normalization is not fitted on IDRiD.
6. **Using DME labels or lesion masks as hidden shortcuts.** The requested input is a fundus image. Auxiliary ground truth must not be available to the model at evaluation or inference.
7. **Mismatched image/label joins.** Never rely on row order. Join using a normalized explicit image identifier and require one-to-one matching.
8. **Applying random augmentation at evaluation.** Validation/test preprocessing must be deterministic unless a documented test-time augmentation average is used.
9. **Oversampling validation or test data.** Rebalancing belongs only in training. Test metrics must reflect the untouched official distribution.
10. **Reporting only accuracy.** Always show macro F1, class-wise metrics, supports, confusion matrices, and QWK.
11. **Selecting a checkpoint on training loss.** Use fixed validation data and a predeclared selection metric.
12. **Inconsistent label order.** Centralize the mapping `0..4`; label names, output logits, plots, and the UI must all use the same order.
13. **Preprocessing train/test differently.** Border cropping, padding, resizing, channel order, and normalization must be shared.
14. **Claiming general clinical performance.** IDRiD is small and comes from a particular acquisition setting. A good internal test score does not establish safety across hospitals, cameras, populations, or ungradable images.

## 12. Three-day execution schedule

### Day 1 — Data certainty and working baseline

**Morning**

- Create the minimal repository skeleton, environment file, and configuration.
- Place the licensed IDRiD data under `data/raw/` without renaming or modifying source files.
- Implement and run the full audit gate.
- Resolve every missing label, duplicate, corrupt image, unexpected dimension, or class-count discrepancy before training.
- Generate and freeze the stratified development manifest.

**Afternoon**

- Implement deterministic border crop/pad/resize and conservative training transforms.
- Visually inspect transform contact sheets for all five grades.
- Implement the PyTorch dataset, EfficientNet-B0 model factory, weighted loss, training loop, and required metrics.
- Run smoke tests: one batch loads, one forward/backward pass succeeds, and a tiny overfit test can fit a handful of examples.

**Evening deliverable**

- Start and complete the first 224-pixel baseline run.
- Save a reproducible checkpoint and validation report.

**Exit criterion:** data audit passes, no test data has entered development, and an end-to-end train/evaluate/predict path works.

### Day 2 — Improve one bottleneck and freeze the model

**Morning**

- Review learning curves, the validation confusion matrix, macro F1, QWK, and especially Mild/Severe/PDR recall.
- Inspect misclassified images without changing labels.
- Fix pipeline bugs before attempting model changes.

**Afternoon**

- Run at most one resolution experiment (320 or 384) if GPU time permits.
- If the main failure is rare-class recall, run one alternative imbalance experiment instead of a broad search.
- Compare runs using the same validation manifest and predeclared metrics.

**Evening deliverable**

- Freeze preprocessing, configuration, label mapping, and model choice.
- Optionally retrain the chosen configuration on all 413 official training images for the previously selected epoch count.
- Run the official test evaluation once and archive its predictions and figures.

**Exit criterion:** one frozen model has a complete, honest test report containing every required metric.

### Day 3 — Demo, verification, and presentation

**Morning**

- Build a minimal Streamlit interface: image upload, preview, predicted grade, five probabilities, model/version information, and disclaimer.
- Ensure inference calls the exact deterministic preprocessing used in evaluation.
- Handle invalid files and CPU fallback cleanly.

**Afternoon**

- Add `predict.py` for a single image and a small batch folder.
- Write setup, dataset-placement, training, evaluation, and demo instructions in README.
- Run smoke tests from a clean process and confirm the checkpoint loads on the demo machine.
- Verify that dataset images, credentials, and large checkpoints are not accidentally committed.

**Final hackathon package**

- Reproducible source and dependency list.
- Frozen model checkpoint or an explicit download location.
- Dataset audit report and split manifests.
- Test metrics JSON, classification report, confusion matrices, and predictions.
- Working local demo.
- Short presentation explaining dataset imbalance, transfer learning, QWK, limitations, and a few correctly/incorrectly classified examples.

**Exit criterion:** another team member can follow the README, launch the demo, reproduce inference, and explain the limitations.

## 13. Decision priorities if time slips

Cut features in this order:

1. Cross-validation and confidence intervals.
2. CLAHE or alternative-loss experiments.
3. Higher-resolution training.
4. Batch inference polish.

Never cut the data audit, sealed test policy, per-class metrics, QWK, reproducibility metadata, or the inference/preprocessing consistency check. Those protect the validity of the result.

## 14. Definition of done

The three-day project is complete when:

- the local IDRiD download passes the documented audit;
- one EfficientNet-B0 model can train reproducibly from the official training split;
- the official test split remains untouched until final evaluation;
- accuracy, macro F1, per-class precision/recall/support, both confusion matrices, and QWK are saved;
- a single uploaded fundus image produces grade 0–4 probabilities through the same preprocessing as evaluation;
- the code, configuration, checkpoint provenance, split manifest, and limitations are documented;
- the UI clearly states that the prototype is not a substitute for professional diagnosis.
