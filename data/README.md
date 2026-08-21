# IDRiD data layout

The official disease-grading ZIP should be extracted locally to:

```text
data/raw/IDRiD/
└── B. Disease Grading/
    ├── 1. Original Images/
    │   ├── a. Training Set/
    │   └── b. Testing Set/
    └── 2. Groundtruths/
        ├── a. IDRiD_Disease Grading_Training Labels.csv
        └── b. IDRiD_Disease Grading_Testing Labels.csv
```

Expected grading counts:

| Split | Images | Grade distribution 0/1/2/3/4 |
|---|---:|---|
| Training | 413 | 134 / 20 / 136 / 74 / 49 |
| Testing | 103 | 34 / 5 / 32 / 19 / 13 |

The `data/raw/` directory is intentionally ignored by Git. Do not commit or push the retinal images.

The complete Kaggle mirror [used for this local intake](https://www.kaggle.com/datasets/abdullahshafi315/indian-diabetic-retinopathy-image-datasetidrid) was downloaded as `data/downloads/kaggle-idrid-complete.zip` (198,269,504 bytes; SHA-256 `532CDBA4B5B2DC05C5F95DAD357EC51A04097CFC4EA9DBC4D1713325E6F16A53`) and arranged without changing the source image or CSV contents. The official [IEEE DataPort IDRiD release](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) remains the provenance reference and should replace the mirror if direct access becomes available.

The audit report is tracked at `data/metadata/idrid_audit.json` and confirms 413 training images, 103 testing images, 4288×2848 RGB JPEGs, complete CSV joins, and the expected class counts. It also finds six exact duplicate-image groups, including one train/test duplicate (`train:IDRiD_118` and `test:IDRiD_064`), with conflicting labels in the duplicate groups. This is a data-leakage and label-integrity warning: do not silently use all 516 files as independent samples. Before training, create a duplicate-aware manifest, keep the official test split sealed, and exclude the training-side member of the cross-split duplicate (or obtain and re-audit the official archive). The earlier Hugging Face copy with the same issue is quarantined locally.
