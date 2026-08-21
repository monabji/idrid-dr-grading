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

Use the official [IEEE DataPort IDRiD release](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid). It requires a free IEEE account and exposes `B. Disease Grading.zip` after login.

An initially tested Hugging Face repackaging was rejected because a SHA-256 audit found six exact duplicate-image groups, including a train/test duplicate. That copy is quarantined locally and must not be used for training or evaluation.
