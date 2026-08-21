# IDRiD Diabetic Retinopathy Grading

Hackathon project for transfer-learning-based classification of retinal fundus images into five diabetic retinopathy severity grades:

0. No DR
1. Mild DR
2. Moderate DR
3. Severe DR
4. Proliferative DR

The planned baseline is ImageNet-pretrained EfficientNet-B0 using PyTorch and `timm`. See [PLAN.md](PLAN.md) for the inspected dataset assumptions, evaluation design, leakage safeguards, repository architecture, and three-day schedule.

## Dataset

IDRiD is third-party research data and is not stored in this Git repository. Place or generate the disease-grading portion under `data/raw/IDRiD/` as documented in [data/README.md](data/README.md).

Dataset source: the complete Kaggle mirror [Indian Diabetic Retinopathy Image Dataset (IDRiD)](https://www.kaggle.com/datasets/abdullahshafi315/indian-diabetic-retinopathy-image-datasetidrid), with the official [IEEE DataPort release](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) retained as the provenance reference. The Kaggle page identifies the mirror as CC BY-NC 4.0; keep the raw files private and follow that license until the official archive is obtained directly.

## Status

Repository and grading dataset preparation only. The local copy has been arranged and audited; the exact-duplicate warning in [data/README.md](data/README.md) must be handled before training. Model implementation has not started.
