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

Dataset source: [Indian Diabetic Retinopathy Image Dataset (IDRiD)](https://idrid.grand-challenge.org/Data/), licensed under CC BY 4.0.

## Status

Repository and grading dataset preparation only. Model implementation has not started.
