# LSViT Video Action Recognition

This project is a refactored version of the original Jupyter Notebook into a clean, modular Python project for training and inference on a video action recognition model using **frame folders** as input.

## Overview

This repository implements a video action recognition pipeline that includes:

- HMDB51 dataset preparation
- Video preprocessing and augmentation
- **LSViT** model with motion-aware modules:
  - **SMIF** (Spatial Motion Information Fusion)
  - **LMI** (Local Motion Interaction)
- Training and validation
- Checkpoint saving
- Inference on a single video clip stored as a frame folder

## Project Structure

```text
video-action-recognition-lsvit/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml
├── notebooks/
│   └── video-action-recognition-LSViT.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── outputs/
│   ├── figures/
│   └── logs/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
├── src/
│   └── lsvit_action/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── transforms.py
│       │   └── dataloaders.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── layers.py
│       │   ├── motion.py
│       │   └── lsvit.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── checkpoint.py
│       │   ├── evaluator.py
│       │   └── trainer.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           ├── logging_utils.py
│           ├── seed.py
│           └── visualization.py
└── tests/
