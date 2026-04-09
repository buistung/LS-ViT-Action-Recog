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

## Requirements
- Python 3.10+
- pip

## Installation
Clone the repository:
```
git clone https://github.com/buistung/LS-ViT-Action-Recog.git
cd video-action-recognition-lsvit
```
Create a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
## Dataset Preparation
Run:
```
python scripts/prepare_data.py
```
## Train the model
Run:
```
python scripts/train.py --data-root ./data/processed/hmdb51 --epochs 10 --batch-size 4
```
## Run inference
```
python scripts/predict.py \
  --video-dir path/to/frame_folder \
  --checkpoint checkpoints/lsvit_hmdb51_best.pt \
  --data-root ./data/processed/hmdb51
```
## Important Notes
- This project currently uses frame folders as input, not raw `.mp4` video files.
- To change default settings, edit:
  - `src/lsvit_action/config.py`
- The codebase is organized into:
  - `data`: dataset handling, transforms, dataloaders
  - `models`: LSViT model and related modules
  - `engine`: training, evaluation, checkpoint handling
  - `utils`: logging, seed control, I/O, visualization
