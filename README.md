# Fetal Head Ultrasound Segmentation

## Overview
This project applies deep learning (U-Net) to segment fetal head regions in ultrasound scans.  
Dataset: [Fetal Head Ultrasound Dataset](https://datasetninja.com/fetal-head-ultrasound).

## Steps
1. Data preprocessing (resize, normalize, augment)
2. Model: U-Net architecture
3. Training (20 epochs)
4. Evaluation (Dice, IoU)
5. Visualization: input, ground truth mask, predicted mask

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
