#!/usr/bin/env python3
import matplotlib
import config as cfg
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


with open(cfg.HISTORY_PATH, "r") as f:
    history_dict = eval(f.read())

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history_dict["loss"], label="Train Loss")
plt.plot(history_dict["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

# Dice
if "dice_metric" in history_dict:
    plt.subplot(1, 3, 2)
    plt.plot(history_dict["dice_metric"], label="Train Dice")
    plt.plot(history_dict["val_dice_metric"], label="Val Dice")
    plt.legend()
    plt.title("Dice Coefficient")

# IoU
plt.subplot(1, 3, 3)
plt.plot(history_dict["mean_io_u"], label="Train IoU")
plt.plot(history_dict["val_mean_io_u"], label="Val IoU")
plt.legend()
plt.title("Mean IoU")

plt.tight_layout()
plt.savefig(f"{cfg.RESULTS_DIR}/training_curves.png")
print("📊 Saved training curves to .results/training_curves.png")
