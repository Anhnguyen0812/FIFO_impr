#!/bin/bash
# Train Stage 2 on Kaggle T4 (single GPU) with frequency loss and fog-pass fine-tuning.
# Fill in the two paths below before running.

set -euo pipefail

# <<< EDIT THESE: set your pretrained checkpoints >>>
PRETRAINED_SEG="/kaggle/input/fifo-pretrained/Cityscapes_pretrained_model.pth"
PRETRAINED_FOGPASS="/kaggle/input/fifo-pretrained/FogPassFilter_pretrained.pth"

# WandB offline to avoid network prompts on Kaggle
export WANDB_MODE=offline
# Choose which GPU to use (Kaggle often gives GPU 0 as first T4)
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --modeltrain train \
  --file-name FIFO_freq_t4 \
  --restore-from "$PRETRAINED_SEG" \
  --restore-from-fogpass "$PRETRAINED_FOGPASS" \
  --batch-size 1 \
  --iter-size 1 \
  --accum-steps 2 \
  --num-workers 2 \
  --save-pred-every 2000 \
  --snapshot-dir /kaggle/working/snapshots/FIFO_model \
  --gpu 0
