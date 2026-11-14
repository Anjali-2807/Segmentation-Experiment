#!/usr/bin/env bash
# End-to-end execution script for reproducible segmentation experiment
# Author: Anjali Tiwari (CS107203CS - Unit 3)

set -e

echo "=========================================================="
echo " Starting Reproducible Segmentation Experiment Suite"
echo "=========================================================="

echo "[1/4] Training Baseline Model (Standard Bottleneck)..."
python3 train.py --mechanism baseline --epochs 8 --batch_size 16

echo "[2/4] Training Controlled Variant (Dilated Bottleneck)..."
python3 train.py --mechanism dilation --epochs 8 --batch_size 16

echo "[3/4] Running Exact Evaluation Script on Test Split..."
python3 evaluate.py --results_dir results

echo "[4/4] Generating Qualitative Comparative Panels..."
python3 visualize.py

echo "=========================================================="
echo " Experiment Suite Completed Successfully!"
echo " Check results/evaluation_summary.json and results/figures/"
echo "=========================================================="
