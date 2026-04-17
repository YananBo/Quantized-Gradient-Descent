#!/bin/bash
# Run DGD baseline on Robust PCA 20 times for error bars
set -e

for run in $(seq 1 20); do
  echo "DGD run $run / 20"
  python rpca_dgd.py
done

echo "All DGD runs completed."