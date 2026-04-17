#!/bin/bash
# Run QGD on Robust PCA 20 times for error bars
set -e

for run in $(seq 1 20); do
  echo "QGD run $run / 20"
  python rpca_qgd.py
done

echo "All QGD runs completed."