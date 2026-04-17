#!/bin/bash
# Run all 8 methods × 3 runs × 2 data settings (IID and non-IID)
# Total: 8 × 3 × 2 = 48 experiments

set -e

for stratified in 0 1; do
  for method in $(seq 0 7); do
    for run in $(seq 1 3); do
      if [ $stratified -eq 0 ]; then
        echo "Running method=$method run=$run stratified=true"
        python main.py -t $method -r $run -s
      else
        echo "Running method=$method run=$run stratified=false"
        python main.py -t $method -r $run
      fi
    done
  done
done

echo "All experiments completed."
