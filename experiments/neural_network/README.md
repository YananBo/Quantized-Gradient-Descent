# Experiment: Distributed Neural Network Training

This experiment benchmarks QGD against 7 baseline decentralized optimizers on image classification tasks (CIFAR-10 and MNIST). A network of 5 agents trains CNN models collaboratively, each holding a local data shard and communicating compressed parameters with neighbors.

---

## Quick Start

```bash
# QGD (proposed method) on CIFAR-10, IID data split
python main.py -t 0 -r 1 -s

# CDSGD baseline for comparison
python main.py -t 1 -r 1 -s

# Run all methods × 3 runs × 2 data settings
bash scripts/run_all.sh
```

---

## Implemented Methods

| Index (`-t`) | Method | Class | Description |
|:---:|--------|-------|-------------|
| 0 | **QGD** (ours) | `QGD` | Quantized GD with switching quantization + stepsize holding |
| 1 | CDSGD | `CDSGD` | Consensus-based decentralized SGD |
| 2 | CDSGD-P | `CDSGDP` | CDSGD + Polyak heavy-ball momentum |
| 3 | CDSGD-N | `CDSGDN` | CDSGD + Nesterov momentum |
| 4 | D-LAS | `DLAS` | Decentralized learning with adaptive stepsize |
| 5 | DAMSGrad | `DAMSGrad` | Decentralized AMSGrad |
| 6 | DAdaGrad | `DAdaGrad` | Decentralized AdaGrad |
| 7 | DAdSGD | `DAdSGD` | Decentralized adaptive SGD |

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `-t, --test_num` | Algorithm index (0–7, see table above) |
| `-r, --run_num` | Run number for repeated trials |
| `-s, --stratified` | Use IID (stratified) data partitioning; omit for non-IID |

---

## Data Partitioning

| Mode | Flag | Description |
|------|------|-------------|
| **IID** | `-s` | Dataset randomly split into 5 equal shards |
| **Non-IID** | (omit `-s`) | Each agent holds 50% of 2 classes + small fractions of others |

---

## Network Topology

All 5 agents communicate via a fixed doubly-stochastic mixing matrix:

```
W = [[0.6, 0.2, 0.0, 0.2, 0.0],
     [0.2, 0.8, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.6, 0.1, 0.3],
     [0.2, 0.0, 0.1, 0.3, 0.4],
     [0.0, 0.0, 0.3, 0.4, 0.3]]
```

---

## Project Structure

```
neural_network/
├── main.py          # Entry point — parses args and dispatches to trainers
├── ops.py           # All optimizer implementations (QGD + 7 baselines)
├── models.py        # CNN architectures: CifarCNN, MnistCNN
├── train.py         # Distributed training framework + per-method trainers
├── cuda.py          # CUDA availability check utility
└── scripts/
    └── run_all.sh   # Batch runner: all methods × runs × data settings
```

---

## Default Hyperparameters (QGD)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Quantization levels (`d`) | 1100 | Controls quantization granularity |
| Initial LR (consensus) | 0.5 | Learning rate for consensus step |
| Initial LR (gradient) | 0.5 | Learning rate for gradient step |
| LR decay α | 0.6 | Consensus LR decay exponent |
| LR decay β | 0.9 | Gradient LR decay exponent |
| Holding start (t₀) | 200 | First stepsize-holding boundary |
| Batch size | 32 | Per-agent mini-batch size |
| Epochs | 4000 | Total training epochs |

---

## Output

Results are saved to `results/` as CSV files with the naming pattern:

```
{dataset}_e{epochs}_hom{stratified}_{test_num}_{run_num}.csv
```

Each file contains: training iterations, training accuracy, test accuracy, loss values, and per-agent learning rate logs.
