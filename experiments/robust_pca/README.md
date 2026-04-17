# Experiment: Robust PCA (Video Background Separation)

This experiment applies QGD to **Robust Principal Component Analysis** — decomposing a data matrix `D` into a low-rank component `L = UV` (background) and a sparse component `S` (foreground). The nonconvex factored formulation `D ≈ UV + S` introduces saddle points; QGD's quantization noise helps escape them.

The dataset consists of video frames from the [Wallflower benchmark](https://www.microsoft.com/en-us/download/details.aspx?id=54651), where the goal is to separate static backgrounds from moving foreground objects.

---

## Quick Start

### 1. Download the Dataset

Download the "Test Images for Wallflower Paper" from:
[https://www.microsoft.com/en-us/download/details.aspx?id=54651](https://www.microsoft.com/en-us/download/details.aspx?id=54651)

Extract the video sequences into a `videos/` directory:

```
robust_pca/
├── videos/
│   ├── Bootstrap/
│   ├── Camouflage/
│   ├── ForegroundAperture/
│   ├── LightSwitch/
│   ├── MovedObject/
│   ├── TimeOfDay/
│   └── WavingTrees/
├── rpca_qgd.py
├── rpca_dgd.py
└── ...
```

### 2. Run Experiments

```bash
# QGD (proposed) — escapes saddle points in the nonconvex RPCA landscape
python rpca_qgd.py

# DGD baseline — vanilla decentralized gradient descent
python rpca_dgd.py

# Batch runs for error bars (20 runs each)
bash scripts/run_qgd.sh
bash scripts/run_dgd.sh
```

---

## Methods

| Script | Method | Description |
|--------|--------|-------------|
| `rpca_qgd.py` | **QGD** | Quantized GD with switching quantization + stepsize holding |
| `rpca_dgd.py` | DGD | Vanilla decentralized gradient descent (no quantization) |

Both methods use the same RPCA formulation. The key difference: QGD quantizes the `U` and `V` matrices before the consensus step, providing the stochastic perturbation needed to escape saddle points.

---

## RPCA Formulation

Each agent `i` solves:

```
minimize  ‖Dᵢ - UᵢVᵢ - Sᵢ‖² + (1/μ)(‖Uᵢ‖² + ‖Vᵢ‖²)
```

where `Dᵢ` is the local data matrix (flattened video frames), `Uᵢ ∈ ℝ^{m×r}` and `Vᵢ ∈ ℝ^{r×n}` parameterize the low-rank background, and `Sᵢ` is the sparse foreground component updated via a thresholding operator.

---

## Default Hyperparameters

### QGD

| Parameter | Value | Description |
|-----------|-------|-------------|
| Quantization levels | 32 | QSGD granularity |
| LR consensus init | 0.003 | Initial consensus learning rate |
| LR gradient init | 0.0003 | Initial gradient learning rate |
| LR decay α | 0.6 | Consensus decay exponent |
| LR decay β | 0.9 | Gradient decay exponent |
| Holding start (t₀) | 100 | First holding-stage boundary |
| Rank (`r`) | 30 | Low-rank approximation rank |
| Regularizer (μ) | 100 | Balances reconstruction vs. regularization |
| Sparsity (α) | 0.2 | Fraction of elements kept in sparse operator |
| Agents | 5 | Number of distributed agents |
| Iterations | 5000 | Maximum iterations |
| Convergence tol | 1e-6 | Early stopping threshold |

### DGD

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Rank | 30 |
| μ | 100 |
| Other params | Same as QGD |

---

## Output

Results are saved to `results/` as CSV files. Each checkpoint row contains:

```
iteration, loss_agent0, ..., loss_agent4, reconstruction_error
```

---

## Network Topology

Same 5-agent doubly-stochastic graph as other experiments:

```
W = [[0.6, 0.0, 0.0, 0.4, 0.0],
     [0.2, 0.8, 0.0, 0.0, 0.0],
     [0.2, 0.1, 0.4, 0.0, 0.3],
     [0.0, 0.0, 0.0, 0.6, 0.4],
     [0.0, 0.1, 0.6, 0.0, 0.3]]
```
