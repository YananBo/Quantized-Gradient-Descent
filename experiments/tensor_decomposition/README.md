# Experiment: Distributed Tucker Decomposition

This experiment applies QGD to **Tucker tensor decomposition** — factorizing a 3-D tensor `A ≈ G ×₁ U⁽¹⁾ ×₂ U⁽²⁾ ×₃ U⁽³⁾` where `G` is a small core tensor and `U⁽ⁿ⁾` are factor matrices. The optimization landscape is nonconvex with many saddle points; QGD's switching quantization provides the perturbation needed to escape them and converge to better solutions.

---

## Quick Start

```bash
# Ensure the input tensor exists
ls data/observed.npy

# Run QGD Tucker decomposition (default: 100K iterations, rank [5,5,5])
python tensor_decomposition.py --data data/observed.npy

# Custom rank and quantization
python tensor_decomposition.py --data data/observed.npy \
                               --ranks 3 3 3 \
                               --quantization-levels 128 \
                               --iterations 50000
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/observed.npy` | Path to input tensor (`.npy` file) |
| `--ranks` | `5 5 5` | Tucker rank for each mode |
| `--n-agents` | `5` | Number of distributed agents |
| `--iterations` | `100001` | Total optimization iterations |
| `--quantization-levels` | `64` | QSGD quantization levels |
| `--penalty` | `0.1` | L2 regularization on factor matrices |
| `--lr-consensus` | `0.03` | Initial consensus learning rate |
| `--lr-gradient` | `0.003` | Initial gradient learning rate |
| `--t0` | `100` | First holding-stage boundary |
| `--seed` | `1234` | Random seed |
| `--log-every` | `1000` | Logging interval (iterations) |
| `--output-dir` | `results` | Output directory |

---

## Algorithm Details

At each iteration, every agent:

1. **Quantizes** its parameters `[G, U⁽¹⁾, U⁽²⁾, U⁽³⁾]` using even/odd switching quantization
2. **Consensus**: receives quantized parameters from neighbors, computes weighted average via mixing matrix `W`
3. **Gradient step**: reconstructs the tensor from its own factors, computes the loss gradient, and updates

The loss per agent is:

```
L = ‖A - tucker_to_tensor(G, [U⁽¹⁾, U⁽²⁾, U⁽³⁾])‖₂ + λ Σ ‖U⁽ⁿ⁾‖²_F
```

---

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Quantization levels | 64 | QSGD granularity |
| Tucker rank | (5, 5, 5) | Rank for each tensor mode |
| LR consensus init | 0.03 | Initial consensus learning rate |
| LR gradient init | 0.003 | Initial gradient learning rate |
| LR decay α | 0.6 | Consensus decay exponent |
| LR decay β | 0.9 | Gradient decay exponent |
| LR regularizer | 0.3 | Denominator growth rate |
| Holding ck | 80 | Holding-stage duration parameter |
| Holding start (t₀) | 100 | First holding-stage boundary |
| L2 penalty | 0.1 | Regularization on factor matrices |
| Agents | 5 | Number of distributed agents |
| Iterations | 100,001 | Total iterations |

---

## Output

Results are saved to `results/` as CSV with the naming pattern:

```
tucker_qgd_r{rank}x{rank}x{rank}_l{levels}_{timestamp}.csv
```

Columns: `checkpoint, rec_error, loss_agent0, ..., loss_agent4`

---

## Project Structure

```
tensor_decomposition/
├── tensor_decomposition.py   # Main script (config, quantization, training loop)
├── data/
│   └── observed.npy          # Input tensor (user-provided)
└── results/                  # Output CSV files (auto-created)
```
