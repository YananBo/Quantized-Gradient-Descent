# Experiment: Binary Classification (Saddle-Point Escape)

This experiment demonstrates QGD's ability to escape saddle points on a simple 2-D regularized logistic regression landscape. The objective function `f(x₁, x₂) = log(1 + exp(-x₁x₂)) + (λ/2)‖x‖²` has a **saddle point at the origin** — the key test case.

A network of `n` agents collaborates via a doubly-stochastic mixing matrix, communicating only *quantized* iterates. The switching quantization scheme ensures persistent perturbations that drive the iterates away from the saddle point, while vanilla decentralized GD (DGD) gets stuck.

---

## Quick Start

```bash
# QGD (proposed) — escapes the saddle point
python binary_classification.py --method qgd --iterations 8000

# Vanilla DGD (baseline) — gets stuck at the saddle point
python binary_classification.py --method dgd --iterations 8000
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `qgd` | Optimizer: `qgd` (proposed) or `dgd` (baseline) |
| `--iterations` | `8000` | Total optimization iterations |
| `--n-agents` | `5` | Number of distributed agents |
| `--quantization-levels` | `10` | QSGD levels (higher = more bits, less noise) |
| `--topology` | `connected` | Network graph: `connected`, `ring`, or `centralized` |
| `--lr-init` | `1.0` | Initial learning rate |
| `--lr-alpha` | `0.62` | Consensus LR decay exponent |
| `--lr-beta` | `0.94` | Gradient LR decay exponent |
| `--t0` | `10` | First holding-stage boundary |
| `--seed` | `42` | Random seed |
| `--output-dir` | `results` | Directory for CSV output |

---

## What to Expect

**QGD**: Agents escape the saddle point at the origin and converge toward a second-order stationary point where loss decreases significantly.

**DGD**: Without quantization noise, all agents remain trapped near the saddle point with stagnant loss.

---

## Output

Results are saved to `results/binary_{method}_n{agents}_iter{iterations}.csv` with columns:

```
iteration, loss, x0_agent0, x1_agent0, ..., x0_agentN, x1_agentN, lr_consensus, lr_gradient
```

---

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Quantization levels | 10 | Controls perturbation magnitude |
| Initial LR | 1.0 | Starting learning rate |
| LR decay (α) | 0.62 | Consensus step decay exponent |
| LR decay (β) | 0.94 | Gradient step decay exponent |
| Holding start (t₀) | 10 | First stepsize-holding boundary |
| L2 regularization | 0.1 | Regularization coefficient |
| Agents | 5 | Number of distributed agents |
