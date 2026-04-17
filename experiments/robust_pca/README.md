# Experiment: Video Background/Foreground Separation via Distributed Robust PCA

> **Industry relevance**: Video background subtraction is a core component in production surveillance systems (Amazon, Google, Ring), autonomous driving perception pipelines (Waymo, Tesla), video conferencing background blur (Zoom, Teams), and content moderation at scale. This experiment demonstrates how QGD enables **distributed, communication-efficient** video decomposition across edge devices — a key requirement when raw video cannot be centralized due to bandwidth or privacy constraints.

---

## Problem

Given a matrix `D` of video frames (each frame flattened as a row), decompose it into:

- **`L = UV`** — low-rank background (static scene)
- **`S`** — sparse foreground (moving objects, people, vehicles)

This is the classic **Robust PCA** problem. The factored formulation `D ≈ UV + S` is nonconvex and riddled with saddle points — standard gradient methods get stuck, producing blurry or incomplete foreground masks.

**QGD solves this**: its switching quantization injects structured perturbations that escape saddle points *for free*, while simultaneously compressing inter-device communication by up to 10× compared to full-precision exchange.

---

## Why This Matters at Scale

| Challenge | How QGD Addresses It |
|-----------|---------------------|
| **Edge deployment** — cameras can't stream raw video to a central server | Agents process local video shards; only quantized parameters are exchanged |
| **Bandwidth constraints** — network links between edge nodes are limited | Quantized communication reduces bits per iteration (adjustable via `--quantization-levels`) |
| **Privacy** — raw frames never leave the local device | Only compressed model parameters (U, V factors) are shared, not pixel data |
| **Nonconvex landscape** — saddle points degrade separation quality | Switching quantization provably escapes saddle points → cleaner foreground masks |

---

## Quick Start

### 1. Download the Dataset

This experiment uses the [Wallflower benchmark](https://www.microsoft.com/en-us/download/details.aspx?id=54651) — a standard video surveillance dataset with 7 challenging sequences (lighting changes, camouflage, waving trees, etc.).

```bash
# Download and extract into videos/ directory
mkdir -p videos
# Extract each sequence as a subdirectory:
#   videos/Bootstrap/, videos/Camouflage/, videos/ForegroundAperture/,
#   videos/LightSwitch/, videos/MovedObject/, videos/TimeOfDay/, videos/WavingTrees/
```

### 2. Run Experiments

```bash
# QGD (proposed) — escapes saddle points, cleaner separation
python rpca_qgd.py

# DGD baseline — vanilla decentralized GD, no quantization
python rpca_dgd.py

# Batch runs for error bars (20 runs each)
bash scripts/run_qgd.sh
bash scripts/run_dgd.sh
```

---

## Architecture

```
┌──────────┐     quantized U, V     ┌──────────┐
│ Camera 1 │◄───────────────────────►│ Camera 2 │
│ (Agent)  │    mixing matrix W      │ (Agent)  │
└────┬─────┘                         └────┬─────┘
     │              ┌──────────┐          │
     └──────────────► Camera 3 ◄──────────┘
                    │ (Agent)  │
                    └────┬─────┘
                         │
                    ┌────┴─────┐     ┌──────────┐
                    │ Camera 4 ├─────► Camera 5 │
                    │ (Agent)  │     │ (Agent)  │
                    └──────────┘     └──────────┘

Each agent holds local video frames → computes local U, V, S
→ quantizes U, V → exchanges with neighbors → updates via QGD
```

---

## Methods

| Script | Method | Quantization | Saddle-Point Escape |
|--------|--------|:---:|:---:|
| `rpca_qgd.py` | **QGD** (ours) | ✅ Switching even/odd | ✅ Provably escapes |
| `rpca_dgd.py` | DGD (baseline) | ❌ Full precision | ❌ Can get stuck |

---

## RPCA Formulation

Each agent `i` minimizes:

```
min  ‖Dᵢ - UᵢVᵢ - Sᵢ‖² + (1/μ)(‖Uᵢ‖² + ‖Vᵢ‖²)
```

- `Dᵢ ∈ ℝ^{m×n}` — local video frames (flattened, resized to 56×56)
- `Uᵢ ∈ ℝ^{m×r}`, `Vᵢ ∈ ℝ^{r×n}` — low-rank factors (background model)
- `Sᵢ` — sparse component (foreground), updated via top-k thresholding (α=0.2)

**QGD update rule** per iteration:
1. Quantize `Uᵢ`, `Vᵢ` (even/odd switching scheme)
2. Consensus: `Ũᵢ = Σⱼ wᵢⱼ Q(Uⱼ)` (weighted sum of quantized neighbors)
3. Gradient: `Uᵢ ← (1-η)Uᵢ + η·Ũᵢ - α·∇_U Loss`
4. Sparse update: `Sᵢ ← Threshold(Dᵢ - UᵢVᵢ)`

---

## Default Hyperparameters

### QGD

| Parameter | Value | Description |
|-----------|-------|-------------|
| Quantization levels | 32 | Bits per parameter (higher = finer, less noise) |
| LR consensus | 0.003 | Consensus step learning rate |
| LR gradient | 0.0003 | Gradient step learning rate |
| LR decay (α, β) | 0.6, 0.9 | Polynomial decay exponents |
| Holding start (t₀) | 100 | Stepsize-holding boundary |
| Holding ck | 40 | Holding-stage duration |
| Rank (`r`) | 30 | Background model rank |
| Regularizer (μ) | 100 | Reconstruction vs. regularization |
| Sparsity (α) | 0.2 | Fraction of elements kept in foreground |
| Agents | 5 | Distributed cameras / nodes |
| Max iterations | 5000 | Convergence threshold: rec. error < 1e-6 |

### DGD (baseline)

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Other params | Same as QGD (no quantization, no holding) |

---

## Output

Results are saved to `results/` as CSV. Each row:

```
iteration, loss_agent0, ..., loss_agent4, mean_reconstruction_error
```

---

## Network Topology

5-agent doubly-stochastic graph (sparse, realistic for edge networks):

```
W = [[0.6, 0.0, 0.0, 0.4, 0.0],
     [0.2, 0.8, 0.0, 0.0, 0.0],
     [0.2, 0.1, 0.4, 0.0, 0.3],
     [0.0, 0.0, 0.0, 0.6, 0.4],
     [0.0, 0.1, 0.6, 0.0, 0.3]]
```

---

## Project Structure

```
robust_pca/
├── rpca_qgd.py          # QGD-based robust PCA
├── rpca_dgd.py          # DGD baseline
├── scripts/
│   ├── run_qgd.sh       # 20× batch run for error bars
│   └── run_dgd.sh       # 20× batch run for error bars
├── videos/              # Wallflower dataset (user-provided)
│   ├── Bootstrap/
│   ├── Camouflage/
│   └── ...
├── results/             # Output CSVs (auto-created)
└── README.md
```
