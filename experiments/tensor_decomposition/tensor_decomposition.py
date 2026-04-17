"""Decentralized Tucker Decomposition with Quantized Gradient Descent (QGD).

This script performs distributed Tucker decomposition of a 3-D tensor across
multiple agents.  Each agent maintains its own copy of the core tensor and
factor matrices, exchanges *quantized* parameters with neighbors, and
updates via the QGD algorithm — demonstrating saddle-point escape on a
classic nonconvex factorization problem.

Usage
-----
    # Run with defaults
    python tensor_decomposition.py --data data/observed.npy

    # Custom rank and agents
    python tensor_decomposition.py --data data/observed.npy \
                                   --ranks 3 3 3 \
                                   --n-agents 5 \
                                   --iterations 50000

    # Adjust quantization and learning rate
    python tensor_decomposition.py --data data/observed.npy \
                                   --quantization-levels 128 \
                                   --lr-consensus 0.05 \
                                   --lr-gradient 0.005

Reference
---------
Bo & Wang, "Quantization Avoids Saddle Points in Distributed Optimization",
PNAS 121(17), 2024.  https://doi.org/10.1073/pnas.2319625121
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
from sklearn.utils import check_random_state

tl.set_backend("pytorch")


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class TuckerConfig:
    """Hyperparameters for decentralized Tucker decomposition."""

    # Data
    data_path: str = "data/observed.npy"
    ranks: tuple[int, ...] = (5, 5, 5)

    # Distributed setting
    n_agents: int = 5
    mixing_matrix: Optional[np.ndarray] = None  # defaults to 5-agent graph

    # Optimization
    n_iterations: int = 100_001
    penalty: float = 0.1  # L2 regularization on factor matrices

    # Quantization
    quantization_levels: int = 64

    # Learning-rate schedule (piecewise-constant holding strategy)
    lr_consensus_init: float = 0.03
    lr_gradient_init: float = 0.003
    lr_regularizer: float = 0.3
    lr_alpha: float = 0.6     # consensus LR decay exponent
    lr_beta: float = 0.9      # gradient LR decay exponent
    lr_holding_ck: float = 80  # holding-stage duration
    t0: int = 100              # first holding-stage boundary

    # Reproducibility
    seed: int = 1234

    # I/O
    log_every: int = 1000
    output_dir: str = "results"

    def __post_init__(self) -> None:
        if self.mixing_matrix is None:
            self.mixing_matrix = np.array([
                [0.6, 0.0, 0.0, 0.4, 0.0],
                [0.2, 0.8, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.4, 0.0, 0.3],
                [0.0, 0.0, 0.0, 0.6, 0.4],
                [0.0, 0.1, 0.6, 0.0, 0.3],
            ])


# ---------------------------------------------------------------------------
#  Quantization primitives (PyTorch)
# ---------------------------------------------------------------------------

def quantize_even(
    params: list[torch.Tensor],
    levels: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Standard QSGD quantization (even iterations).

    Parameters
    ----------
    params : list[torch.Tensor]
        Agent's parameters [core, factor_0, factor_1, ...].
    levels : int
        Number of quantization levels.
    device : torch.device
        Target device.

    Returns
    -------
    list[torch.Tensor]
        Quantized copies (gradient graph detached).
    """
    quantized = []
    for p in params:
        data = p.data
        norm = max(tl.norm(data, 2).item(), 1e-1)
        level_float = levels * torch.abs(data) / norm
        lower = torch.floor(level_float)
        prob = level_float - lower
        rounded = lower + (torch.rand_like(data, device=device) < prob).float()
        q = torch.sign(data) * norm * rounded / levels
        quantized.append(q)
    return quantized


def quantize_odd(
    params: list[torch.Tensor],
    levels: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Shifted-level quantization (odd iterations).

    Uses odd-numbered thresholds to ensure the quantization grid shifts
    between consecutive iterations — providing persistent perturbations
    that help escape saddle points.
    """
    quantized = []
    for p in params:
        data = p.data
        norm = max(tl.norm(data, 2).item(), 1e-1)
        scaled = 2 * levels * data / norm
        lower = torch.floor(scaled)
        lower_odd = torch.where(lower % 2 == 0, lower - 1, lower)
        prob = (scaled - lower_odd) / 2.0
        rounded = lower_odd + 2 * (torch.rand_like(data, device=device) < prob).float()
        q = norm * rounded / (2 * levels)
        quantized.append(q)
    return quantized


# ---------------------------------------------------------------------------
#  Stepsize scheduler (holding strategy)
# ---------------------------------------------------------------------------

class StepsizeScheduler:
    """Piecewise-constant learning-rate schedule with holding stages.

    Holds rates constant inside each stage so that quantization
    perturbations integrate long enough to escape saddle points.
    """

    def __init__(self, config: TuckerConfig) -> None:
        self.cfg = config
        first_boundary = config.t0 + math.ceil(
            config.lr_holding_ck
            / (config.lr_consensus_init
               / (1 + config.lr_regularizer * config.t0 ** config.lr_alpha))
        )
        self._boundaries: list[int] = [config.t0, first_boundary]

    def __call__(self, t: int) -> tuple[float, float]:
        """Return ``(lr_consensus, lr_gradient)`` for iteration *t*."""
        c = self.cfg
        if t <= self._boundaries[0]:
            lr_c = c.lr_consensus_init / (1 + c.lr_regularizer * t ** c.lr_alpha)
            lr_g = c.lr_gradient_init / (1 + c.lr_regularizer * t ** c.lr_beta)
        elif self._boundaries[-2] < t < self._boundaries[-1]:
            prev = self._boundaries[-2]
            lr_c = c.lr_consensus_init / (1 + c.lr_regularizer * prev ** c.lr_alpha)
            lr_g = c.lr_gradient_init / (1 + c.lr_regularizer * prev ** c.lr_beta)
        else:
            lr_c = c.lr_consensus_init / (1 + c.lr_regularizer * t ** c.lr_alpha)
            lr_g = c.lr_gradient_init / (1 + c.lr_regularizer * t ** c.lr_beta)
            next_b = self._boundaries[-1] + math.ceil(c.lr_holding_ck / lr_c)
            self._boundaries.append(next_b)
        return lr_c, lr_g


# ---------------------------------------------------------------------------
#  Agent initialization
# ---------------------------------------------------------------------------

def initialize_agent(
    tensor: torch.Tensor,
    ranks: tuple[int, ...],
    rng: np.random.RandomState,
    device: torch.device,
) -> list[torch.Tensor]:
    """Create random Tucker parameters [core, factor_0, factor_1, ...].

    Parameters
    ----------
    tensor : torch.Tensor
        Original tensor (used only for shape information).
    ranks : tuple[int, ...]
        Target Tucker rank for each mode.
    rng : np.random.RandomState
        Random state for reproducibility.
    device : torch.device
        Target device.

    Returns
    -------
    list[torch.Tensor]
        ``[core, factor_0, ..., factor_{N-1}]``, all with ``requires_grad=True``.
    """
    core = tl.tensor(
        rng.random_sample(ranks), device=device, requires_grad=True
    )
    factors = [
        tl.tensor(
            rng.random_sample((tensor.shape[i], ranks[i])),
            device=device,
            requires_grad=True,
        )
        for i in range(tl.ndim(tensor))
    ]
    return [core] + factors


# ---------------------------------------------------------------------------
#  One QGD iteration for Tucker decomposition
# ---------------------------------------------------------------------------

def qgd_tucker_step(
    agents_params: list[list[torch.Tensor]],
    mixing_matrix: np.ndarray,
    lr_consensus: float,
    lr_gradient: float,
    target_tensor: torch.Tensor,
    iteration: int,
    levels: int,
    device: torch.device,
    penalty: float = 0.1,
) -> tuple[list[list[torch.Tensor]], list[float]]:
    """Perform one QGD update across all agents.

    Steps per agent:
      1. Quantize own parameters (even/odd switching).
      2. Consensus: weighted sum of quantized neighbor parameters.
      3. Gradient: compute loss, backprop, update.

    Parameters
    ----------
    agents_params : list[list[Tensor]]
        Each element is one agent's ``[core, factor_0, ...]``.
    mixing_matrix : np.ndarray
        Doubly-stochastic mixing matrix (n_agents × n_agents).
    lr_consensus, lr_gradient : float
        Current learning rates from the stepsize scheduler.
    target_tensor : torch.Tensor
        Ground-truth tensor to decompose.
    iteration : int
        Current iteration index (controls even/odd quantization).
    levels : int
        Number of quantization levels.
    device : torch.device
        Compute device.
    penalty : float
        L2 regularization coefficient on factor matrices.

    Returns
    -------
    agents_params : list[list[Tensor]]
        Updated parameters (in-place).
    losses : list[float]
        Per-agent loss values.
    """
    n_agents = len(agents_params)

    # --- Quantize all agents' parameters ---
    quantize_fn = quantize_even if iteration % 2 == 0 else quantize_odd
    quantized = [quantize_fn(agents_params[i], levels, device) for i in range(n_agents)]

    losses = []

    for i in range(n_agents):
        params = agents_params[i]
        core, factors = params[0], params[1:]

        # --- Forward: reconstruct tensor and compute loss ---
        reconstructed = tucker_to_tensor((core, factors))
        loss = tl.norm(reconstructed - target_tensor, 2)
        for f in factors:
            loss = loss + penalty * torch.sum(f.pow(2))

        loss.backward()
        losses.append(loss.item())

        # --- Update each parameter tensor ---
        with torch.no_grad():
            for j, p in enumerate(params):
                # Consensus term: weighted sum of quantized neighbors
                consensus = torch.zeros_like(p.data)
                for k in range(n_agents):
                    consensus += mixing_matrix[i, k] * quantized[k][j]

                # QGD update: x = (1 - lr_c) * x + lr_c * consensus - lr_g * grad
                p.data = (1 - lr_consensus) * p.data \
                         + lr_consensus * consensus \
                         - lr_gradient * p.grad.data

                p.grad.zero_()

    return agents_params, losses


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def run_experiment(config: TuckerConfig) -> dict:
    """Run decentralized Tucker decomposition with QGD.

    Returns
    -------
    dict
        Keys: ``loss_history``, ``rec_error_history``, ``config``.
    """
    rng = check_random_state(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    raw = np.load(config.data_path)
    target = tl.tensor(raw, device=device, requires_grad=False)
    target_norm = tl.norm(target, 2).item()
    print(f"Loaded tensor: shape={list(raw.shape)}, norm={target_norm:.4f}")

    # --- Initialize agents ---
    agents_params = [
        initialize_agent(target, config.ranks, rng, device)
        for _ in range(config.n_agents)
    ]

    scheduler = StepsizeScheduler(config)

    # Storage
    loss_history: list[list[float]] = []
    rec_error_history: list[float] = []

    print(f"\nStarting QGD Tucker decomposition")
    print(f"  agents={config.n_agents}  ranks={config.ranks}  "
          f"levels={config.quantization_levels}  iterations={config.n_iterations}")
    t_start = time.time()

    for t in range(config.n_iterations):
        lr_c, lr_g = scheduler(t)

        agents_params, losses = qgd_tucker_step(
            agents_params=agents_params,
            mixing_matrix=config.mixing_matrix,
            lr_consensus=lr_c,
            lr_gradient=lr_g,
            target_tensor=target,
            iteration=t,
            levels=config.quantization_levels,
            device=device,
            penalty=config.penalty,
        )

        if t % config.log_every == 0:
            rec_error = np.mean(losses) / target_norm
            loss_history.append(losses)
            rec_error_history.append(rec_error)
            print(f"  [iter {t:>7d}]  avg_loss={np.mean(losses):.6f}  "
                  f"rec_error={rec_error:.6f}  lr_c={lr_c:.6f}  lr_g={lr_g:.6f}")

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Final reconstruction error: {rec_error_history[-1]:.6f}")

    return {
        "loss_history": loss_history,
        "rec_error_history": rec_error_history,
        "config": config,
    }


# ---------------------------------------------------------------------------
#  Results I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str) -> Path:
    """Write per-checkpoint losses and reconstruction errors to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = results["config"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = out / (
        f"tucker_qgd_r{'x'.join(map(str, cfg.ranks))}"
        f"_l{cfg.quantization_levels}"
        f"_{timestamp}.csv"
    )

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["checkpoint", "rec_error"] + [f"loss_agent{i}" for i in range(cfg.n_agents)]
        writer.writerow(header)
        for idx, (losses, rec_err) in enumerate(
            zip(results["loss_history"], results["rec_error_history"])
        ):
            iteration = idx * cfg.log_every
            writer.writerow([iteration, rec_err] + losses)

    print(f"Results saved to {fname}")
    return fname


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> TuckerConfig:
    """Parse command-line arguments into a :class:`TuckerConfig`."""
    p = argparse.ArgumentParser(
        description="Decentralized Tucker decomposition with QGD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=str, default="data/observed.npy",
                   help="Path to the input tensor (.npy)")
    p.add_argument("--ranks", type=int, nargs="+", default=[5, 5, 5],
                   help="Tucker rank for each mode")
    p.add_argument("--n-agents", type=int, default=5,
                   help="Number of distributed agents")
    p.add_argument("--iterations", type=int, default=100_001,
                   help="Total optimization iterations")
    p.add_argument("--quantization-levels", type=int, default=64,
                   help="QSGD quantization levels")
    p.add_argument("--penalty", type=float, default=0.1,
                   help="L2 regularization on factor matrices")
    p.add_argument("--lr-consensus", type=float, default=0.03,
                   help="Initial consensus learning rate")
    p.add_argument("--lr-gradient", type=float, default=0.003,
                   help="Initial gradient learning rate")
    p.add_argument("--t0", type=int, default=100,
                   help="First holding-stage boundary")
    p.add_argument("--seed", type=int, default=1234,
                   help="Random seed")
    p.add_argument("--log-every", type=int, default=1000,
                   help="Logging interval (iterations)")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Output directory")

    args = p.parse_args()

    return TuckerConfig(
        data_path=args.data,
        ranks=tuple(args.ranks),
        n_agents=args.n_agents,
        n_iterations=args.iterations,
        quantization_levels=args.quantization_levels,
        penalty=args.penalty,
        lr_consensus_init=args.lr_consensus,
        lr_gradient_init=args.lr_gradient,
        t0=args.t0,
        seed=args.seed,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    config = parse_args()
    results = run_experiment(config)
    save_results(results, config.output_dir)
