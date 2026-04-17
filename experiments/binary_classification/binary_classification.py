"""Binary Classification with Quantized Gradient Descent (QGD).

This script demonstrates how the QGD optimizer escapes saddle points on a
simple 2-D logistic regression landscape.  A network of ``n_agents`` nodes
collaborates via a doubly-stochastic mixing matrix, communicating only
*quantized* iterates to save bandwidth while provably avoiding saddle points.

Usage
-----
    # Run with defaults (QGD, 8 000 iterations, 5 agents)
    python binary_classification.py

    # Compare QGD against vanilla decentralized GD
    python binary_classification.py --method qgd --iterations 10000
    python binary_classification.py --method dgd --iterations 10000

    # Customize quantization and learning-rate schedule
    python binary_classification.py --quantization-levels 20 \
                                     --lr-init 0.8 \
                                     --lr-alpha 0.7

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.linalg as la


# ---------------------------------------------------------------------------
#  Quantization primitives
# ---------------------------------------------------------------------------

def qsgd_quantize_even(
    x: np.ndarray,
    levels: int,
    biased: bool = False,
) -> np.ndarray:
    """Standard QSGD quantization (used at even iterations).

    Each coordinate is stochastically rounded to one of ``levels``
    uniformly-spaced magnitudes between 0 and ``‖x‖``.

    Parameters
    ----------
    x : np.ndarray
        Vector to quantize.
    levels : int
        Number of quantization levels (higher → finer granularity).
    biased : bool
        If True, apply a variance-reduction scaling factor.

    Returns
    -------
    np.ndarray
        Quantized vector with the same shape as *x*.
    """
    norm = max(np.linalg.norm(x), 1e-1)
    level_float = levels * np.abs(x) / norm
    lower = np.floor(level_float)
    prob = level_float - lower
    rounded = lower + (np.random.rand(*x.shape) < prob).astype(float)

    scale = 1.0
    if biased:
        n = len(x)
        scale = 1.0 / (min(n / levels**2, np.sqrt(n) / levels) + 1.0)

    return scale * np.sign(x) * norm * rounded / levels


def qsgd_quantize_odd(
    x: np.ndarray,
    levels: int,
    biased: bool = False,
) -> np.ndarray:
    """Shifted-level quantization (used at odd iterations).

    Uses *odd*-numbered quantization thresholds so that the quantization
    grid shifts between consecutive iterations, ensuring persistent
    perturbation amplitude — the key mechanism for saddle-point escape.

    Parameters
    ----------
    x, levels, biased
        Same as :func:`qsgd_quantize_even`.

    Returns
    -------
    np.ndarray
        Quantized vector.
    """
    norm = max(np.linalg.norm(x), 1e-1)
    scaled = 2 * levels * x / norm
    lower = np.floor(scaled)
    # Snap to nearest odd level
    lower_odd = np.where(lower % 2 == 0, lower - 1, lower)
    prob = (scaled - lower_odd) / 2.0
    rounded = lower_odd + 2 * (np.random.rand(*x.shape) < prob).astype(float)

    scale = 1.0
    if biased:
        n = len(x)
        scale = 1.0 / (min(n / levels**2, np.sqrt(n) / levels) + 1.0)

    return scale * norm * rounded / (2 * levels)


# ---------------------------------------------------------------------------
#  Objective: Regularized logistic loss on a 2-D saddle-point landscape
# ---------------------------------------------------------------------------

def logistic_loss(x: np.ndarray, l2: float = 0.1) -> np.ndarray:
    """Compute element-wise regularized logistic loss.

    f(x1, x2) = log(1 + exp(-x1 * x2)) + (l2 / 2) * ‖x‖²

    This landscape has a saddle point at the origin — the key test case.

    Parameters
    ----------
    x : np.ndarray, shape (2, n_agents)
        Each column is one agent's parameter vector.
    l2 : float
        L2 regularization coefficient.
    """
    log_term = np.log(1 + np.exp(-np.prod(x, axis=0)))
    reg_term = (l2 / 2) * la.norm(x, axis=0) ** 2
    return log_term + reg_term


def logistic_gradient(x: np.ndarray, l2: float = 0.1) -> list[float]:
    """Gradient of the regularized logistic loss w.r.t. a single agent.

    Parameters
    ----------
    x : np.ndarray, shape (2,)
        Parameter vector for one agent.
    l2 : float
        L2 regularization coefficient.

    Returns
    -------
    list[float]
        Two-element gradient [∂f/∂x1, ∂f/∂x2].
    """
    sigmoid_neg = 1.0 / (1 + np.exp(x[0] * x[1]))
    grad_x1 = -x[1] * sigmoid_neg + l2 * x[0]
    grad_x2 = -x[0] * sigmoid_neg + l2 * x[1]
    return [grad_x1, grad_x2]


# ---------------------------------------------------------------------------
#  Experiment configuration
# ---------------------------------------------------------------------------

TOPOLOGIES = {
    "connected": np.array([
        [0.6, 0.0, 0.0, 0.4, 0.0],
        [0.2, 0.8, 0.0, 0.0, 0.0],
        [0.2, 0.1, 0.4, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.6, 0.4],
        [0.0, 0.1, 0.6, 0.0, 0.3],
    ]),
    "ring": None,        # built dynamically based on n_agents
    "centralized": None,  # built dynamically based on n_agents
}


@dataclass
class ExperimentConfig:
    """All tuneable knobs for the binary classification experiment."""

    # Optimization
    method: Literal["qgd", "dgd"] = "qgd"
    n_iterations: int = 8_000
    n_agents: int = 5
    l2_reg: float = 0.1

    # Quantization
    quantization_levels: int = 10
    biased_quantization: bool = False

    # Learning-rate schedule  (piecewise-constant "holding" strategy)
    lr_init: float = 1.0
    lr_alpha: float = 0.62       # decay exponent for consensus step
    lr_beta: float = 0.94        # decay exponent for gradient step
    lr_regularizer: float = 1.0  # denominator growth rate
    lr_holding_ck: float = 3.0   # holding-stage duration parameter
    t0: int = 10                 # first holding-stage boundary

    # Network
    topology: str = "connected"

    # Reproducibility
    seed: int = 42
    init_std: float = 0.05

    # I/O
    log_every: int = 10
    output_dir: str = "results"


# ---------------------------------------------------------------------------
#  Learning-rate schedule with holding stages
# ---------------------------------------------------------------------------

class StepsizeScheduler:
    """Piecewise-constant learning-rate schedule.

    The schedule decays polynomially but *holds* the rates constant inside
    each "holding stage".  This ensures the quantization perturbation
    integrates long enough to escape saddle points.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment hyperparameters.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config
        first_boundary = config.t0 + math.ceil(
            config.lr_holding_ck
            / (config.lr_init / (1 + config.lr_regularizer * config.t0 ** config.lr_alpha))
        )
        self._boundaries: list[int] = [config.t0, first_boundary]

    def __call__(self, t: int) -> tuple[float, float]:
        """Return ``(lr_consensus, lr_gradient)`` for iteration *t*."""
        c = self.cfg

        if t <= self._boundaries[0]:
            lr_cons = c.lr_init / (1 + c.lr_regularizer * t ** c.lr_alpha)
            lr_grad = c.lr_init / (1 + c.lr_regularizer * t ** c.lr_beta)

        elif self._boundaries[-2] < t < self._boundaries[-1]:
            # Inside a holding stage — rates stay constant
            prev = self._boundaries[-2]
            lr_cons = c.lr_init / (1 + c.lr_regularizer * prev ** c.lr_alpha)
            lr_grad = c.lr_init / (1 + c.lr_regularizer * prev ** c.lr_beta)

        else:  # t >= last boundary → start a new stage
            lr_cons = c.lr_init / (1 + c.lr_regularizer * t ** c.lr_alpha)
            lr_grad = c.lr_init / (1 + c.lr_regularizer * t ** c.lr_beta)
            next_boundary = self._boundaries[-1] + math.ceil(
                c.lr_holding_ck / lr_cons
            )
            self._boundaries.append(next_boundary)

        return lr_cons, lr_grad


# ---------------------------------------------------------------------------
#  Core optimizer
# ---------------------------------------------------------------------------

def _build_mixing_matrix(topology: str, n: int) -> np.ndarray:
    """Construct a doubly-stochastic mixing matrix for *n* agents."""
    if topology == "connected" and n == 5:
        return TOPOLOGIES["connected"]
    if topology == "ring":
        W = np.zeros((n, n))
        val = 1.0 / 3 if n >= 3 else 0.5
        np.fill_diagonal(W, val)
        np.fill_diagonal(W[1:], val)
        np.fill_diagonal(W[:, 1:], val)
        W[0, n - 1] = val
        W[n - 1, 0] = val
        return W
    if topology == "centralized":
        return np.ones((n, n)) / n
    raise ValueError(f"Unknown topology: {topology}")


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the binary classification experiment.

    Returns
    -------
    dict
        Keys: ``losses`` (per-iteration avg loss), ``trajectories``
        (agent parameter snapshots), ``lr_consensus``, ``lr_gradient``.
    """
    np.random.seed(config.seed)
    W = _build_mixing_matrix(config.topology, config.n_agents)
    scheduler = StepsizeScheduler(config)

    # Initialize: all agents start from the same small random point
    x0 = np.random.normal(0, config.init_std, size=(2,))
    x = np.tile(x0, (config.n_agents, 1)).T  # shape (2, n_agents)

    # Storage
    losses = np.zeros(config.n_iterations)
    trajectories = np.zeros((config.n_iterations, 2, config.n_agents))
    lr_cons_log, lr_grad_log = [], []

    print(f"Starting {config.method.upper()} | {config.n_iterations} iters | "
          f"{config.n_agents} agents | topology={config.topology}")
    print(f"  Initial x = {x0}")
    t_start = time.time()

    for t in range(config.n_iterations):
        # Record
        loss = logistic_loss(x, l2=config.l2_reg)
        losses[t] = np.mean(loss)
        trajectories[t] = x

        # Learning rates
        lr_cons, lr_grad = scheduler(t)
        lr_cons_log.append(lr_cons)
        lr_grad_log.append(lr_grad)

        # --- Local gradient step ---
        x_plus = np.zeros_like(x)
        for agent in range(config.n_agents):
            grad = logistic_gradient(x[:, agent], l2=config.l2_reg)
            x_plus[:, agent] = (1 - lr_cons) * x[:, agent] - lr_grad * np.array(grad)

        # --- Communication step (quantize → mix) ---
        if config.method == "qgd":
            quantize_fn = qsgd_quantize_even if t % 2 == 0 else qsgd_quantize_odd
            x_q = np.zeros_like(x)
            for agent in range(config.n_agents):
                x_q[:, agent] = quantize_fn(
                    x[:, agent], config.quantization_levels, config.biased_quantization
                )
            x = x_plus + lr_cons * x_q @ W

        elif config.method == "dgd":
            # Vanilla decentralized GD (no quantization)
            x = x_plus + lr_cons * x @ W

        else:
            raise ValueError(f"Unknown method: {config.method}")

        # --- Logging ---
        if t % config.log_every == 0:
            print(f"  [iter {t:>6d}] loss={losses[t]:.6f}  "
                  f"x_agent0=({x[0, 0]:+.4f}, {x[1, 0]:+.4f})  "
                  f"lr_cons={lr_cons:.5f}  lr_grad={lr_grad:.5f}")

            if np.isnan(losses[t]) or np.isinf(losses[t]):
                print("  ⚠ Divergence detected — stopping early.")
                losses = losses[:t + 1]
                trajectories = trajectories[:t + 1]
                break

    elapsed = time.time() - t_start
    print(f"Finished in {elapsed:.1f}s  |  final loss = {losses[-1]:.6f}")

    return {
        "losses": losses,
        "trajectories": trajectories,
        "lr_consensus": np.array(lr_cons_log),
        "lr_gradient": np.array(lr_grad_log),
        "config": config,
    }


# ---------------------------------------------------------------------------
#  Results I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str) -> Path:
    """Save experiment results to CSV.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = results["config"]
    fname = out / f"binary_{cfg.method}_n{cfg.n_agents}_iter{cfg.n_iterations}.csv"

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "loss",
                         *(f"x{d}_agent{a}" for a in range(cfg.n_agents) for d in range(2)),
                         "lr_consensus", "lr_gradient"])
        for t in range(len(results["losses"])):
            row = [t, results["losses"][t]]
            for a in range(cfg.n_agents):
                row.extend(results["trajectories"][t, :, a].tolist())
            row.append(results["lr_consensus"][t])
            row.append(results["lr_gradient"][t])
            writer.writerow(row)

    print(f"Results saved to {fname}")
    return fname


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    """Parse command-line arguments into an :class:`ExperimentConfig`."""
    p = argparse.ArgumentParser(
        description="Binary classification with QGD — saddle-point escape demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", choices=["qgd", "dgd"], default="qgd",
                   help="Optimization method")
    p.add_argument("--iterations", type=int, default=8_000,
                   help="Number of optimization iterations")
    p.add_argument("--n-agents", type=int, default=5,
                   help="Number of distributed agents")
    p.add_argument("--quantization-levels", type=int, default=10,
                   help="QSGD quantization levels (higher = more bits)")
    p.add_argument("--topology", choices=["connected", "ring", "centralized"],
                   default="connected", help="Network topology")
    p.add_argument("--lr-init", type=float, default=1.0,
                   help="Initial learning rate")
    p.add_argument("--lr-alpha", type=float, default=0.62,
                   help="Consensus LR decay exponent")
    p.add_argument("--lr-beta", type=float, default=0.94,
                   help="Gradient LR decay exponent")
    p.add_argument("--t0", type=int, default=10,
                   help="First holding-stage boundary")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--log-every", type=int, default=10,
                   help="Print interval (iterations)")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for CSV output")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving results to disk")

    args = p.parse_args()

    return ExperimentConfig(
        method=args.method,
        n_iterations=args.iterations,
        n_agents=args.n_agents,
        quantization_levels=args.quantization_levels,
        topology=args.topology,
        lr_init=args.lr_init,
        lr_alpha=args.lr_alpha,
        lr_beta=args.lr_beta,
        t0=args.t0,
        seed=args.seed,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    config = parse_args()
    results = run_experiment(config)

    if not config.output_dir == "none":
        save_results(results, config.output_dir)
