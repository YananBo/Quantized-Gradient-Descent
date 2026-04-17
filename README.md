# QGD: Quantized Gradient Descent for Distributed Nonconvex Optimization

[![Paper](https://img.shields.io/badge/PNAS-2024-blue)](https://doi.org/10.1073/pnas.2319625121)
[![arXiv](https://img.shields.io/badge/arXiv-2403.10423-b31b1b)](https://arxiv.org/abs/2403.10423)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

A PyTorch implementation of **Quantized Gradient Descent (QGD)** — a communication-efficient distributed optimizer that exploits stochastic quantization to escape saddle points in nonconvex optimization. Published in *Proceedings of the National Academy of Sciences* (PNAS), 2024.

---

## Highlights

- **Saddle-point avoidance via quantization**: Unlike prior work that treats quantization noise as harmful, QGD exploits it as a built-in perturbation mechanism to escape saddle points — no additional noise injection needed.
- **Communication efficiency**: Adjustable quantization granularity allows aggressive reduction of bits per iteration, addressing the communication bottleneck in distributed training.
- **Convergence guarantees**: Provably converges to second-order stationary points in distributed nonconvex optimization.
- **Drop-in PyTorch optimizer**: Implemented as a standard `torch.optim.Optimizer` subclass for easy integration.

---

## Repository Structure

```
├── experiments/
│   ├── binary_classification/   # 2-D logistic regression (saddle-point landscape)
│   ├── neural_network/          # Distributed CNN training on CIFAR-10 / MNIST
│   ├── robust_pca/              # Robust PCA on video surveillance data
│   └── tensor_decomposition/    # Distributed Tucker decomposition
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

Each experiment has its **own README** with setup instructions, CLI usage, and hyperparameter tables. See the links below.

---

## Experiments

| Experiment | Task | Key Demonstration | README |
|-----------|------|-------------------|--------|
| [Binary Classification](experiments/binary_classification/) | 2-D logistic regression | Saddle-point escape visualization | [→ README](experiments/binary_classification/README.md) |
| [Neural Network](experiments/neural_network/) | CIFAR-10 / MNIST classification | QGD vs. 7 baseline optimizers | [→ README](experiments/neural_network/README.md) |
| [Robust PCA](experiments/robust_pca/) | Low-rank + sparse decomposition | QGD vs. vanilla DGD on video data | [→ README](experiments/robust_pca/README.md) |
| [Tensor Decomposition](experiments/tensor_decomposition/) | Tucker factorization | Distributed nonconvex factorization | [→ README](experiments/tensor_decomposition/README.md) |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YananBo/qgd-distributed-optimizer.git
cd qgd-distributed-optimizer

# Install dependencies
pip install -r requirements.txt

# Run a quick demo (binary classification, ~30 seconds)
cd experiments/binary_classification
python binary_classification.py --method qgd --iterations 5000
```

---

## Requirements

```
torch>=1.10
torchvision>=0.11
numpy>=1.21
tensorly>=0.7
scikit-learn>=1.0
pandas>=1.3
Pillow>=8.0
```

---

## Citation

```bibtex
@article{bo2024quantization,
  title     = {Quantization Avoids Saddle Points in Distributed Optimization},
  author    = {Bo, Yanan and Wang, Yongqiang},
  journal   = {Proceedings of the National Academy of Sciences},
  volume    = {121},
  number    = {17},
  pages     = {e2319625121},
  year      = {2024},
  publisher = {National Academy of Sciences},
  doi       = {10.1073/pnas.2319625121}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
