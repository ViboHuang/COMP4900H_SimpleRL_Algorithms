# Survey project - Lightweight_RL_algorithms.py (Gridworld RL)

## Description
This script runs a small tabular RL experiment on a **5×5 stochastic gridworld** and generates the two plots:

- **Figure 3**: SARSA vs Q-learning vs Dyna-Q (k=10), averaged over 6 random seeds  
- **Figure 4**: Dyna-Q planning budget sensitivity (k = 0, 5, 10), averaged over 6 seeds

The environment is intentionally simple so results run fast and are easy to inspect.

---

## Requirements

- Python **3.9+** (3.10/3.11 also fine)
- Packages:
  - `numpy`
  - `matplotlib`

### Installation

1. Clone the repository to a local folder.
2. Navigate to the project directory.

Install dependencies:
```bash
pip install numpy matplotlib

```

run pyuthon direct from directory:
```bash
python Lightweight_RL_algorithms.py

```

### Authors & contact

Author: YiLong Huang
Email: vibohuang@cmail.carleton.ca
Course: COMP4900H
SID: 101187050
Verison: 1.0