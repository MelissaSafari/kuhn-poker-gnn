# GNN-based Neural Scoring for Epistemic Models in Kuhn Poker

This repository contains a minimal prototype of a **Graph Neural Network (GNN)** that assigns a **scalar heuristic score** to an epistemic state (Kripke model) in **Kuhn Poker**.

The goal is **not** to build a full machine learning system, but to demonstrate **how a DEL-style epistemic model can be turned into a graph**, processed by a GNN, and used as a **scoring function** that a planner could call to guide search.

The code is part of a project on **epistemic planning in poker** and is meant as a **proof-of-concept** for a neural scoring heuristic.

---

## 1. Conceptual Overview

In our setting:

- An **epistemic state** is a **Kripke model** of the game:

  - Nodes = possible worlds (who holds which cards).
  - Edges = indistinguishability relations for each player (Alice = `a`, Bob = `b`).

- The GNN-based scorer:

  1. Encodes each world as a **node embedding** (a feature vector).
  2. Uses the indistinguishability relations as **graph edges**.
  3. Performs **message passing** to propagate information between indistinguishable worlds.
  4. **Pools** all node embeddings into a single **graph-level embedding**.
  5. Feeds this embedding to a small **MLP (feed-forward network)** to output **one scalar score**.

- A planner can then use this score as a **heuristic**:
  - One score **per epistemic model / branch**.
  - Used to **order** successors or **prune** unpromising branches.

> Important: The current implementation uses **random initial weights** and is **not trained**. It is meant to demonstrate the _architecture_ and _integration_ with an epistemic planner, not to provide a high-quality value function.

---

## 2. Repository Contents

- `gnn_kuhn_scorer.py`  
  Minimal PyTorch implementation of the GNN-based scoring function for the **3-card Kuhn Poker** epistemic model.

- (Optional) `graphs_kuhn.ipynb`  
  Jupyter notebook that visualises:
  - the **3-card** epistemic graph (6 worlds),
  - the **5-card** epistemic graph (20 worlds),
    using NetworkX and Matplotlib.


---

## 3. Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) (CPU version is fine)
- For graph visualisation in the notebook (optional):
  - `networkx`
  - `matplotlib`

Install them in a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# .\venv\Scripts\activate       # Windows PowerShell

pip install torch networkx matplotlib
```

## 4. Running the Code

After installing the dependencies and activating your virtual environment, you can run the neural scorer directly:

```bash
python gnn_kuhn_scorer.py


