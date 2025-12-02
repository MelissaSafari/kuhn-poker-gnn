import torch
import torch.nn as nn
import torch.nn.functional as F


# Define worlds (3-card Kuhn game)


# index: 0         1         2         3         4         5
worlds = [("J", "Q"),
          ("J", "K"),
          ("Q", "J"),
          ("Q", "K"),
          ("K", "J"),
          ("K", "Q")]

card_to_idx = {"J": 0, "Q": 1, "K": 2}
num_cards = len(card_to_idx)
num_worlds = len(worlds)

# Node features: [one-hot P1 card | one-hot P2 card]
X = torch.zeros(num_worlds, 2 * num_cards)

for i, (c1, c2) in enumerate(worlds):
    # P1 card one-hot
    X[i, card_to_idx[c1]] = 1.0
    # P2 card one-hot
    X[i, num_cards + card_to_idx[c2]] = 1.0

print("Node feature matrix X:")
print(X)
print("Shape:", X.shape)



# 2. Build adjacency for indistinguishability (a and b)


A = torch.zeros(num_worlds, num_worlds)

# P1 indistinguishability: connect worlds with same a card
for i, (c1_i, _) in enumerate(worlds):
    for j, (c1_j, _) in enumerate(worlds):
        if c1_i == c1_j:
            A[i, j] = 1.0

# P2 indistinguishability: connect worlds with same b card
for i, (_, c2_i) in enumerate(worlds):
    for j, (_, c2_j) in enumerate(worlds):
        if c2_i == c2_j:
            A[i, j] = 1.0

# self-loops (each node connected to itself)
A = A + torch.eye(num_worlds)

# Row-normalise so each row sums to 1 (average of neighbours)
D_inv = torch.diag(1.0 / A.sum(dim=1))
A_norm = D_inv @ A

print("\nNormalised adjacency A_norm:")
print(A_norm)


#  Define tiny GNN + scoring head


class EpistemicGNNScorer(nn.Module):
    """
    Minimal GNN-based scorer for an epistemic model.

    - X: node features (world-level info)
    - A_norm: normalised adjacency (indistinguishability edges)

    Output: single scalar score.
    """
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        # "Graph convolution" weight matrix
        self.W1 = nn.Linear(in_dim, hidden_dim)

        # MLP scoring head: graph embedding -> scalar
        self.fc_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, X, A_norm):
        # Message passing: aggregate neighbours
        # h has same shape as X: (num_worlds, in_dim)
        h = A_norm @ X
        h = self.W1(h)
        h = F.relu(h)

        # Global mean pooling: average over all worlds
        g = h.mean(dim=0)  # shape: (hidden_dim,)

        # MLP scoring head
        score = self.fc_score(g)  # shape: (1,)
        return score.squeeze(-1)  # scalar tensor

#  Instantiate model and compute one score
model = EpistemicGNNScorer(in_dim=X.shape[1], hidden_dim=16)

with torch.no_grad():
    score = model(X, A_norm)

print("\nHeuristic score for this epistemic model:", float(score))
