import torch
import torch.nn as nn

class SoftBinningCalibrationLoss(nn.Module):

```
def __init__(
    self,
    num_bins=15,
    sigma=0.05
):
    super().__init__()

    self.num_bins = num_bins
    self.sigma = sigma

    centers = torch.linspace(
        0.0,
        1.0,
        num_bins
    )

    self.register_buffer(
        "bin_centers",
        centers
    )

def forward(
    self,
    logits,
    targets
):

    probs = torch.sigmoid(logits)

    probs = probs.reshape(-1)

    targets = targets.reshape(-1).float()

    ece = 0.0

    for center in self.bin_centers:

        weights = torch.exp(
            -((probs - center) ** 2)
            /
            (2 * self.sigma ** 2)
        )

        weights = weights + 1e-8

        conf = (
            weights * probs
        ).sum() / weights.sum()

        acc = (
            weights * targets
        ).sum() / weights.sum()

        ece += (
            weights.mean()
            *
            torch.abs(
                acc - conf
            )
        )

    return ece
```

# --------------------------------------------------

# ECE

# --------------------------------------------------

def calculate_ece(
probs,
targets,
num_bins=15
):

```
probs = probs.reshape(-1)

targets = targets.reshape(-1)

bins = torch.linspace(
    0,
    1,
    num_bins + 1,
    device=probs.device
)

ece = 0

for i in range(num_bins):

    mask = (
        (probs >= bins[i])
        &
        (probs < bins[i + 1])
    )

    if mask.sum() > 0:

        conf = probs[mask].mean()

        acc = targets[mask].float().mean()

        ece += (
            mask.float().mean()
            *
            torch.abs(
                acc - conf
            )
        )

return ece.item()
```

# --------------------------------------------------

# ACE

# --------------------------------------------------

def calculate_ace(
probs,
targets,
num_bins=15
):

```
probs = probs.reshape(-1)

targets = targets.reshape(-1)

sorted_probs, indices = torch.sort(
    probs
)

sorted_targets = targets[
    indices
]

bin_size = (
    len(sorted_probs)
    //
    num_bins
)

ace = 0

for i in range(num_bins):

    start = i * bin_size

    end = (
        start + bin_size
        if i < num_bins - 1
        else len(sorted_probs)
    )

    p = sorted_probs[start:end]

    t = sorted_targets[start:end]

    conf = p.mean()

    acc = t.float().mean()

    ace += torch.abs(
        acc - conf
    )

ace /= num_bins

return ace.item()
```

# --------------------------------------------------

# BRIER SCORE

# --------------------------------------------------

def calculate_brier_score(
probs,
targets
):

```
probs = probs.reshape(-1)

targets = targets.reshape(-1).float()

brier = (
    (probs - targets) ** 2
).mean()

return brier.item()
```
