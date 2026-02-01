import torch
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class PGDConfig:
    eps: float        # L_inf bound in pixel space [0,1]
    alpha: float      # step size in pixel space
    steps: int = 10
    random_start: bool = True


def pgd_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,          # pixel-space tensor in [0,1]
    y: torch.Tensor,
    normalize_fn,
    cfg: PGDConfig,
) -> torch.Tensor:
    """
    PGD (L_inf) in pixel space:
      - x is clamped to [0,1]
      - delta is projected to [-eps, eps] in pixel space
      - x_adv is clamped to [0,1]
      - model input is normalize_fn(x_adv)

    Returns x_adv in pixel space [0,1].
    """
    model.eval()
    x = x.detach().clamp(0.0, 1.0)

    if cfg.random_start:
        delta = torch.empty_like(x).uniform_(-cfg.eps, cfg.eps)
        x_adv = (x + delta).clamp(0.0, 1.0).detach()
    else:
        x_adv = x.clone().detach()

    for _ in range(cfg.steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        logits = model(normalize_fn(x_adv))
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv + cfg.alpha * grad.sign()

        # Project back to L_inf ball around x
        delta = torch.clamp(x_adv - x, min=-cfg.eps, max=cfg.eps)
        x_adv = (x + delta).detach().clamp(0.0, 1.0)

    return x_adv