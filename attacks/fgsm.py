import torch
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class FGSMConfig:
    eps: float  # in pixel space [0,1]


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,          # pixel-space tensor in [0,1]
    y: torch.Tensor,
    normalize_fn,
    cfg: FGSMConfig,
) -> torch.Tensor:
    """
    FGSM in pixel space:
      - x is clamped to [0,1]
      - perturbation is bounded in L_inf by eps in pixel space
      - model input is normalize_fn(x_adv)

    Returns x_adv in pixel space [0,1].
    """
    model.eval()
    x = x.detach().clamp(0.0, 1.0)

    x_adv = x.clone().detach().requires_grad_(True)

    # Disable autocast for stable gradients
    logits = model(normalize_fn(x_adv))
    loss = F.cross_entropy(logits, y)
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + cfg.eps * grad_sign
    x_adv = x_adv.detach().clamp(0.0, 1.0)
    return x_adv