from __future__ import annotations

import torch


def compute_support_penalty(
    foot_height: torch.Tensor,
    ray_hits_z: torch.Tensor,
    height_offset: float,
    max_clearance: float = 0.25,
    unsupported_threshold: float = 0.03,
    variation_scale: float = 0.5,
    unsupported_scale: float = 0.5,
) -> torch.Tensor:
    """Estimate how unsupported a contacting foot is from local height rays.

    The penalty stays at zero when all rays see a flat, fully supported contact patch and grows when the
    contact patch spans an edge or a drop. It combines three signals:

    - mean positive clearance to nearby support
    - variation in clearance across the sole
    - fraction of rays that exceed a support-loss threshold
    """

    clearances = torch.clamp(foot_height.unsqueeze(-1) - ray_hits_z - height_offset, min=0.0, max=max_clearance)
    mean_gap = clearances.mean(dim=-1)
    support_range = clearances.max(dim=-1).values - clearances.min(dim=-1).values
    unsupported_ratio = (clearances > unsupported_threshold).float().mean(dim=-1)
    return mean_gap + variation_scale * support_range + unsupported_scale * unsupported_ratio
