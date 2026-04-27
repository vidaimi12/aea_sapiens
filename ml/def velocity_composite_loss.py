def velocity_composite_loss(pred, target, epsilon=0.05, alpha=0.7):
    """Per-finger CCC (weight alpha) + motion-weighted L1 (weight 1−alpha).

    CCC penalises flat predictions, phase shifts, and mean bias per finger.
    Weighted L1 down-weights static periods (target≈0) and is robust to
    staircase-artifact spikes (L1 vs L2 on heavy tails).
    """
    ccc_term = per_finger_ccc_loss(pred, target)
    weights  = (torch.abs(target) + epsilon)
    weights  = weights / weights.mean()
    l1_term  = (weights * torch.abs(pred - target)).mean()
    return alpha * ccc_term + (1 - alpha) * l1_term

