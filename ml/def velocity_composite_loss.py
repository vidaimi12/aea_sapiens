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

def per_finger_ccc_loss(pred, target):
    """1 − CCC computed independently per finger, then averaged.
    Avoids the global-CCC failure mode where all 5 outputs are merged into one distribution."""
    total = 0.0
    for i in range(pred.shape[1]):
        p, t           = pred[:, i], target[:, i]
        p_mean, t_mean = p.mean(), t.mean()
        p_std,  t_std  = p.std() + 1e-8, t.std() + 1e-8
        rho = ((p - p_mean) * (t - t_mean)).mean() / (p_std * t_std)
        ccc = (2 * rho * p_std * t_std) / (p.var() + t.var() + (p_mean - t_mean) ** 2 + 1e-8)
        total += 1 - ccc
    return total / pred.shape[1]