
import torch.nn.functional as F

def specular_loss(batch, out, recon_loss, weights=None, eps=1e-7):
    """
    batch: dict with 'rgb' and 'f_spec'
    out: dict with 'specular','diffuse','recon'
    recon_loss: your SSIMLoss instance
    """
    if weights is None:
        weights = dict(ssim=1.0, bce=1.0, dice=0.5, off=0.5, on=0.5, achro=0.1)

    S = out["specular"]
    R = batch["rgb"].cuda()
    F_gt = batch["f_spec"].cuda().clamp(0, 1)

    # spec magnitude map
    m_pred = S.mean(dim=1, keepdim=True)  # (B,1,H,W)

    # mask losses
    L_bce = F.binary_cross_entropy(m_pred, F_gt)

    num = 2.0 * (m_pred * F_gt).sum(dim=(1, 2, 3)) + eps
    den = m_pred.sum(dim=(1, 2, 3)) + F_gt.sum(dim=(1, 2, 3)) + eps
    L_dice = 1.0 - (num / den).mean()

    L_off = ((1.0 - F_gt) * m_pred).mean()
    L_on = (F_gt * (1.0 - m_pred)).mean()

    # reconstruction with SSIM
    L_ssim = recon_loss(out["recon"], R)

    # optional: achromaticity of specular
    S_mean = S.mean(dim=1, keepdim=True)
    L_achro = ((S - S_mean) ** 2).mean()

    total = (
        weights["ssim"] * L_ssim
        + weights["bce"] * L_bce
        + weights["dice"] * L_dice
        + weights["off"] * L_off
        + weights["on"] * L_on
        + weights["achro"] * L_achro
    )

    return {
        "total": total,
        "SSIM": L_ssim,
        "BCE": L_bce,
        "Dice": L_dice,
        "OffMask": L_off,
        "OnMask": L_on,
        "Achro": L_achro,
    }
