"""Losses + metrics for the physics-informed forecaster.

- Differentiable SSIM (Gaussian window) used both as a metric and as a loss
  term (directly optimising the target metric).
- Physics-informed regularisers: temporal smoothness, total variation,
  divergence and advection consistency (atmospheric-flow priors).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Differentiable SSIM / PSNR                                                   #
# --------------------------------------------------------------------------- #
def _gaussian_window(ws, sigma, channels, device, dtype):
    coords = torch.arange(ws, device=device, dtype=dtype) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :]).expand(channels, 1, ws, ws).contiguous()
    return win


class SSIM(nn.Module):
    """SSIM over [N, C, H, W] tensors, data_range=1.0, channel-wise then mean."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.ws, self.sigma = window_size, sigma
        self.C1, self.C2 = 0.01 ** 2, 0.03 ** 2

    def forward(self, x, y):
        ch = x.shape[1]
        win = _gaussian_window(self.ws, self.sigma, ch, x.device, x.dtype)
        pad = self.ws // 2
        mu_x = F.conv2d(x, win, padding=pad, groups=ch)
        mu_y = F.conv2d(y, win, padding=pad, groups=ch)
        mx2, my2, mxy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sx = F.conv2d(x * x, win, padding=pad, groups=ch) - mx2
        sy = F.conv2d(y * y, win, padding=pad, groups=ch) - my2
        sxy = F.conv2d(x * y, win, padding=pad, groups=ch) - mxy
        s = ((2 * mxy + self.C1) * (2 * sxy + self.C2)) / \
            ((mx2 + my2 + self.C1) * (sx + sy + self.C2))
        return s.mean()


class MSSSIM(nn.Module):
    """Multi-scale SSIM: SSIM averaged over `scales` successive 2x downsamplings.
    Correlates better with the SSIM metric than single-scale -> stronger objective."""
    def __init__(self, scales=3):
        super().__init__()
        self.ssim = SSIM()
        self.scales = scales

    def forward(self, x, y):
        vals = []
        for i in range(self.scales):
            vals.append(self.ssim(x, y))
            if i < self.scales - 1 and min(x.shape[-2:]) >= 22:
                x = F.avg_pool2d(x, 2); y = F.avg_pool2d(y, 2)
        return sum(vals) / len(vals)


def ssim_metric(pred, target, ssim_mod):
    """pred/target: [B, T, C, H, W] -> scalar mean SSIM."""
    B, T, C, H, W = pred.shape
    return ssim_mod(pred.reshape(B * T, C, H, W).clamp(0, 1),
                    target.reshape(B * T, C, H, W).clamp(0, 1))


def psnr_metric(pred, target):
    """pred/target: [B, T, C, H, W] -> mean PSNR (dB) over frames/bands."""
    p = pred.clamp(0, 1)
    t = target.clamp(0, 1)
    mse = F.mse_loss(p, t, reduction='none').mean(dim=[-1, -2])   # [B,T,C]
    mse = torch.clamp(mse, min=1e-10)
    return (10.0 * torch.log10(1.0 / mse)).mean()


# --------------------------------------------------------------------------- #
#  Physics-informed regularisers (atmospheric flow priors)                     #
# --------------------------------------------------------------------------- #
def total_variation(x):                         # spatial smoothness
    return (x[..., :, :-1] - x[..., :, 1:]).abs().mean() + \
           (x[..., :-1, :] - x[..., 1:, :]).abs().mean()


def temporal_smoothness(x):                     # 2nd-order temporal (acceleration)
    if x.shape[1] < 3:
        return x.new_tensor(0.0)
    return ((x[:, 2] - 2 * x[:, 1] + x[:, 0]) ** 2).mean()


def _ddx(z):                                    # central difference along W (same size, any ND)
    sh = z.shape
    z4 = z.reshape(-1, 1, sh[-2], sh[-1])
    zp = F.pad(z4, (1, 1, 0, 0), mode="replicate")
    return (0.5 * (zp[..., 2:] - zp[..., :-2])).reshape(sh)


def _ddy(z):                                    # central difference along H (same size, any ND)
    sh = z.shape
    z4 = z.reshape(-1, 1, sh[-2], sh[-1])
    zp = F.pad(z4, (0, 0, 1, 1), mode="replicate")
    return (0.5 * (zp[..., 2:, :] - zp[..., :-2, :])).reshape(sh)


def velocity_divergence(u, v):
    """Divergence of the VELOCITY field, div(u) = du/dx + dv/dy (incompressibility
    diagnostic on the flow, not on image intensity). u, v: [...,H,W]."""
    return (_ddx(u) + _ddy(v)).abs().mean()


def advection_consistency(pred, target):
    """Match the predicted optical-flow-like transport to the target's."""
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    vp = pred[:, 1:] - pred[:, :-1]
    vt = target[:, 1:] - target[:, :-1]
    return F.l1_loss(vp, vt)


def dense_lk_flow(prev, curr, ws=9, sigma=2.0):
    """Differentiable dense Lucas-Kanade optical flow (u,v) per pixel from two
    grayscale frames prev,curr [B,1,H,W] -> u,v [B,1,H,W] in pixels/frame."""
    Ix = _ddx(curr); Iy = _ddy(curr); It = curr - prev
    win = _gaussian_window(ws, sigma, 1, prev.device, prev.dtype)
    blur = lambda z: F.conv2d(z, win, padding=ws // 2)
    A11 = blur(Ix * Ix); A12 = blur(Ix * Iy); A22 = blur(Iy * Iy)
    b1 = -blur(Ix * It); b2 = -blur(Iy * It)
    det = A11 * A22 - A12 * A12 + 1e-6
    u = (A22 * b1 - A12 * b2) / det
    v = (A11 * b2 - A12 * b1) / det
    return u, v                                  # [B,1,H,W]


def continuity_pde_residual(pred, x, return_div=False):
    """Mass-continuity PDE residual  dI/dt + div(I u) = dI/dt + d(I u)/dx + d(I v)/dy,
    where the 2-D velocity field u(x,y) is estimated by dense Lucas-Kanade from the
    last two observed frames. This is the correct image-plane formulation: it uses
    the velocity field and its divergence (div(Iu) = u.grad I + I div u), and adds
    NO out-of-plane (Stokes settling) term to the in-plane motion.
    pred:[B,T,C,H,W]  x(inputs):[B,Tin,C,H,W]."""
    prev = x[:, -2].mean(1, keepdim=True); curr = x[:, -1].mean(1, keepdim=True)
    u, v = dense_lk_flow(prev, curr)             # [B,1,H,W]
    u = u.unsqueeze(1); v = v.unsqueeze(1)        # [B,1,1,H,W] (broadcast over T,C)
    It = pred[:, 1:] - pred[:, :-1]               # [B,T-1,C,H,W]
    divIu = _ddx(pred * u) + _ddy(pred * v)       # [B,T,C,H,W]
    res = (It + divIu[:, 1:]).abs().mean()        # forward-Euler continuity
    if return_div:
        return res, velocity_divergence(u, v)
    return res


def gradient_loss(pred, target):                # edge / high-freq fidelity
    pdx = (pred[..., :, :-1] - pred[..., :, 1:]).abs()
    pdy = (pred[..., :-1, :] - pred[..., 1:, :]).abs()
    tdx = (target[..., :, :-1] - target[..., :, 1:]).abs()
    tdy = (target[..., :-1, :] - target[..., 1:, :]).abs()
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


class CombinedLoss(nn.Module):
    """Weighted multi-term objective. SSIM term is weighted strongly because the
    headline target is SSIM; physics terms are light regularisers."""
    def __init__(self, w_l1=1.0, w_ssim=1.0, w_mse=0.6, w_msssim=0.0, w_grad=0.15,
                 w_temporal=0.15, w_tv=0.02, w_div=0.02, w_adv=0.1, w_pde=0.0,
                 v_stokes=0.02, lead_weights=None):
        super().__init__()
        self.ssim = SSIM()
        self.msssim = MSSSIM() if w_msssim > 0 else None
        self.lead_weights = lead_weights      # list of per-lead-time weights or None
        # w_div / v_stokes kept for API compatibility but unused: the corrected
        # physics folds the velocity divergence into the continuity residual (PDE)
        self.w = dict(l1=w_l1, ssim=w_ssim, mse=w_mse, msssim=w_msssim, grad=w_grad,
                      temporal=w_temporal, tv=w_tv, adv=w_adv, pde=w_pde)

    def forward(self, pred, target, x=None):
        B, T, C, H, W = pred.shape
        # per-lead-time weights (normalised to mean 1 so overall scale is stable)
        if self.lead_weights is not None:
            lw = pred.new_tensor(self.lead_weights[:T])
        else:
            lw = pred.new_ones(T)
        lw = lw / lw.mean()
        lwv = lw.view(1, T, 1, 1, 1)

        # lead-time-weighted pixel terms
        l1 = (F.l1_loss(pred, target, reduction='none') * lwv).mean()
        mse = (F.mse_loss(pred, target, reduction='none') * lwv).mean()

        # lead-time-weighted structural terms (SSIM / MS-SSIM computed per frame)
        ssim_t = [self.ssim(pred[:, t], target[:, t]) for t in range(T)]
        ssim_val = sum(ssim_t) / T                       # unweighted mean (reporting)
        ssim_loss = sum((1.0 - ssim_t[t]) * lw[t] for t in range(T)) / T
        if self.msssim is not None:
            mss_t = [self.msssim(pred[:, t], target[:, t]) for t in range(T)]
            msssim_loss = sum((1.0 - mss_t[t]) * lw[t] for t in range(T)) / T
        else:
            msssim_loss = pred.new_tensor(0.0)

        # global regularisers
        grad = gradient_loss(pred, target)
        temporal = (F.l1_loss(pred[:, 1:] - pred[:, :-1],
                              target[:, 1:] - target[:, :-1])
                    if T > 1 else pred.new_tensor(0.0))
        tv = total_variation(pred)
        adv = advection_consistency(pred, target)

        w = self.w
        # PINN mass-continuity residual: dI/dt + div(I u), velocity field u from
        # dense Lucas-Kanade (correct image-plane physics; no Stokes-in-2D term)
        pde = (continuity_pde_residual(pred, x)
               if (w['pde'] > 0 and x is not None)
               else pred.new_tensor(0.0))
        total = (w['l1'] * l1 + w['mse'] * mse + w['ssim'] * ssim_loss +
                 w['msssim'] * msssim_loss +
                 w['grad'] * grad + w['temporal'] * temporal + w['tv'] * tv +
                 w['adv'] * adv + w['pde'] * pde)
        return total, {'l1': l1.item(), 'mse': mse.item(),
                       'ssim': float(ssim_val), 'grad': grad.item(),
                       'temporal': float(temporal), 'tv': tv.item(),
                       'adv': adv.item(), 'pde': float(pde)}
