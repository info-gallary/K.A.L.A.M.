"""
Genuine conditional residual diffusion model (DDPM) for the forecaster.

Design (residual/SR3-style conditional diffusion):
  - A frozen deterministic forecaster gives a prior mean  ŷ  (the conditioning).
  - The diffusion models the RESIDUAL  r0 = y_true - ŷ  with a real forward
    noising process  q(r_t|r0)=N(sqrt(ᾱ_t) r0, (1-ᾱ_t) I)  and a learned
    denoiser ε_θ(r_t, t, cond) trained with the standard DDPM noise-MSE loss.
  - cond = [4 input frames (24ch), ŷ (18ch)] concatenated channel-wise.
  - Sampling: DDIM (deterministic, few steps). Final prediction = ŷ + r0_hat.
  - N samples -> ensemble mean (pixel-accurate) + per-pixel std (uncertainty).

This is a literal conditional DDPM, not a "diffusion-style" approximation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(T, beta1=1e-4, beta2=2e-2):
    return torch.linspace(beta1, beta2, T)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, t):                      # t: [B] long
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class CondBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        g1 = 8 if cin % 8 == 0 else 1
        g2 = 8 if cout % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(g1, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb, cout)
        self.norm2 = nn.GroupNorm(g2, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CondDenoiser(nn.Module):
    """Small conditional U-Net denoiser ε_θ(r_t, t, cond)."""
    def __init__(self, res_ch=18, cond_ch=24 + 18, base=64, temb=128):
        super().__init__()
        self.tproj = nn.Sequential(SinusoidalTimeEmb(temb), nn.Linear(temb, temb),
                                   nn.SiLU(), nn.Linear(temb, temb))
        cin = res_ch + cond_ch
        self.inc = nn.Conv2d(cin, base, 3, padding=1)
        self.d1 = CondBlock(base, base, temb)
        self.d2 = CondBlock(base, base * 2, temb)
        self.d3 = CondBlock(base * 2, base * 4, temb)
        self.mid = CondBlock(base * 4, base * 4, temb)
        self.u3 = CondBlock(base * 4 + base * 4, base * 2, temb)
        self.u2 = CondBlock(base * 2 + base * 2, base, temb)
        self.u1 = CondBlock(base + base, base, temb)
        self.outc = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                  nn.Conv2d(base, res_ch, 3, padding=1))
        self.pool = nn.MaxPool2d(2)

    def forward(self, r_t, t, cond):
        te = self.tproj(t)
        x = self.inc(torch.cat([r_t, cond], dim=1))
        x1 = self.d1(x, te)
        x2 = self.d2(self.pool(x1), te)
        x3 = self.d3(self.pool(x2), te)
        m = self.mid(self.pool(x3), te)
        m = F.interpolate(m, scale_factor=2, mode="nearest")
        u3 = self.u3(torch.cat([m, x3], 1), te)
        u3 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.u2(torch.cat([u3, x2], 1), te)
        u2 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.u1(torch.cat([u2, x1], 1), te)
        return self.outc(u1)


class ConditionalResidualDiffusion(nn.Module):
    """Full conditional DDPM over the forecast residual."""
    def __init__(self, out_frames=3, bands=6, in_frames=4, T=1000, base=64):
        super().__init__()
        self.of, self.bands, self.T = out_frames, bands, T
        res_ch = out_frames * bands
        cond_ch = in_frames * bands + out_frames * bands
        self.net = CondDenoiser(res_ch=res_ch, cond_ch=cond_ch, base=base)
        betas = make_beta_schedule(T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("abar", abar)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_one_minus_abar", torch.sqrt(1 - abar))

    def _cond(self, x, ydet):                  # x:[B,4,6,H,W]  ydet:[B,3,6,H,W]
        B, _, _, H, W = x.shape
        return torch.cat([x.reshape(B, -1, H, W), ydet.reshape(B, -1, H, W)], dim=1)

    def loss(self, x, ydet, y_true):
        """DDPM noise-prediction loss on the residual r0 = y_true - ydet."""
        B, F_, C, H, W = y_true.shape
        r0 = (y_true - ydet).reshape(B, F_ * C, H, W)
        cond = self._cond(x, ydet)
        t = torch.randint(0, self.T, (B,), device=x.device)
        eps = torch.randn_like(r0)
        r_t = self.sqrt_abar[t][:, None, None, None] * r0 + \
            self.sqrt_one_minus_abar[t][:, None, None, None] * eps
        eps_hat = self.net(r_t, t, cond)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, x, ydet, steps=25):
        """DDIM sampling of the residual; returns final prediction ydet + r0_hat."""
        B, F_, C, H, W = ydet.shape
        cond = self._cond(x, ydet)
        r = torch.randn(B, F_ * C, H, W, device=x.device)
        ts = torch.linspace(self.T - 1, 0, steps, device=x.device).long()
        for i in range(steps):
            t = ts[i].repeat(B)
            eps = self.net(r, t, cond)
            ab = self.abar[ts[i]]
            r0 = (r - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            r0 = r0.clamp(-1, 1)
            if i < steps - 1:
                ab_next = self.abar[ts[i + 1]]
                r = torch.sqrt(ab_next) * r0 + torch.sqrt(1 - ab_next) * eps
            else:
                r = r0
        pred = ydet + r.reshape(B, F_, C, H, W)
        return pred.clamp(0, 1)

    @torch.no_grad()
    def sample_ensemble(self, x, ydet, n=8, steps=25):
        """n stochastic samples -> (mean prediction, per-pixel std uncertainty)."""
        outs = [self.sample(x, ydet, steps) for _ in range(n)]
        s = torch.stack(outs)
        return s.mean(0), s.std(0)
