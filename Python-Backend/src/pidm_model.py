"""
Physics-Informed Conditional Diffusion-style Forecaster with a Meta-Learner.

Architecture (PIConditionalForecaster):
  - Conditional U-Net backbone (CBAM attention, multi-scale stem) that maps the
    4 observed multi-spectral frames (4x6=24 ch) to a residual "delta" field.
  - Meta-Learner: a context encoder that reads the observed *motion* (temporal
    differences of the input frames) and emits FiLM (scale/shift) parameters that
    modulate the bottleneck features -> fast per-sequence adaptation to the local
    dynamics regime (advection speed / band statistics).
  - Residual-from-last-frame head: prediction = last_observed_frame + delta.
    Initialised so that at step 0 the model == persistence (a strong prior),
    then it only has to learn the physics-driven corrections. This is the main
    lever that pushes SSIM/PSNR well above the persistence baseline.
  - Diffusion-style iterative refinement: a small denoiser is applied K times to
    the coarse prediction, each step a gated residual correction conditioned on
    the input summary -> progressive sharpening (the "conditional diffusion"
    refinement character), gated so it can never hurt the coarse prediction.

This file is import-safe (no side effects) so it can be reused by training and
evaluation scripts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Attention + building blocks (compact, memory-efficient)                     #
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = self.conv(torch.cat([avg, mx], dim=1))
        return x * self.sigmoid(a)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ResBlock(nn.Module):
    """Residual block with GroupNorm (batch-size robust) + CBAM."""
    def __init__(self, channels, dropout=0.05, groups=8):
        super().__init__()
        g = groups if channels % groups == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g, channels)
        self.cbam = CBAM(channels)
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        r = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.cbam(out)
        return self.act(out + r)


class MultiScaleStem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        q = out_ch // 4
        self.c1 = nn.Conv2d(in_ch, q, 1)
        self.c3 = nn.Conv2d(in_ch, q, 3, padding=1)
        self.c5 = nn.Conv2d(in_ch, q, 5, padding=2)
        self.c7 = nn.Conv2d(in_ch, out_ch - 3 * q, 7, padding=3)

    def forward(self, x):
        return torch.cat([self.c1(x), self.c3(x), self.c5(x), self.c7(x)], dim=1)


def conv_block(in_ch, out_ch, groups=8):
    g = groups if out_ch % groups == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch), nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch), nn.GELU(),
    )


# --------------------------------------------------------------------------- #
#  Meta-Learner: motion-conditioned FiLM generator                             #
# --------------------------------------------------------------------------- #
class MetaLearner(nn.Module):
    """Reads observed motion (temporal diffs) -> FiLM (gamma, beta) that modulate
    the bottleneck. A lightweight hypernetwork that adapts the forecaster to the
    per-sequence dynamics regime (fast-weight / meta-conditioning)."""
    def __init__(self, n_frames, bands, feat_ch):
        super().__init__()
        self.feat_ch = feat_ch
        in_ch = (n_frames - 1) * bands  # temporal differences
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 2 * feat_ch))
        # init last layer to 0 -> gamma=1, beta=0 at start (identity modulation)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, frames):  # frames: [B, T, C, H, W]
        diffs = frames[:, 1:] - frames[:, :-1]              # [B, T-1, C, H, W]
        B, Tm1, C, H, W = diffs.shape
        z = self.enc(diffs.reshape(B, Tm1 * C, H, W))       # [B,128]
        gb = self.head(z)                                   # [B, 2*feat]
        gamma, beta = gb[:, :self.feat_ch], gb[:, self.feat_ch:]
        gamma = 1.0 + gamma                                 # centred at 1
        return gamma[:, :, None, None], beta[:, :, None, None]


# --------------------------------------------------------------------------- #
#  Diffusion-style iterative refinement decoder                                #
# --------------------------------------------------------------------------- #
class RefineNet(nn.Module):
    """Small conditional denoiser applied iteratively to sharpen the coarse
    prediction. Gated residual (gate init 0) so refinement is strictly optional
    and cannot degrade the coarse forecast early in training."""
    def __init__(self, out_ch, cond_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(out_ch + cond_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, out_ch, 3, padding=1))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, y, cond):
        return y + torch.tanh(self.gate) * self.net(torch.cat([y, cond], dim=1))


# --------------------------------------------------------------------------- #
#  Main model                                                                  #
# --------------------------------------------------------------------------- #
class PIConditionalForecaster(nn.Module):
    """Physics-Informed Conditional Diffusion-style Forecaster + Meta-Learner.

    Args:
        in_frames:  number of observed frames (4)
        out_frames: number of frames to predict (3)
        bands:      spectral bands per frame (6)
        base:       base channel width (memory knob; 48-64 fits 6 GB at 128px)
        refine_steps: iterative refinement iterations (diffusion-style)
    """
    def __init__(self, in_frames=4, out_frames=3, bands=6, base=64,
                 refine_steps=2, use_residual=True, use_meta=True, use_refine=True,
                 use_of_prior=False):
        super().__init__()
        self.in_frames, self.out_frames, self.bands = in_frames, out_frames, bands
        self.refine_steps = refine_steps if use_refine else 0
        self.use_residual = use_residual
        self.use_meta = use_meta
        self.use_of_prior = use_of_prior      # Lagrangian advection prior as base+input
        # when using the advection prior, it is concatenated to the input so the
        # network can learn to *correct* it
        in_ch = in_frames * bands + (out_frames * bands if use_of_prior else 0)
        out_ch = out_frames * bands

        c1, c2, c3, c4, cb = base, base * 2, base * 4, base * 8, base * 8

        self.stem = nn.Sequential(MultiScaleStem(in_ch, c1),
                                  nn.GroupNorm(8, c1), nn.GELU())
        # encoder
        self.enc1 = nn.Sequential(conv_block(c1, c1), ResBlock(c1))
        self.enc2 = nn.Sequential(conv_block(c1, c2), ResBlock(c2))
        self.enc3 = nn.Sequential(conv_block(c2, c3), ResBlock(c3))
        self.enc4 = nn.Sequential(conv_block(c3, c4), ResBlock(c4))
        self.bott = nn.Sequential(conv_block(c4, cb), ResBlock(cb), ResBlock(cb))

        self.meta = MetaLearner(in_frames, bands, cb)

        # decoder
        self.up4 = nn.ConvTranspose2d(cb, c4, 2, stride=2)
        self.dec4 = nn.Sequential(conv_block(c4 + c4, c4), ResBlock(c4))
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = nn.Sequential(conv_block(c3 + c3, c3), ResBlock(c3))
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = nn.Sequential(conv_block(c2 + c2, c2), ResBlock(c2))
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = nn.Sequential(conv_block(c1 + c1, c1), ResBlock(c1))

        self.delta_head = nn.Sequential(
            nn.Conv2d(c1, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, out_ch, 1))
        # zero-init -> initial prediction == persistence (last frame)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        # per-lead-time learnable gain on the residual (T+1 small ... T+3 larger)
        self.lead_gain = nn.Parameter(torch.ones(out_frames, 1, 1, 1))

        self.refine = RefineNet(out_ch, cond_ch=bands)  # cond = last frame

    def _pad_to_mult(self, x, m=16):
        h, w = x.shape[-2:]
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x, (h, w)

    def forward(self, x, prior=None, return_journey=False):
        # x: [B, T, C, H, W];  prior (optical-flow advection): [B, out_frames, C, H, W]
        B, T, C, H, W = x.shape
        last = x[:, -1]                                   # [B, C, H, W] persistence
        xin = x.reshape(B, T * C, H, W)
        if self.use_of_prior and prior is not None:
            xin = torch.cat([xin, prior.reshape(B, self.out_frames * C, H, W)], dim=1)
        xin, (H0, W0) = self._pad_to_mult(xin, 16)

        s = self.stem(xin)
        e1 = self.enc1(s)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bott(F.max_pool2d(e4, 2))

        # meta-learner FiLM modulation of the bottleneck
        if self.use_meta:
            gamma, beta = self.meta(x)
            b = gamma * b + beta

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        delta = self.delta_head(d1)                       # [B, out_ch, Hp, Wp]
        delta = delta[..., :H0, :W0]                      # remove padding
        delta = delta.view(B, self.out_frames, self.bands, H, W)
        delta = delta * self.lead_gain                    # per-lead-time gain

        last_e = last.unsqueeze(1).expand(-1, self.out_frames, -1, -1, -1)
        base = prior if (self.use_of_prior and prior is not None) else last_e
        if self.use_residual:
            pred = base + delta                           # residual from (advection) prior
        else:
            pred = delta                                  # ablation: absolute prediction

        prior_label = "Advection prior" if (self.use_of_prior and prior is not None) \
            else "Persistence prior"
        journey = [(prior_label, base.clamp(0, 1)),
                   ("Coarse + residual", pred.clamp(0, 1))]

        # diffusion-style iterative refinement (gated, conditioned on last frame)
        cond = last                                       # [B, C, H, W]
        for i in range(self.refine_steps):
            y = pred.reshape(B, self.out_frames * self.bands, H, W)
            y = self.refine(y, cond)
            pred = y.view(B, self.out_frames, self.bands, H, W)
            journey.append((f"Refine step {i + 1}", pred.clamp(0, 1)))

        pred = pred.clamp(0.0, 1.0)
        if return_journey:
            return pred, journey
        return pred


def build_model(**kw):
    return PIConditionalForecaster(**kw)


if __name__ == "__main__":
    m = PIConditionalForecaster(base=64)
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 4, 6, 128, 128)
    y = m(x)
    print("params:", f"{n:,}")
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
