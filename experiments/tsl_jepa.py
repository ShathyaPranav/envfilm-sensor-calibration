"""CalJEPA: joint-embedding predictive pretraining of the LogTCN trunk on unlabelled fleet data.

The calibration benchmark labels only each unit's first co-location. The archive holds a
year of minute readings from 85 units in three cities, almost all of it without a
reference. JEPA uses that: a context encoder (the LogTCN convolution trunk) reads the last
360 minutes; a predictor must produce, from that summary, the embedding that a slowly
updated copy of the encoder (exponential moving average, no gradient) gives to the window
ending k minutes later, for k in {5, 15, 60, 180}. Predicting in embedding space, not raw
values, lets the encoder ignore unpredictable minute noise and keep what persists: the
level, the regime and how this sensor tends to respond.

Leakage rule: the validation and test units of every city (the last two of each official
list) are left out of pretraining entirely, in all periods.
The pretrained trunk is then fine-tuned with exactly the recipe used for LogTCN from scratch.
"""
import copy
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import tsl_bench as B
from tsl_data import LISTS, PMS, UNITS

DEV = 'cuda'
N = B.N
KS = (5, 15, 60, 180)
CK = os.path.join(os.environ.get('SEC_DATA', 'C:/Users/luckf/dq_data/senseurcity'), 'jepa')
CFG = dict(steps=20000, batch=512, lr=1e-3, ema=0.996, C=32)
CFG.update(eval(os.environ.get('JEPA_CFG', '{}')))


def streams(pm):
    """All minute readings of the input channel, flagged values removed, per unit, for all
    units except each city's validation and test units."""
    held = {c + '_' + u for c, l in LISTS.items() for u in l[-2:]}
    col = PMS[pm]
    out = []
    for f in sorted(os.listdir(UNITS)):
        name = f[:-8]
        if 'REF' in name or name in held:
            continue
        d = pd.read_parquet(os.path.join(UNITS, f), columns=[col, col + '_flag'])
        v = d[col].where(d[col + '_flag'].isna()).dropna().to_numpy(np.float32)
        if len(v) > N + max(KS) + 10:
            out.append(np.clip(v, 0, None) / B.SCALE)
    return out


class Trunk(nn.Module):
    """The LogTCN feature extractor: returns the state the LogTCN head reads."""

    def __init__(self, C=32):
        super().__init__()
        layers, ch = [], 1
        for d in (1, 2, 4, 8, 16, 32, 64):
            layers += [nn.Conv1d(ch, C, 3, padding=d, dilation=d), nn.GELU()]
            ch = C
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        lx = torch.log1p(torch.clamp(x * B.SCALE, min=0))
        h = self.conv(lx[:, None, :])
        return torch.cat([h[..., -1], h.mean(-1)], -1)


def pretrain(pm, seed=0):
    path = os.path.join(CK, 'trunk_%s_s%d.pt' % (pm.replace('.', ''), seed))
    if os.path.exists(path):
        return path
    os.makedirs(CK, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    S = streams(pm)
    x = torch.tensor(np.concatenate(S), device=DEV)
    starts = np.cumsum([0] + [len(s) for s in S[:-1]])
    valid = np.concatenate([np.arange(a + N, a + len(s) - max(KS)) for a, s in zip(starts, S)])
    print('  %s: %d units, %d minutes, %d windows' % (pm, len(S), len(x), len(valid)), flush=True)
    enc = Trunk(CFG['C']).to(DEV)
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    D = 2 * CFG['C']
    kemb = nn.Embedding(len(KS), D).to(DEV)
    pred = nn.Sequential(nn.Linear(2 * D, 128), nn.GELU(), nn.Linear(128, D)).to(DEV)
    params = list(enc.parameters()) + list(kemb.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=CFG['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG['lr'], total_steps=CFG['steps'])
    offs = torch.arange(-N, 0, device=DEV)
    t0 = time.time()
    for step in range(CFG['steps']):
        i = torch.as_tensor(rng.choice(valid, CFG['batch']), device=DEV)
        kk = torch.as_tensor(rng.integers(0, len(KS), CFG['batch']), device=DEV)
        shift = torch.as_tensor(KS, device=DEV)[kk]
        ctx = enc(x[i[:, None] + offs[None, :]])
        with torch.no_grad():
            z = tgt(x[(i + shift)[:, None] + offs[None, :]])
            z = (z - z.mean(0)) / (z.std(0) + 1e-4)          # standardise each dimension over the batch
        p = pred(torch.cat([ctx, kemb(kk)], -1))
        # VICReg-style variance term on the context embedding: every dimension must keep a
        # spread of at least 1 across samples, so the encoder cannot map all windows alike
        var = torch.relu(1 - torch.sqrt(ctx.var(0) + 1e-4)).mean()
        cz = ctx - ctx.mean(0)
        cov = (cz.T @ cz) / (len(cz) - 1)
        off = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / cov.shape[0]
        loss = nn.functional.smooth_l1_loss(p, z) + var + 0.04 * off
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            m = CFG['ema'] + (1 - CFG['ema']) * step / CFG['steps']
            for a, b in zip(tgt.parameters(), enc.parameters()):
                a.mul_(m).add_(b, alpha=1 - m)
        if step % 2000 == 0:
            spread = float(ctx.std(0).mean())
            if step:
                print('    variance term %.4f' % var.item(), flush=True)
            print('    step %5d  loss %.4f  embedding spread %.3f  %.0f s' % (step, loss.item(), spread,
                                                                            time.time() - t0), flush=True)
    torch.save(enc.conv.state_dict(), path)
    return path


class JepaTCN(B.TCN):
    """LogTCN whose convolution trunk starts from the CalJEPA weights for this PM size."""

    def __init__(self):
        super().__init__()
        self.conv.load_state_dict(torch.load(pretrain(B.CURRENT['pm'], 0), map_location='cpu'))


B.MODELS['CalJEPA-TCN'] = JepaTCN

LAMBDA = float(os.environ.get('JEPA_LAMBDA', '0.1'))
_POOL = {}


def pool(pm):
    """Unlabelled windows for the auxiliary JEPA loss (same leakage rule as pretraining)."""
    if pm not in _POOL:
        S = streams(pm)
        x = torch.tensor(np.concatenate(S), device=DEV)
        starts = np.cumsum([0] + [len(s) for s in S[:-1]])
        valid = np.concatenate([np.arange(a + N, a + len(s) - max(KS)) for a, s in zip(starts, S)])
        _POOL[pm] = (x, valid)
    return _POOL[pm]


class JointJEPA(JepaTCN):
    """CalJEPA-joint: the pretrained trunk keeps optimising its JEPA objective on unlabelled
    fleet windows WHILE it learns calibration, so the labelled data cannot pull the trunk
    away from the representation of the 30M unlabelled minutes.
    loss = MSE(calibration) + LAMBDA * (JEPA prediction loss + VICReg terms)."""

    def __init__(self):
        super().__init__()
        D = 2 * 32
        self.tgt = copy.deepcopy(self.conv)
        for q in self.tgt.parameters():
            q.requires_grad_(False)
        self.kemb = nn.Embedding(len(KS), D)
        self.pred = nn.Sequential(nn.Linear(2 * D, 128), nn.GELU(), nn.Linear(128, D))
        self.x, self.valid = pool(B.CURRENT['pm'])
        self.rng = np.random.default_rng(0)
        self.offs = torch.arange(-N, 0)

    def feats(self, conv, x):
        lx = torch.log1p(torch.clamp(x * B.SCALE, min=0))
        h = conv(lx[:, None, :])
        return torch.cat([h[..., -1], h.mean(-1)], -1)

    def aux_loss(self, batch=256):
        dev = self.x.device
        i = torch.as_tensor(self.rng.choice(self.valid, batch), device=dev)
        kk = torch.as_tensor(self.rng.integers(0, len(KS), batch), device=dev)
        shift = torch.as_tensor(KS, device=dev)[kk]
        offs = self.offs.to(dev)
        ctx = self.feats(self.conv, self.x[i[:, None] + offs[None, :]])
        with torch.no_grad():
            z = self.feats(self.tgt, self.x[(i + shift)[:, None] + offs[None, :]])
            z = (z - z.mean(0)) / (z.std(0) + 1e-4)
        p = self.pred(torch.cat([ctx, self.kemb(kk)], -1))
        var = torch.relu(1 - torch.sqrt(ctx.var(0) + 1e-4)).mean()
        cz = ctx - ctx.mean(0)
        cov = (cz.T @ cz) / (len(cz) - 1)
        off = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / cov.shape[0]
        return LAMBDA * (nn.functional.smooth_l1_loss(p, z) + var + 0.04 * off)

    @torch.no_grad()
    def after_step(self):
        for a, b in zip(self.tgt.parameters(), self.conv.parameters()):
            a.mul_(0.996).add_(b, alpha=0.004)


class FrozenPlus(B.TCN):
    """CalJEPA-frozen: a fresh LogTCN trunk plus the frozen pretrained JEPA trunk; the head
    reads both. The frozen branch carries what was learned from unlabelled fleet data and
    cannot be overwritten by the small labelled set."""

    def __init__(self):
        super().__init__()
        self.frozen = Trunk(32)
        self.frozen.conv.load_state_dict(torch.load(pretrain(B.CURRENT['pm'], 0), map_location='cpu'))
        for q in self.frozen.parameters():
            q.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(32 * 4 + 1, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, c=None):
        lx = torch.log1p(torch.clamp(x * B.SCALE, min=0))
        h = self.conv(lx[:, None, :])
        with torch.no_grad():
            f = self.frozen(x)
        return self.head(torch.cat([h[..., -1], h.mean(-1), f, lx[:, -1:]], -1)).squeeze(-1)


class GentleJEPA(JepaTCN):
    """CalJEPA-gentle: pretrained trunk fine-tuned at one tenth of the head's learning rate."""

    def param_groups(self):
        return [{'params': self.conv.parameters(), 'lr': B.RECIPE['lr'] * 0.1},
                {'params': self.head.parameters()}]


B.MODELS['CalJEPA-joint'] = JointJEPA
B.MODELS['CalJEPA-frozen'] = FrozenPlus
B.MODELS['CalJEPA-gentle'] = GentleJEPA

if __name__ == '__main__':
    for pm in [a for a in sys.argv[1:] if a in PMS] or list(PMS):
        pretrain(pm)
