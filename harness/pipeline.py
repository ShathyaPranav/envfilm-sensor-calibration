"""Vectorized (lane-batched) pipeline: feature operator -> PSI drift monitor ->
online logistic learner with drift-triggered reset.

One `LanePipeline` instance advances an arbitrary number of independent
simulation lanes in lockstep, one event per `step()` call, fully vectorized
across lanes with numpy. This is what makes N=1000 incidents x H=5000 events
tractable: instead of a Python loop over (incidents x steps), we loop only
over steps (H iterations) and vectorize the incident dimension.
"""
import numpy as np

D = 4
W_S = 30
W_L = 300
ETA = 0.05
EPS = 1e-6

# Bucket edge percentiles: deciles in the bulk, finer resolution in the tails.
# Plain 10-decile PSI is bucket-COUNT based, not magnitude based -- a
# perturbation that already clears the outermost decile edge contributes the
# same single bucket-count shift no matter how far out it lands, making
# severity a dead lever for short-span incidents (see PREREGISTRATION.md
# amendment 7). Extra tail edges give magnitude in the tails somewhere to go.
QS_PERCENTILES = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 99.5]
N_BUCKETS = len(QS_PERCENTILES) + 1


def fit_psi_reference(x_burnin, n_buckets=N_BUCKETS):
    """Bucket edges (deciles + tail percentiles) + reference pct per raw
    dimension, from clean burn-in (see PREREGISTRATION.md amendments 3, 5, 7
    for why this monitors raw values with tail-heavy bucket edges)."""
    edges = np.empty((D, n_buckets - 1))
    ref_counts = np.empty((D, n_buckets))
    for i in range(D):
        qs = QS_PERCENTILES
        e = np.percentile(x_burnin[:, i], qs)
        edges[i] = e
        buckets = np.digitize(x_burnin[:, i], e)
        ref_counts[i] = np.bincount(buckets, minlength=n_buckets)
    ref_pct = ref_counts / ref_counts.sum(axis=1, keepdims=True)
    return edges, ref_pct


class LanePipeline:
    def __init__(self, n_lanes, edges, ref_pct, tau_drift, reset_enabled_mask,
                 w_init=None, b_init=0.0, from_snapshot=False, freeze_learner=False,
                 feature_mode='meanvol'):
        self.n_lanes = n_lanes
        self.freeze_learner = freeze_learner
        assert feature_mode in ('meanvol', 'robust')
        self.feature_mode = feature_mode
        self.edges = edges          # (D, n_buckets-1)
        self.ref_pct = ref_pct      # (D, n_buckets)
        self.tau_drift = tau_drift
        self.reset_enabled_mask = np.asarray(reset_enabled_mask, dtype=bool)
        assert self.reset_enabled_mask.shape == (n_lanes,)

        self.w_init = np.zeros(2 * D) if w_init is None else np.asarray(w_init)
        self.b_init = b_init
        self.w = np.tile(self.w_init, (n_lanes, 1)).astype(np.float64)
        self.b = np.full(n_lanes, b_init, dtype=np.float64)

        self.buf_x = np.zeros((n_lanes, D, W_S))
        self.sum_x = np.zeros((n_lanes, D))
        self.sumsq_x = np.zeros((n_lanes, D))

        self.buf_bucket = np.zeros((n_lanes, D, W_L), dtype=np.int16)
        self.counts = np.zeros((n_lanes, D, N_BUCKETS), dtype=np.float64)

        self.above_flag = np.zeros(n_lanes, dtype=bool)
        self.local_step = 0
        # step_offset: 0 for fresh (windows fill up from empty), W_L for
        # snapshot-warm-started lanes (windows already full from history).
        self.step_offset = W_L if from_snapshot else 0

        # Per-lane circular-buffer write pointers. MUST be per-lane, not a
        # single shared scalar: different lanes in a snapshot-warm-started
        # batch were snapshotted from DIFFERENT absolute source timesteps
        # (different incident positions), so each has its own true pointer
        # position (source_local_step % W_S / % W_L) that must be restored
        # exactly on load -- see PREREGISTRATION.md bug-discovery amendment.
        # Fresh (non-snapshot) pipelines start every lane's pointer at 0,
        # matching a from-scratch run.
        self.ptr_s = np.zeros(n_lanes, dtype=np.int64)
        self.ptr_l = np.zeros(n_lanes, dtype=np.int64)
        self._lane_range = np.arange(n_lanes)

        lane_idx = np.arange(n_lanes)[:, None] * np.ones((1, D), dtype=int)
        dim_idx = np.ones((n_lanes, 1), dtype=int) * np.arange(D)[None, :]
        self._lane_idx_flat = lane_idx.ravel()
        self._dim_idx_flat = dim_idx.ravel()

    def load_snapshot(self, snap):
        """snap: dict produced by `snapshot()` on a source pipeline (broadcast
        a single-lane or matching-lane snapshot into this pipeline's lanes).
        Restores the exact circular-buffer pointers recorded at snapshot
        time -- NOT reset to 0 -- so eviction order after restore exactly
        matches uninterrupted execution."""
        self.w = snap['w'].copy()
        self.b = snap['b'].copy()
        self.buf_x = snap['buf_x'].copy()
        self.sum_x = snap['sum_x'].copy()
        self.sumsq_x = snap['sumsq_x'].copy()
        self.buf_bucket = snap['buf_bucket'].copy()
        self.counts = snap['counts'].copy()
        self.above_flag = snap['above_flag'].copy()
        self.ptr_s = snap['ptr_s'].copy()
        self.ptr_l = snap['ptr_l'].copy()
        self.local_step = 0
        self.step_offset = W_L

    def snapshot(self):
        return dict(
            w=self.w.copy(), b=self.b.copy(),
            buf_x=self.buf_x.copy(), sum_x=self.sum_x.copy(), sumsq_x=self.sumsq_x.copy(),
            buf_bucket=self.buf_bucket.copy(), counts=self.counts.copy(),
            above_flag=self.above_flag.copy(),
            ptr_s=self.ptr_s.copy(), ptr_l=self.ptr_l.copy(),
        )

    def step(self, x_obs, y_obs, suppress_update=None):
        """x_obs: (n_lanes, D), y_obs: (n_lanes,) -> loss, pred, reset_fired, drift_stat (each (n_lanes,))
        suppress_update: optional (n_lanes,) bool -- for lanes True this step, the
        SGD update is skipped (prediction/loss/window/drift-monitor state still
        advance normally). Used by ARM 3 to isolate window/state propagation from
        learner-parameter contamination: True only during a lane's own incident
        dwell interval, False before and after."""
        n = self.n_lanes
        gstep = self.step_offset + self.local_step

        # --- feature operator: rolling mean/vol over W_S ---
        # Per-lane pointer (see __init__ / bug-discovery amendment): each
        # lane evicts/writes its OWN buffer slot, not a shared scalar index.
        old_s = self.buf_x[self._lane_range, :, self.ptr_s]  # (n, D)
        self.sum_x += x_obs - old_s
        self.sumsq_x += x_obs ** 2 - old_s ** 2
        self.buf_x[self._lane_range, :, self.ptr_s] = x_obs
        self.ptr_s = (self.ptr_s + 1) % W_S
        n_eff_s = min(gstep + 1, W_S)
        if self.feature_mode == 'meanvol':
            mean = self.sum_x / n_eff_s
            var = np.maximum(self.sumsq_x / n_eff_s - mean ** 2, 1e-8)
            vol = np.sqrt(var)
            e_obs = np.concatenate([mean, vol], axis=1)  # (n, 2D)
        else:  # 'robust': rolling median / MAD over the SAME buf_x window
            valid = self.buf_x[:, :, :n_eff_s]  # (n, D, n_eff_s)
            median = np.median(valid, axis=2)
            mad = 1.4826 * np.median(np.abs(valid - median[:, :, None]), axis=2)
            mad = np.maximum(mad, 1e-8)
            e_obs = np.concatenate([median, mad], axis=1)  # (n, 2D)

        # --- learner: prequential predict, then SGD update ---
        w_prev = self.w.copy()
        logit = np.einsum('ij,ij->i', e_obs, self.w) + self.b
        pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        pred_c = np.clip(pred, 1e-9, 1 - 1e-9)
        loss = -(y_obs * np.log(pred_c) + (1 - y_obs) * np.log(1 - pred_c))
        err = pred - y_obs
        grad_e = err[:, None] * w_prev  # gradient of loss wrt e_obs, pre-update weights
        if not self.freeze_learner:
            if suppress_update is None:
                self.w -= ETA * err[:, None] * e_obs
                self.b -= ETA * err
            else:
                do_update = ~suppress_update
                self.w[do_update] -= ETA * err[do_update, None] * e_obs[do_update]
                self.b[do_update] -= ETA * err[do_update]

        # --- drift monitor: PSI over W_L on raw x_obs (see PREREGISTRATION.md
        # amendment 5 -- mean-based monitoring was reverted: the W_s=30 mean is
        # heavily autocorrelated/smoothed, so its own short-window (W_l=300)
        # empirical distribution drifts away from any fixed reference purely
        # from slow-mixing noise, giving a persistently-elevated statistic that
        # saturates the hysteresis gate regardless of incidents) ---
        bucket = np.empty((n, D), dtype=np.int16)
        for i in range(D):
            bucket[:, i] = np.digitize(x_obs[:, i], self.edges[i])
        old_bucket = self.buf_bucket[self._lane_range, :, self.ptr_l]  # (n, D)
        active_decrement = (gstep >= W_L)
        counts_flat = self.counts.reshape(-1)
        if active_decrement:
            dec_flat = (self._lane_idx_flat * D + self._dim_idx_flat) * N_BUCKETS + old_bucket.ravel()
            np.add.at(counts_flat, dec_flat, -1.0)
        inc_flat = (self._lane_idx_flat * D + self._dim_idx_flat) * N_BUCKETS + bucket.ravel()
        np.add.at(counts_flat, inc_flat, 1.0)
        self.buf_bucket[self._lane_range, :, self.ptr_l] = bucket
        self.ptr_l = (self.ptr_l + 1) % W_L

        n_eff_l = min(gstep + 1, W_L)
        actual_pct = self.counts / n_eff_l  # (n, D, n_buckets)
        ref = self.ref_pct[None, :, :]
        psi_dim = np.sum((actual_pct - ref) * np.log((actual_pct + EPS) / (ref + EPS)), axis=2)  # (n,D)
        # max across features, not mean: incidents corrupt one dim at a time,
        # so averaging with 3 undisturbed dims dilutes the signal ~4x on top
        # of the 1/W_l dilution already inherent to a short-span incident in a
        # 300-event window (see PREREGISTRATION.md amendment 6). Max-across-
        # features is also standard practice for per-feature drift alerting.
        drift_stat = psi_dim.max(axis=1)  # (n,)

        rising = (drift_stat > self.tau_drift) & (~self.above_flag)
        self.above_flag = drift_stat > self.tau_drift
        reset_fired = rising.copy()
        do_reset = rising & self.reset_enabled_mask
        if do_reset.any():
            self.w[do_reset] = self.w_init
            self.b[do_reset] = self.b_init

        self.local_step += 1
        return loss, pred, reset_fired, drift_stat, e_obs, grad_e
