"""Leakage-free chronological split for overlapping sliding-window sequences.

Each cached sequence i spans observation timestamps [i, i+WINDOW). Two windows
share frames when their indices differ by < WINDOW, so a *random* split leaks
frames across train/val/test. We therefore split chronologically and insert a
GAP = WINDOW buffer between partitions so no test window shares any frame with a
train/val window. This is the split reviewers expect for spatio-temporal
forecasting.
"""
WINDOW = 7   # sequence_length(4) + prediction_length(3)
GAP = WINDOW


def chrono_split(n, train=0.70, val=0.15, test=0.15, gap=GAP):
    """Return (train_idx, val_idx, test_idx) as lists of indices into a
    chronologically-sorted file list, with `gap` buffer dropped between splits."""
    assert abs(train + val + test - 1.0) < 1e-6
    n_tr = int(n * train)
    n_va = int(n * val)
    tr = list(range(0, n_tr))
    va = list(range(n_tr + gap, n_tr + gap + n_va))
    te = list(range(n_tr + gap + n_va + gap, n))
    return tr, va, te


if __name__ == "__main__":
    for N in (1489,):
        tr, va, te = chrono_split(N)
        print(f"N={N}: train={len(tr)} [{tr[0]}..{tr[-1]}]  "
              f"val={len(va)} [{va[0]}..{va[-1]}]  "
              f"test={len(te)} [{te[0]}..{te[-1]}]  (gap={GAP})")
