"""Fill PAPER.md placeholders from results/metrics.json (+ ablation_results.json).
Writes PAPER_filled.md. Run after evaluate_and_plot.py (and run_ablations.py)."""
import os, json, glob
from data_split import chrono_split

OUT = "results"


def g(d, *keys, default="—"):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "—"


def main():
    m = json.load(open(os.path.join(OUT, "metrics.json")))
    abl = {}
    if os.path.exists(os.path.join(OUT, "ablation_results.json")):
        abl = json.load(open(os.path.join(OUT, "ablation_results.json")))
    n_total = len(glob.glob("cache_128/*.pt")) or 1489
    tr, va, te = chrono_split(n_total)
    p_pers = g(m, "significance_wilcoxon_p", "model_vs_persistence")
    p_pers = f"= {p_pers:.1e}" if isinstance(p_pers, (int, float)) else "(n/a)"

    full = g(abl, "full", "ssim", default=None)
    def abld(tag):
        v = g(abl, tag, "ssim", default=None)
        if v is None or full is None:
            return ("—", "—")
        return (f(v, 3), f"{v-full:+.3f}")

    rep = {
        "ssim": f(g(m, "headline", "ssim")),
        "psnr": f(g(m, "headline", "psnr_db"), 2),
        "lpips": f(g(m, "headline", "lpips"), 4),
        "pers_ssim": f(g(m, "baselines", "persistence", "ssim")),
        "pers_psnr": f(g(m, "baselines", "persistence", "psnr_db"), 2),
        "of_ssim": f(g(m, "baselines", "optical_flow", "ssim")),
        "of_psnr": f(g(m, "baselines", "optical_flow", "psnr_db"), 2),
        "d_ssim_pers": f(g(m, "improvement", "vs_persistence_ssim"), 3),
        "d_psnr_pers": f(g(m, "improvement", "vs_persistence_psnr"), 2),
        "d_ssim_of": f(g(m, "improvement", "vs_optflow_ssim"), 3),
        "p_pers": p_pers,
        "n_total": str(n_total), "n_train": str(len(tr)),
        "n_val": str(len(va)), "n_test": str(len(te)),
        "epochs": "260", "params": "43.2 M",
        "abl_full": f(full, 3) if full else "—",
    }
    for tag in ("no_residual", "no_meta", "no_refine", "no_physics"):
        rep[f"abl_{tag}"], rep[f"abl_d_{tag.split('_')[1]}"] = abld(tag)

    txt = open("PAPER.md", encoding="utf-8").read()
    for k, v in rep.items():
        txt = txt.replace("{{" + k + "}}", str(v))
    open("PAPER_filled.md", "w", encoding="utf-8").write(txt)
    print("wrote PAPER_filled.md")


if __name__ == "__main__":
    main()
