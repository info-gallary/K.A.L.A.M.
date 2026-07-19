"""Inference-efficiency benchmark for the deployment table: parameters, measured
latency / throughput, peak GPU memory, and on-disk model size, on the actual GPU.
-> results/efficiency_benchmark.json. Measured (not theoretical) numbers, batch=1,
128x128x6, 4->3 frames, with CUDA sync + warmup."""
import os, json, time, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
from pidm_model import PIConditionalForecaster
try:
    from pidm_diffusion import ConditionalResidualDiffusion
    HAVE_DIFF = True
except Exception:
    HAVE_DIFF = False

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True   # autotune; needs enough warmup to settle
WARMUP, RUNS = 30, 50


def params_m(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def time_fn(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup): fn()
    if DEV == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn()
        if DEV == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    ts = np.array(ts); return float(ts.mean()), float(ts.std())


def main():
    out = {"device": (torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"),
           "input": "1x4x6x128x128", "output": "1x3x6x128x128"}
    ck = torch.load("pidm_best.pth", map_location=DEV, weights_only=False); a = ck.get("args", {})
    det = PIConditionalForecaster(base=a.get("base", 64), refine_steps=a.get("refine", 2)).to(DEV).eval()
    det.load_state_dict(ck["model"])
    x = torch.randn(1, 4, 6, 128, 128, device=DEV)

    pc = params_m(det)
    out["params_M"] = {"deterministic_core": round(pc, 2)}
    # deployable size = weights only (the .pth checkpoint also stores optimizer/EMA)
    out["deployable_weights_MB"] = {"deterministic_fp32": round(pc * 4, 1),
                                    "deterministic_fp16": round(pc * 2, 1)}
    out["checkpoint_disk_MB"] = {"deterministic_core_full_ckpt": round(os.path.getsize("pidm_best.pth")/1e6, 1)}

    with torch.no_grad():
        # deterministic single forward
        if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
        m, s = time_fn(lambda: det(x))
        out["latency_ms"] = {"deterministic": round(m, 2)}; out["latency_std_ms"] = {"deterministic": round(s, 2)}
        out["throughput_fps"] = {"deterministic": round(1000.0 / m, 1)}
        if DEV == "cuda":
            out["peak_mem_MB"] = {"deterministic": round(torch.cuda.max_memory_allocated()/1e6, 1)}

        # deterministic + 4x flip TTA
        def tta():
            o = [det(x)]
            for d in ([-1], [-2], [-1, -2]): o.append(torch.flip(det(torch.flip(x, d)), d))
            return torch.stack(o).mean(0)
        m_t, _ = time_fn(tta)
        out["latency_ms"]["deterministic_TTA4x"] = round(m_t, 2)
        out["throughput_fps"]["deterministic_TTA4x"] = round(1000.0 / m_t, 1)

        # diffusion refiner (DDIM) if available
        if HAVE_DIFF and os.path.exists("diffusion.pth"):
            dck = torch.load("diffusion.pth", map_location=DEV, weights_only=False)
            diff = ConditionalResidualDiffusion(base=dck.get("base", 64)).to(DEV).eval()
            try:
                diff.load_state_dict(dck["model"]); ok = True
            except Exception:
                ok = False
            pdf_ = params_m(diff)
            out["params_M"]["diffusion_refiner"] = round(pdf_, 2)
            out["params_M"]["total"] = round(pc + pdf_, 2)
            out["deployable_weights_MB"]["diffusion_fp32"] = round(pdf_ * 4, 1)
            if ok:
                cond = det(x)                       # deterministic prediction ydet
                for steps in (25,):
                    # sample(x_inputs, y_deterministic, steps)
                    def samp(s=steps): return diff.sample(x, cond, steps=s)
                    try:
                        if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
                        m_d, sd = time_fn(samp, warmup=3, runs=10)
                        out["latency_ms"][f"diffusion_DDIM{steps}_1sample"] = round(m_d, 1)
                        out["latency_ms"][f"det+diffusion_DDIM{steps}"] = round(out["latency_ms"]["deterministic"] + m_d, 1)
                        if DEV == "cuda":
                            out.setdefault("peak_mem_MB", {})[f"diffusion_DDIM{steps}"] = round(torch.cuda.max_memory_allocated()/1e6, 1)
                    except Exception as e:
                        out["latency_ms"][f"diffusion_DDIM{steps}_1sample"] = f"err: {type(e).__name__}: {e}"

    json.dump(out, open("results/efficiency_benchmark.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote results/efficiency_benchmark.json")


if __name__ == "__main__":
    main()
