#!/usr/bin/env python3
"""Head-to-head: white-box vs ``pinn.pt`` vs ``pinn_reduced.pt``.

    python utils/robots/smarc_modelling/train/benchmark.py --speed --accuracy
    python .../benchmark.py --quick        # small batches, 2 bags, no compile

Speed tables (S1-S4) answer "can we afford it", accuracy tables (A1-A3) answer "is it any
good".  Writes ``train/results/benchmark.json`` and prints Markdown.

The one number that makes S2 interpretable is S3: the NumPy control-point packing loop in
``SAM.rollout_parallel`` costs a measured ~62 ms of the 0.113 s white-box figure, so the
GPU floor is ~50 ms and a learned-D delta must be judged against that, not against 113 ms.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

try:
    from . import _bootstrap, cache, quality, rollout, splits, targets
except ImportError:
    import _bootstrap, cache, quality, rollout, splits, targets

from benchmarking.sam_rollout.bench_rollout import timeit, make_batch, B_PROD, B_ONESTEP
from smarc_modelling.piml.pinn.damping import CLAMP_HI, CLAMP_LO, D_WHITEBOX
from smarc_modelling.piml.pinn.pinn import load_pinn_D
from smarc_modelling.vehicles.SAM import SAM
from smarc_modelling.vehicles.SAM_torch import SAMTorch

CKPT = _bootstrap.CHECKPOINT_DIR
VARIANTS = {
    "none":         dict(piml_type=None, ckpt=None),
    "pinn":         dict(piml_type="pinn", ckpt=CKPT / "pinn.pt"),
    "pinn_reduced": dict(piml_type="pinn", ckpt=CKPT / "pinn_reduced.pt"),
}
#: The planner's substep: SAM(dt=1.0, n_integrator=50) in examples/main_sam.py.
PLANNER_H = 1.0 / 50


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def available(names):
    out = []
    for n in names:
        c = VARIANTS[n]["ckpt"]
        if c is not None and not c.exists():
            print(f"  ! skipping {n}: {c} not found")
            continue
        out.append(n)
    return out


def make_model(name, device, dtype):
    """The bare damping network for a variant (``None`` for the white box)."""
    c = VARIANTS[name]
    if c["piml_type"] is None:
        return None
    return load_pinn_D(str(c["ckpt"])).to(device=device, dtype=dtype)


def make_sim(name, device, dtype, compile_mode=None, **kw):
    c = VARIANTS[name]
    return SAMTorch(dt=PLANNER_H, device=str(device), dtype=dtype,
                    piml_type=c["piml_type"],
                    piml_ckpt=None if c["ckpt"] is None else str(c["ckpt"]),
                    compile_mode=compile_mode, **kw)


# ============================================================== S1-S4  speed
def bench_dyn(names, device, batches, dtypes, compiles):
    rows = []
    for name in names:
        for dt_name in dtypes:
            dtype = getattr(torch, dt_name)
            for cm in compiles:
                sim = make_sim(name, device, dtype, compile_mode=None if cm == "off" else cm)
                fn = sim._dyn if cm == "off" else torch.compile(sim._dyn, dynamic=False)
                for b in batches:
                    X, U = make_batch(b, dtype, str(device))
                    try:
                        with torch.no_grad():
                            fn(X, U)      # warm / compile
                            ms = timeit(lambda: fn(X, U))
                            finite = bool(torch.isfinite(fn(X, U)).all())
                        status = "ok"
                    except Exception as e:                       # OOM, compile failure
                        ms, finite, status = float("nan"), False, type(e).__name__
                    rows.append(dict(table="S1", variant=name, dtype=dt_name, compile=cm,
                                     b=b, ms=ms, finite=finite, status=status))
                    print(f"  S1 {name:13s} {dt_name:>7s} {cm:>7s} b={b:>7d}: "
                          f"{ms:9.3f} ms  {status}")
    return rows


def model_cost(names, device):
    """S4 -- params, MACs, and where the time theoretically has to go."""
    rows = []
    for name in names:
        m = make_model(name, device, torch.float32)
        if m is None:
            rows.append(dict(table="S4", variant=name, params=0, mac_per_row=0,
                             serial_layers=0, mac_per_rollout_parallel=0))
            print(f"  S4 {name:13s} params {0:>8d}  MAC/row {0:>8d}  "
                  f"serial layers {0:>3d}   (constant D, folded into _dyn)")
            continue
        params = sum(p.numel() for p in m.parameters())
        mac = sum(int(np.prod(p.shape)) for n_, p in m.named_parameters() if "weight" in n_)
        depth = 1 + sum(1 for n_, _ in m.named_parameters() if n_.endswith(".weight"))
        rows.append(dict(table="S4", variant=name, params=params, mac_per_row=mac,
                         serial_layers=depth,
                         mac_per_rollout_parallel=mac * B_PROD * 252))
        print(f"  S4 {name:13s} params {params:>8d}  MAC/row {mac:>8d}  "
              f"serial layers {depth:>3d}")
    return rows


def bench_rollout_parallel(names, dtypes, compiles, integrators, quick):
    """S2 + S3 -- the number that actually costs GBS iterations."""
    from utils.problem import Problem
    from utils.robots.sam import SAM as SAMRobot

    rows = []
    N, Ms = (5, 4) if quick else (50, 100)
    for name in names:
        spec = VARIANTS[name]
        for dt_name in dtypes:
            for cm in compiles:
                for integ in integrators:
                    try:
                        robot = SAMRobot(
                            dt=1.0, n_integrator=50, integrator=integ, scenario="maze",
                            torch_dtype=dt_name, torch_compile=None if cm == "off" else cm,
                            piml_type=spec["piml_type"],
                            piml_ckpt=None if spec["ckpt"] is None else str(spec["ckpt"]))
                        K = len(robot.regions)
                        goal = np.zeros(robot.nx); goal[0] = 1.0
                        prob = Problem(robot=robot, N=N, start_state=np.zeros(robot.nx),
                                       goal_state=goal, R_stage=1.0, Q_terminal=1.0,
                                       M_s=Ms, u_order=1)
                        robot.configure_residuals(prob)

                        # Same construction as benchmarking/sam_rollout/bench_rollout.py
                        # so the numbers are directly comparable with its JSON.
                        from utils.controls import Controls
                        rng = np.random.default_rng(0)
                        W_X = np.zeros((N, Ms, K, robot.nx))
                        q = rng.standard_normal((N, Ms, K, 4))
                        W_X[..., 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
                        W_X[..., 7:13] = rng.uniform(-0.5, 0.5, (N, Ms, K, 6))
                        W_X[..., 13:15] = 50.0
                        W_U = np.ndarray((N - 1, Ms, K), dtype=object)
                        for i in range(N - 1):
                            for j in range(Ms):
                                c = Controls(cps=rng.uniform(-1, 1, (robot.nu, 2)))
                                for k in range(K):
                                    W_U[i, j, k] = c

                        call = lambda: robot.rollout_parallel(W_X, W_U, robot.regions)
                        out = call()
                        _sync()
                        ms = timeit(call, n=3, warmup=1)
                        finite = bool(np.isfinite(out[0]).all())
                        status = "ok" if finite else "diverged"

                        # --- S3: split the wall clock into CPU prep vs GPU integrate.
                        # rollout_parallel packs b Controls objects in a Python triple
                        # loop before it touches the GPU; that constant is a large part
                        # of the white-box figure, so a learned-D delta measured against
                        # the total is diluted.  Time the packing and the GPU core apart.
                        b = (N - 1) * Ms * K
                        s0 = time.perf_counter()
                        cps_all = np.empty((b, robot.nu, 2))
                        f = 0
                        for i in range(N - 1):
                            for j in range(Ms):
                                for k in range(K):
                                    cps_all[f] = W_U[i, j, k].cps
                                    f += 1
                        pack_ms = (time.perf_counter() - s0) * 1e3
                        X0 = W_X[:N - 1].reshape(b, robot.nx)
                        dur = np.full(b, 1.0)
                        gpu = timeit(lambda: robot._rollout_F_torch(X0, cps_all, dur, 1),
                                     n=3, warmup=1)
                        rows.append(dict(table="S3", variant=name, dtype=dt_name,
                                         compile=cm, integrator=integ, b=b,
                                         total_ms=ms, pack_ms=pack_ms, gpu_ms=gpu))
                        print(f"     S3 total {ms:8.1f} = pack {pack_ms:7.1f} (CPU) "
                              f"+ gpu {gpu:8.1f} ms")
                    except Exception as e:
                        ms, finite = float("nan"), False
                        status = f"{type(e).__name__}: {e}"[:160]
                    rows.append(dict(table="S2", variant=name, dtype=dt_name, compile=cm,
                                     integrator=integ, N=N, M_s=Ms, ms=ms,
                                     finite=finite, status=status,
                                     dyn_calls=(4 if integ == "rk4" else 1) * 63))
                    print(f"  S2 {name:13s} {dt_name:>7s} {cm:>7s} {integ:>5s}: "
                          f"{ms:9.1f} ms   {status}")
    return rows


# ============================================================ A1  one-step
def bench_onestep(names, banks, device, dtype):
    feat, nu, y, M, w = banks["test"].onestep()
    bias = torch.zeros(6, dtype=dtype, device=device)
    rows = []
    for name in names:
        m = make_model(name, device, dtype)
        with torch.no_grad():
            if m is None:
                D = torch.as_tensor(D_WHITEBOX, dtype=dtype, device=device)
                pred = nu @ D.T
            elif hasattr(m, "damping_force"):
                pred = m.damping_force(feat, nu)
            else:
                pred = torch.bmm(m(feat), nu.unsqueeze(-1)).squeeze(-1)
        e = (pred - (y - bias)).cpu().numpy()
        yv = y.cpu().numpy()
        rmse = np.sqrt((e ** 2).mean(0))
        r2 = 1 - (e ** 2).sum(0) / ((yv - yv.mean(0)) ** 2).sum(0)
        r2_tot = 1 - (e ** 2).sum() / ((yv - yv.mean(0)) ** 2).sum()
        rows.append(dict(table="A1", variant=name,
                         rmse=rmse.tolist(), mae=np.abs(e).mean(0).tolist(),
                         r2=r2.tolist(), r2_total=float(r2_tot),
                         rmse_total=float(np.sqrt((e ** 2).mean())),
                         rmse_force=float(np.sqrt((e[:, :3] ** 2).mean())),
                         rmse_moment=float(np.sqrt((e[:, 3:] ** 2).mean()))))
        print(f"  A1 {name:13s} total {rows[-1]['rmse_total']:8.3f}  "
              f"force {rows[-1]['rmse_force']:8.3f} N  moment {rows[-1]['rmse_moment']:7.3f} Nm"
              f"  R2 {r2_tot:+.3f}")
    return rows


# ============================================ A2  multi-step vs mocap
def bench_trajectory(names, banks, trajs, sp, device, dtype, horizons=(10, 30)):
    """Whitened velocity RMSE and position drift over increasing horizons.

    Windowed and re-initialised (like training), not one 15 s open-loop shot: with a
    white-box bias this large a full-bag rollout measures the bias, not the damping.
    Bag-level bootstrap CIs because the test set is 4 bags / ~450 samples.
    """
    s_nu = quality.channel_scales(trajs, quality.QualityConfig())
    w_nu = torch.as_tensor(1.0 / np.maximum(s_nu, 1e-4), dtype=dtype, device=device)
    bank = banks["test"]
    rows = []
    for name in names:
        m = make_model(name, device, dtype)
        # tau_act MUST match across variants and match training (rollout.make_sim pins
        # 0.3).  Left to default it becomes the integration substep, so the white-box arm
        # would run a 15x faster actuator than the learned arms and the comparison would
        # partly measure the actuator model rather than the damping.
        sim = (make_sim(name, device, dtype, tau_act=0.3) if m is None
               else rollout.make_sim(m, device, dtype))
        entry = dict(table="A2", variant=name)
        for H in horizons:
            idx = bank.starts(H)
            if idx.numel() == 0:
                entry[f"h{H}"] = None
                continue
            with torch.no_grad():
                nu_hat = rollout.rollout_windows(sim, bank, idx, H, n_sub=2)
                tgt = torch.stack([bank.NU[idx + j + 1] for j in range(H)], dim=1)
                err = ((nu_hat - tgt) * w_nu).pow(2).mean(dim=(1, 2)).cpu().numpy()
            err = np.nan_to_num(err, nan=1e6, posinf=1e6)
            bag = bank.bag[idx].cpu().numpy()
            per_bag = np.array([np.sqrt(err[bag == b].mean())
                                for b in np.unique(bag) if (bag == b).any()])
            rng = np.random.default_rng(0)
            boot = [per_bag[rng.integers(0, len(per_bag), len(per_bag))].mean()
                    for _ in range(2000)]
            entry[f"h{H}"] = dict(rmse=float(np.sqrt(err.mean())),
                                  per_bag=per_bag.tolist(),
                                  ci=[float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5))],
                                  diverged=bool((err > 1e5).any()), n_windows=int(idx.numel()))
            print(f"  A2 {name:13s} H={H:3d}  whitened RMSE {entry[f'h{H}']['rmse']:8.3f}"
                  f"  CI [{entry[f'h{H}']['ci'][0]:.3f}, {entry[f'h{H}']['ci'][1]:.3f}]"
                  f"{'  DIVERGED' if entry[f'h{H}']['diverged'] else ''}")
        rows.append(entry)
    return rows


# ============================================ A3  damping-field diagnostics
def bench_field(names, device, n=100_000, seed=0):
    """PD margin, spectral bound and the Euler step limit each model actually implies."""
    sam = SAM(0.02)
    x19 = np.zeros(19); x19[3] = 1.0; x19[13:15] = 50.0
    sam.dynamics(x19, np.array([50.0, 50.0, 0.0, 0.0, 0.0, 0.0]))
    Ms = rollout.inv_sqrt_spd(sam.M)
    Ms_t = torch.as_tensor(Ms, dtype=torch.float64, device=device)

    g = torch.Generator(device="cpu").manual_seed(seed)
    lo = torch.tensor(CLAMP_LO, dtype=torch.float64)
    hi = torch.tensor(CLAMP_HI, dtype=torch.float64)
    X = (lo + (hi - lo) * torch.rand(n, 12, generator=g, dtype=torch.float64)).to(device)

    rows = []
    for name in names:
        m = make_model(name, device, torch.float64)
        ev_lo, ev_hi, lam = [], [], []
        # Chunked, and the eigendecomposition is done on the CPU: cuSOLVER's batched
        # syevj picks a workspace proportional to the batch (it asked for 10.5 GiB at
        # n=20000 6x6 matrices and OOM'd).  These are 6x6 -- LAPACK is plenty fast.
        for s in range(0, n, 8192):
            Xc = X[s:s + 8192]
            with torch.no_grad():
                D = (torch.as_tensor(D_WHITEBOX, dtype=torch.float64,
                                     device=device).expand(Xc.shape[0], 6, 6)
                     if m is None else m(Xc))
                S = (Ms_t @ D @ Ms_t).cpu()
                D = D.cpu()
            e = torch.linalg.eigvalsh(0.5 * (D + D.transpose(-2, -1)))
            ev_lo.append(e[:, 0]); ev_hi.append(e[:, -1])
            lam.append(torch.linalg.eigvalsh(0.5 * (S + S.transpose(-2, -1)))[:, -1])
        ev = torch.stack([torch.cat(ev_lo), torch.cat(ev_hi)], dim=1)
        lam = torch.cat(lam).numpy()
        lam_min = float(ev[:, 0].min()); lam_max = float(ev[:, -1].max())
        p50, p99, mx = np.percentile(lam, 50), np.percentile(lam, 99), lam.max()
        cap = getattr(m, "lambda_cap", None) if m is not None else float(np.max(D_WHITEBOX))
        rows.append(dict(table="A3", variant=name, lam_min=lam_min, lam_max=lam_max,
                         lambda_cap=cap,
                         lambda_min_bound=getattr(m, "lambda_min_bound", None) if m else None,
                         Minv_p50=float(p50), Minv_p99=float(p99), Minv_max=float(mx),
                         h_stab=float(2.0 / mx), planner_h=PLANNER_H,
                         euler_stable=bool(2.0 / mx > PLANNER_H)))
        print(f"  A3 {name:13s} lam_min {lam_min:11.3e}  lam_max {lam_max:8.2f}"
              f"  lam_max(M^-1 D) p50 {p50:7.2f} p99 {p99:7.2f} max {mx:8.2f}"
              f"  -> h_stab {2/mx:.4f}s  {'OK' if 2/mx > PLANNER_H else 'UNSTABLE'} at h=0.02")
    return rows


def _md(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def print_markdown(result):
    """Emit the tables ready to paste into benchmarking/sam_pinn_reduced.md."""
    rows = result["rows"]
    pick = lambda t: [r for r in rows if r["table"] == t]
    dof = ["u", "v", "w", "p", "q", "r"]
    buf = ["\n" + "=" * 78, "MARKDOWN", "=" * 78]

    if pick("S4"):
        buf += ["\n### S4 — model cost\n",
                _md(["variant", "params", "MAC/row", "serial layers"],
                    [[r["variant"], f'{r["params"]:,}', f'{r["mac_per_row"]:,}',
                      r["serial_layers"]] for r in pick("S4")])]
    if pick("S1"):
        buf += ["\n### S1 — `_dyn` latency (ms)\n",
                _md(["variant", "dtype", "compile", "b", "ms"],
                    [[r["variant"], r["dtype"], r["compile"], f'{r["b"]:,}',
                      f'{r["ms"]:.3f}' if np.isfinite(r["ms"]) else r["status"]]
                     for r in pick("S1")])]
    if pick("S2"):
        base = {(r["dtype"], r["compile"], r["integrator"]): r["ms"]
                for r in pick("S2") if r["variant"] == "none"}
        buf += ["\n### S2 — `rollout_parallel`, production shape\n",
                _md(["variant", "dtype", "compile", "integrator", "ms", "x white-box", "status"],
                    [[r["variant"], r["dtype"], r["compile"], r["integrator"],
                      f'{r["ms"]:.1f}' if np.isfinite(r["ms"]) else "-",
                      (f'{r["ms"] / base[(r["dtype"], r["compile"], r["integrator"])]:.2f}x'
                       if np.isfinite(r["ms"])
                       and base.get((r["dtype"], r["compile"], r["integrator"])) else "-"),
                      r["status"]] for r in pick("S2")])]
    if pick("S3"):
        buf += ["\n### S3 — where the wall clock goes\n",
                _md(["variant", "integrator", "total ms", "CPU pack ms", "GPU ms"],
                    [[r["variant"], r["integrator"], f'{r["total_ms"]:.1f}',
                      f'{r["pack_ms"]:.1f}', f'{r["gpu_ms"]:.1f}'] for r in pick("S3")])]
    if pick("A1"):
        buf += ["\n### A1 — one-step damping force, held-out bags\n",
                _md(["variant", "RMSE total", "force (N)", "moment (N·m)", "R²"]
                    + [f"RMSE {d}" for d in dof],
                    [[r["variant"], f'{r["rmse_total"]:.3f}', f'{r["rmse_force"]:.3f}',
                      f'{r["rmse_moment"]:.3f}', f'{r["r2_total"]:+.3f}']
                     + [f"{v:.3f}" for v in r["rmse"]] for r in pick("A1")])]
    if pick("A2"):
        hs = [k for k in pick("A2")[0] if k.startswith("h")]
        buf += ["\n### A2 — multi-step rollout vs mocap (whitened RMSE, 95% bag bootstrap)\n",
                _md(["variant"] + [f"H={h[1:]}" for h in hs],
                    [[r["variant"]] + [
                        ("diverged" if r[h]["diverged"] else
                         f'{r[h]["rmse"]:.3f} [{r[h]["ci"][0]:.2f}, {r[h]["ci"][1]:.2f}]')
                        if r.get(h) else "-" for h in hs] for r in pick("A2")])]
    if pick("A3"):
        buf += ["\n### A3 — damping field over the planner's clamp box\n",
                _md(["variant", "λ_min(D)", "λ_max(D)", "cap", "λ_max(M⁻¹D) p50 / p99 / max",
                     "Euler h_stab", "stable at h=0.02"],
                    [[r["variant"], f'{r["lam_min"]:.3e}', f'{r["lam_max"]:.1f}',
                      f'{r["lambda_cap"]:.0f}' if r["lambda_cap"] else "-",
                      f'{r["Minv_p50"]:.2f} / {r["Minv_p99"]:.2f} / {r["Minv_max"]:.2f}',
                      f'{r["h_stab"]:.4f} s', "yes" if r["euler_stable"] else "**NO**"]
                     for r in pick("A3")])]
    print("\n".join(buf))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="none,pinn,pinn_reduced")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dtypes", default="float32")
    ap.add_argument("--compile", dest="compiles", default="off,default")
    ap.add_argument("--integrators", default="rk4,euler")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.speed or args.accuracy):
        args.speed = args.accuracy = True

    names = available([n.strip() for n in args.variants.split(",")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtypes = args.dtypes.split(",")
    compiles = args.compiles.split(",")
    out_dir = pathlib.Path(args.out) if args.out else _bootstrap.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f" device {device}   variants {names}")
    result = {"device": str(device), "torch": torch.__version__,
              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
              "planner_h": PLANNER_H, "rows": []}

    if args.speed:
        print("\n--- S4 model cost ---")
        result["rows"] += model_cost(names, device)
        print("\n--- S1 _dyn latency ---")
        batches = [1, B_ONESTEP] + ([] if args.quick else [B_PROD])
        result["rows"] += bench_dyn(names, device, batches, dtypes, compiles)
        print("\n--- S2 rollout_parallel (production shape) ---")
        result["rows"] += bench_rollout_parallel(
            names, dtypes, compiles, args.integrators.split(","), args.quick)

    if args.accuracy:
        print("\n--- loading corpus ---")
        qcfg = quality.QualityConfig()
        trajs = cache.load_dataset(qcfg, verbose=True)
        byname = cache.as_dict(trajs)
        sp = splits.make_splits(trajs, splits.SplitConfig(), qcfg)
        dtype = torch.float32
        banks = {"test": rollout.WindowBank(byname, sp.names("test"), sp.weight,
                                            device, dtype, sam=SAM(0.02))}
        print(f"  test: {len(sp.names('test'))} bags, {len(banks['test'])} samples "
              f"({', '.join(sp.names('test'))})")
        print("\n--- A1 one-step damping force ---")
        result["rows"] += bench_onestep(names, banks, device, dtype)
        print("\n--- A2 multi-step rollout vs mocap ---")
        result["rows"] += bench_trajectory(names, banks, trajs, sp, device, dtype)
        print("\n--- A3 damping field ---")
        result["rows"] += bench_field(names, device, n=20000 if args.quick else 100000)
        result["test_bags"] = sp.names("test")

    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    print_markdown(result)
    print(f"\n wrote {path}")


if __name__ == "__main__":
    main()
