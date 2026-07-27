#!/usr/bin/env python3
"""Train the Cholesky damping model and write ``checkpoints/pinn_reduced.pt``.

    python utils/robots/smarc_modelling/train/train_damping.py --seed 0
    python .../train_damping.py --arch-sweep          # {32,64} x {tanh,silu}
    python .../train_damping.py --skip-stage-a        # rollout only

Stage A fits the one-step algebraic target (what the previous model was trained on);
Stage B fine-tunes on a short differentiable rollout.  Model selection is always on the
Stage-B validation rollout metric, so if Stage A hurts, its weights are simply not the ones
that get shipped -- see the R^2 = -0.071 caveat in ``targets.py``.
"""
import argparse
import dataclasses
import json
import pathlib
import subprocess
import time

import numpy as np
import torch

try:
    from . import _bootstrap, cache, config, quality, rollout, splits, targets
except ImportError:
    import _bootstrap, cache, config, quality, rollout, splits, targets

from smarc_modelling.piml.pinn.damping import CholeskyDamping
from smarc_modelling.vehicles.SAM import SAM


def _git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _cosine_lr(opt, base_lr, step, total, warmup_frac):
    w = max(1, int(warmup_frac * total))
    f = step / w if step < w else 0.5 * (1 + np.cos(np.pi * (step - w) / max(1, total - w)))
    for g in opt.param_groups:
        g["lr"] = base_lr * float(f)


def build_banks(cfg, device, verbose=True):
    """Decode, filter, split, and materialise the train/val/test window banks."""
    qcfg = quality.QualityConfig()
    trajs = cache.load_dataset(qcfg, verbose=verbose)
    byname = cache.as_dict(trajs)
    scfg = splits.SplitConfig(seed=cfg.seed, use_bad_bags=cfg.use_bad_bags,
                              bad_bag_weight=cfg.bad_bag_weight)
    sp = splits.make_splits(trajs, scfg, qcfg)
    splits.write_split(sp)

    dtype = getattr(torch, cfg.dtype)
    sam = SAM(0.02)
    banks = {}
    for name in ("train", "val", "test"):
        banks[name] = rollout.WindowBank(
            byname, sp.names(name), sp.weight, device, dtype, sam=sam)
        if verbose:
            print(f"  {name:5s}: {len(sp.names(name)):3d} bags, {len(banks[name]):5d} samples")
    return banks, sp, trajs


def train(cfg, args, device, banks, sp, trajs, verbose=True):
    dtype = getattr(torch, cfg.dtype)
    rng = config.seed_everything(cfg)

    tr, va = banks["train"], banks["val"]
    feat, nu, y, M, w = tr.onestep()

    # --- statistics, all from the TRAINING split only ----------------------
    mu, sigma = targets.robust_stats(feat.cpu().numpy())
    bias = targets.fit_bias(y.cpu().numpy())
    w_y = 1.0 / targets.whiten_scales(y.cpu().numpy(), bias)
    # Corpus-wide (all bags, box-filtered) on purpose: this is an INPUT scale that makes
    # the six DOFs comparable in the loss and in the reported metric, not a fit to any
    # target.  Keeping it split-independent is what lets the training curve, the
    # validation metric and benchmark.py's A2 table be read on the same axis.
    s_nu = quality.channel_scales(trajs, quality.QualityConfig())
    w_nu = torch.as_tensor(1.0 / np.maximum(s_nu, 1e-4), dtype=dtype, device=device)

    t_ = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    bias_t, w_y_t = t_(bias), t_(w_y)
    Minv_sqrt = t_(rollout.inv_sqrt_spd(M.mean(0).double().cpu().numpy()))

    model = CholeskyDamping(hidden=cfg.hidden, n_hidden=cfg.n_hidden,
                            activation=cfg.activation, fossen_split=cfg.fossen_split,
                            x_mu=mu, x_sigma=sigma).to(device=device, dtype=dtype)
    sim_tr = rollout.make_sim(model, device, dtype)
    sim_va = rollout.make_sim(model, device, dtype)

    n_par = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n model: {cfg.hidden}x{cfg.n_hidden} {cfg.activation}, {n_par} params, "
              f"lambda_cap {model.lambda_cap:.1f}, lambda_min_bound "
              f"{model.lambda_min_bound:.2e}")
        print(f" one-step target bias (unreachable by any PD D): {np.round(bias, 3)}")

    hist = {"stage_a": [], "stage_b": []}
    best = {"metric": float("inf"), "state": None, "stage": "init", "epoch": -1}

    def evaluate(stage, epoch):
        model.eval()
        m = rollout.eval_rollout(sim_va, va, 8, w_nu, cfg.n_sub, cfg.integrator)
        model.train()
        if np.isfinite(m) and m < best["metric"]:
            best.update(metric=m, stage=stage, epoch=epoch,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        return m

    # The white-box initialisation is a real candidate, not just a starting point.
    init_metric = evaluate("init", -1)
    if verbose:
        print(f" val rollout(H=8) at white-box init: {init_metric:.4f}")

    # ---------------------------------------------------------------- stage A
    if not args.skip_stage_a and cfg.stage_a_epochs > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.stage_a_lr,
                                weight_decay=cfg.weight_decay)
        for ep in range(cfg.stage_a_epochs):
            _cosine_lr(opt, cfg.stage_a_lr, ep, cfg.stage_a_epochs, cfg.warmup_frac)
            opt.zero_grad(set_to_none=True)
            ld = rollout.data_loss(model, feat, nu, y, w, w_y_t, bias_t,
                                   cfg.huber_delta, cfg.min_speed)
            sub = rollout.subsample(feat.shape[0], cfg.reg_subsample, rng, device)
            la = rollout.anchor_loss(model, feat[sub])
            ls = rollout.stiffness_loss(model, feat[sub], Minv_sqrt, cfg.stiff_target)
            loss = cfg.stage_a_w_data * ld + cfg.stage_a_w_anchor * la + cfg.stage_a_w_stiff * ls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if (ep + 1) % cfg.eval_every == 0 or ep == cfg.stage_a_epochs - 1:
                m = evaluate("A", ep)
                hist["stage_a"].append(dict(epoch=ep, loss=loss.item(), data=ld.item(),
                                            anchor=la.item(), stiff=ls.item(), val=m))
                if verbose and (ep + 1) % (cfg.eval_every * 5) == 0:
                    print(f"  A {ep+1:4d}/{cfg.stage_a_epochs}  loss {loss.item():9.4f}"
                          f"  data {ld.item():8.4f}  anchor {la.item():7.4f}  val {m:.4f}")

    # ---------------------------------------------------------------- stage B
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stage_b_lr,
                            weight_decay=cfg.weight_decay)
    bad_epochs = 0
    for ep in range(cfg.stage_b_epochs):
        H = cfg.horizon_for(ep, cfg.stage_b_epochs)
        starts = tr.starts(H)
        if starts.numel() == 0:
            continue
        _cosine_lr(opt, cfg.stage_b_lr, ep, cfg.stage_b_epochs, cfg.warmup_frac)
        ep_loss = 0.0
        for _ in range(cfg.batches_per_epoch):
            sel = torch.as_tensor(
                rng.integers(0, starts.numel(), size=min(cfg.batch_windows, starts.numel())),
                device=device)
            idx = starts[sel]
            opt.zero_grad(set_to_none=True)
            nu_hat = rollout.rollout_windows(sim_tr, tr, idx, H, cfg.n_sub, cfg.integrator)
            tgt = torch.stack([tr.NU[idx + j + 1] for j in range(H)], dim=1)
            lr_ = rollout.rollout_loss(nu_hat, tgt, w_nu, tr.w[idx], cfg.gamma,
                                       cfg.huber_delta)
            sub = rollout.subsample(feat.shape[0], cfg.reg_subsample, rng, device)
            ld = rollout.data_loss(model, feat[sub], nu[sub], y[sub], w[sub], w_y_t,
                                   bias_t, cfg.huber_delta, cfg.min_speed)
            la = rollout.anchor_loss(model, feat[sub])
            ls = rollout.stiffness_loss(model, feat[sub], Minv_sqrt, cfg.stiff_target)
            loss = (cfg.stage_b_w_roll * lr_ + cfg.stage_b_w_data * ld
                    + cfg.stage_b_w_anchor * la + cfg.stage_b_w_stiff * ls)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.stage_b_epochs - 1:
            prev = best["metric"]
            m = evaluate("B", ep)
            hist["stage_b"].append(dict(epoch=ep, H=H, loss=ep_loss / cfg.batches_per_epoch,
                                        val=m))
            # Only arm early stopping once the curriculum has reached its final horizon.
            # Validation legitimately plateaus WITHIN a stage (H is constant for a quarter
            # of the run), so patience would otherwise fire before the horizon ever grows
            # -- it truncated the 32x2-silu sweep arm at epoch 80, before H left 2.
            final_H = H == cfg.horizon_schedule[-1]
            bad_epochs = 0 if (m < prev or not final_H) else bad_epochs + cfg.eval_every
            if verbose and (ep + 1) % (cfg.eval_every * 5) == 0:
                print(f"  B {ep+1:4d}/{cfg.stage_b_epochs}  H={H:2d}"
                      f"  loss {ep_loss/cfg.batches_per_epoch:9.4f}  val {m:.4f}"
                      f"  (best {best['metric']:.4f} @ {best['stage']}{best['epoch']})")
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {ep+1} (no val improvement in "
                          f"{bad_epochs} epochs)")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return model, best, hist, dict(mu=mu, sigma=sigma, bias=bias, w_y=w_y,
                                   s_nu=s_nu, n_par=n_par, init_metric=init_metric)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--activation", default=None)
    ap.add_argument("--skip-stage-a", action="store_true")
    ap.add_argument("--arch-sweep", action="store_true")
    ap.add_argument("--epochs-a", type=int, default=None)
    ap.add_argument("--epochs-b", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.TrainConfig(seed=args.seed)
    if args.hidden:
        cfg.hidden = args.hidden
    if args.activation:
        cfg.activation = args.activation
    if args.epochs_a is not None:
        cfg.stage_a_epochs = args.epochs_a
    if args.epochs_b is not None:
        cfg.stage_b_epochs = args.epochs_b
    if args.out:
        cfg.out_name = args.out

    device = config.resolve_device(cfg)
    print(f" device {device}, dtype {cfg.dtype}, seed {cfg.seed}")
    banks, sp, trajs = build_banks(cfg, device)

    arms = [(cfg.hidden, cfg.activation)]
    if args.arch_sweep:
        arms = [(h, a) for h in (32, 64) for a in ("tanh", "silu")]

    results = []
    for hidden, act in arms:
        c = dataclasses.replace(cfg, hidden=hidden, activation=act)
        t0 = time.perf_counter()
        model, best, hist, stats = train(c, args, device, banks, sp, trajs)
        dt = time.perf_counter() - t0
        test_m = rollout.eval_rollout(
            rollout.make_sim(model, device, getattr(torch, c.dtype)),
            banks["test"], 8,
            torch.as_tensor(1.0 / np.maximum(stats["s_nu"], 1e-4),
                            dtype=getattr(torch, c.dtype), device=device),
            c.n_sub, c.integrator)
        print(f"\n [{hidden}x{c.n_hidden} {act}] params {stats['n_par']}  "
              f"val {best['metric']:.4f} (best from stage {best['stage']})  "
              f"test {test_m:.4f}  init {stats['init_metric']:.4f}  {dt:.0f}s")
        results.append(dict(cfg=c, model=model, best=best, hist=hist, stats=stats,
                            test=test_m, seconds=dt))

    pick = min(results, key=lambda r: r["best"]["metric"])

    # Save every sweep arm, not just the accuracy winner.  Accuracy alone is the wrong
    # criterion here: the whole point of the exercise is rollout cost, and a 2.6x-cheaper
    # arm that is 3% less accurate may well be the one to ship.  benchmark.py can then
    # time them all and the choice is made on the Pareto front rather than assumed.
    if len(results) > 1:
        for r in results:
            alt = (_bootstrap.CHECKPOINT_DIR /
                   f"pinn_reduced_h{r['cfg'].hidden}_{r['cfg'].activation}.pt")
            torch.save(r["model"].checkpoint(
                train={"config": r["cfg"].to_dict(), "split_sha256": sp.sha256(),
                       "val_rollout_h8": r["best"]["metric"], "test_rollout_h8": r["test"],
                       "best_stage": r["best"]["stage"], "best_epoch": r["best"]["epoch"],
                       "whitebox_init_val": r["stats"]["init_metric"]}), str(alt))
            print(f"  saved sweep arm {alt.name}  ({r['stats']['n_par']} params, "
                  f"val {r['best']['metric']:.4f})")

    c, model, stats = pick["cfg"], pick["model"], pick["stats"]

    out = _bootstrap.CHECKPOINT_DIR / c.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = model.checkpoint(
        train={
            "config": c.to_dict(),
            "split_sha256": sp.sha256(),
            "best_stage": pick["best"]["stage"], "best_epoch": pick["best"]["epoch"],
            "val_rollout_h8": pick["best"]["metric"],
            "test_rollout_h8": pick["test"],
            "whitebox_init_val": stats["init_metric"],
            "onestep_bias": stats["bias"].tolist(),
            "nu_scales": stats["s_nu"].tolist(),
            "history": pick["hist"],
            "sweep": [{"hidden": r["cfg"].hidden, "activation": r["cfg"].activation,
                       "params": r["stats"]["n_par"], "val": r["best"]["metric"],
                       "test": r["test"], "seconds": r["seconds"]} for r in results],
        },
        data={"train_bags": sorted(sp.train), "val_bags": sorted(sp.val),
              "test_bags": sorted(sp.test), "md5_alias": sp.md5_alias,
              "n_train_samples": len(banks["train"])},
        git={"parent": _git_sha(_bootstrap.REPO_ROOT),
             "submodule": _git_sha(_bootstrap.SUBMODULE_ROOT)},
    )
    torch.save(ck, str(out))
    (out.with_suffix(".config.json")).write_text(json.dumps(c.to_dict(), indent=2))
    print(f"\n wrote {out}  ({stats['n_par']} params, "
          f"val {pick['best']['metric']:.4f}, test {pick['test']:.4f})")


if __name__ == "__main__":
    main()
