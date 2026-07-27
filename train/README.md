# `train/` — learning a fast, physics-informed damping matrix

Trains `pinn_reduced.pt`: a small network that predicts SAM's 6×6 damping matrix as
`D = L Lᵀ`, replacing `checkpoints/pinn.pt`.

Run everything from this directory, in the `admm` conda env:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate admm
cd utils/robots/smarc_modelling/train

python bags.py --rebuild-cache --report   # decode + filter + split, print the audit
python train_damping.py --seed 0          # -> ../checkpoints/pinn_reduced.pt
python benchmark.py --speed --accuracy    # head-to-head vs pinn.pt and the white box
```

## Why a new model

Measured on an RTX 3500 Ada at the planner's production batch (`b = 176,400`):

| | `pinn.pt` | white-box | `pinn_reduced.pt` |
|---|---|---|---|
| parameters | 815,396 (50 × 128) | 0 | **3,242** |
| `λ_min(D)` | 3.2e-10 → numerically singular | 5.0 | **≥ 7.1e-3, guaranteed** |
| `λ_max(D)` | 925, unbounded in principle | 60 | **≤ 237, guaranteed** |
| `λ_max(M⁻¹D)` max | 244 → Euler `h_stab` 0.008 s | 3.30 | 3.88 → `h_stab` 0.51 s |

The planner integrates at `h = dt/n_integrator = 0.02 s`, so `pinn.pt` is *Euler-unstable* —
which is why `examples/main_sam.py` had to switch to RK4 and pay 4× the network
evaluations. The three defects it has are all training outcomes; here they are structural:

- **Strictly PD**, because `L` is lower-triangular with a positive diagonal.
- **Bounded spectrum**, because the diagonal is a scaled `sigmoid` and the off-diagonals a
  scaled `tanh`, so `λ_max ≤ maxᵢ Σⱼ|Dᵢⱼ|` follows from the head caps alone — for *any*
  weights and *any* input, including saturating ones.
- **C¹ smooth**, because the trunk is `tanh`. A 50-layer ReLU net is only C⁰, so RK4
  silently drops to first order across its kinks (measured: relu's second difference grows
  19.5× as the stencil shrinks 100×; tanh's shrinks to 0.05×).
- **Fossen structure exactly**: `L = L_lin(u) + ‖ν‖·L_quad(ν,u)` ⟺ `D = D_lin + D_quad(|ν|)`,
  so `D(ν=0)` is bounded and the `ν → 0` blow-up cannot happen.

## Layout

| file | role |
|---|---|
| `_bootstrap.py` | `sys.path` shim (this dir is outside `src/`, so not importable as a package) |
| `cdr.py` | ROS-free rosbag reader: sqlite3 + `struct` CDR walk |
| `bags.py` | `BagLabel` / `BagTrajectory`, `notes.txt` parser. **CLI entry point** |
| `quality.py` | the five per-sample filter layers + segmentation |
| `splits.py` | md5 dedup, session-grouped val, leakage assertions |
| `cache.py` | npz cache keyed on decoder + filter config |
| `targets.py` | derivatives, the `y = τ − Cν − g − Mν̇` target, robust statistics |
| `config.py` | `TrainConfig` — every hyperparameter |
| `rollout.py` | differentiable windowed rollout + all loss terms |
| `train_damping.py` | Stage A → Stage B → checkpoint. **CLI entry point** |
| `benchmark.py` | speed (S1–S4) and accuracy (A1–A3) tables. **CLI entry point** |

The model class itself lives in `../src/smarc_modelling/piml/pinn/damping.py`, because
`SAM_torch` has to import it at runtime.

## Reading the bags without ROS

`piml/utils/utility_functions.py::load_rosbag` needs `rosbag2_py`, `rclpy` and the
generated `piml_msgs` module. None of those are installed in `admm`, and `piml_msgs` is
not installed anywhere on a typical machine — so that reader simply cannot open these
bags. `cdr.py` walks the CDR payload straight out of the sqlite3 `.db3` instead, with no
dependencies.

Validated against the whole corpus: **12,418 / 12,418 messages decoded, 0 failures**,
exactly 3 bytes of trailing CDR pad on every message, `‖q‖ = 1.0` to 7.5e-9, and the
recovered feature ranges reproduce `pinn.pt`'s own stored `x_min`/`x_range` exactly.

Two things the message layout gets right that are easy to get wrong:

- The smarc messages put the **payload first and `std_msgs/Header` last** — the opposite
  of most ROS messages.
- Quaternions are returned **scalar-first** (`[qw,qx,qy,qz]`). `lib/gnc.py` and
  `SAM_torch` both treat `eta[3]` as the scalar part, while the ROS wire order is scalar
  *last*; `load_rosbag` passes the wire order straight through, which silently permutes
  the quaternion. Measured on rosbag_3, that shifts the roll/pitch target rms 39 %/47 %.

## Data hygiene

`data/notes.txt`'s footer tally (63/51/21) is **stale** — the per-line counts are
**Good 51 / Bad 41 / DNU 21** over 113 labelled bags. `rosbag_113`, `rosbag_114` and the
`evaluate_*` directories carry no label.

Three pairs of bags are **byte-identical**: `rosbag_85 == rosbag_114`,
`rosbag_3 == evaluate_4`, `rosbag_113 == evaluate_2 == evaluate_5`. Splitting by name puts
the same run on both sides of the train/test boundary, so `splits.py` dedups on the `.db3`
md5 and asserts the md5 sets are pairwise disjoint.

Label-based filtering is **not sufficient**: the `p = −757.9 rad/s` mocap glitch that
poisoned `pinn.pt`'s normalisation lives in `rosbag_5`, which `notes.txt` labels *"Good"*.
Rejection has to be per-sample, on fixed physical bounds — the 0.1 % quantile of `p` over
this corpus is still −746.99, so no quantile clip removes the glitches.

Current corpus after filtering: **7,706 / 12,418 samples (62 %)**, 52 train / 5 val / 4
test bags (test = `rosbag_3, 18, 66, 73`, the `[test]` tags minus `rosbag_112`, which has
5 messages over 0.2 s).

## The training objective, and its honest ceiling

Stage A fits the classical one-step target `D(x)ν ≈ τ − Cν − g − Mν̇`. **That target is
not fittable by any positive-definite matrix.** On the training split, ordinary least
squares gives:

| model | R² |
|---|---|
| white-box constant `D` | −0.232 |
| best constant **full** `D` | −0.071 (LS optimum is *indefinite*: eigenvalues −29.6 … +58.9) |
| full `D` + a constant bias | +0.073 |

The residual is dominated by white-box thrust and buoyancy error — the fitted per-DOF bias
is `[-0.26, -0.10, -0.37, -1.75, -2.97, 0.11]`, i.e. ~3 N·m of unmodelled pitch moment.
A PD `D` can only emit a constant force by blowing up as `ν → 0`, which is precisely why
the previous model came out stiff.

So Stage A is kept only for a like-for-like comparison, and **Stage B does the work**: a
short differentiable rollout (`H = 2 → 4 → 8 → 16`) through `SAMTorch._dyn`, matching
recorded velocity. It never differentiates the noisy 10 Hz mocap, and it optimises exactly
the quantity the benchmark reports. Model selection is always on the Stage-B validation
rollout metric, so if Stage A hurts, its weights are simply not the ones shipped — the
white-box initialisation is itself a candidate.

## Caveat: `SAM_PIML` is a diverged fork

`SAMTorch` reproduces the reference `SAM.py` dynamics to **5.3e-15**. `SAM_PIML` does
*not* — it differs from `SAM.py` by ~9.5 % relative on `ν̇` (roll and yaw rows) with the
same white-box damping and identical inputs. `benchmark.py` therefore evaluates trajectory
accuracy through `SAMTorch` only; routing it through `SAM_PIML` (as `piml/piml_sim.py::SIM`
does) would report a different vehicle's error. See
`tests/test_sam_pinn_numpy_torch_parity.py`.
