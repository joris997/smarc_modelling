"""Train / val / test assignment, with the leakage traps this corpus actually contains.

Two things make a naive split wrong here:

1. **Byte-identical duplicate bags.** ``rosbag_85 == rosbag_114``, ``rosbag_3 ==
   evaluate_4`` and ``rosbag_113 == evaluate_2 == evaluate_5`` are the same ``.db3``
   under different names.  Splitting by name puts the same run on both sides.  We dedup
   by md5 and assert the md5 sets are pairwise disjoint.
2. **Session structure.** Bags recorded in one session share water, ballast, calibration
   and mocap-glitch epoch.  A per-bag random val split therefore leaks; we draw val by
   bag but forbid a whole session landing in it.

Test set is the notes' own ``[test]`` tags minus ``rosbag_112`` (5 messages over 0.2 s,
zero usable after filtering).
"""
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

try:
    from . import _bootstrap, bags, quality
except ImportError:
    import _bootstrap, bags, quality

SPLIT_DIR = _bootstrap.TRAIN_DIR / "splits"
SPLIT_FILE = SPLIT_DIR / "split_v1.yaml"


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    val_frac: float = 0.15
    #: Include Bad bags whose failure was behavioural, at a reduced weight.  Those ~29
    #: bags ("goes wrong direction", "drops rpm", "low speed") carry this corpus's only
    #: real reverse-thrust excitation; the ones whose reason names corrupt ground truth
    #: are excluded by `BagLabel.corrupt_ground_truth` regardless of this flag.
    use_bad_bags: bool = True
    bad_bag_weight: float = 0.25
    good_bag_weight: float = 1.0


@dataclass
class Splits:
    train: list = field(default_factory=list)     # bag names
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)
    eval_extra: list = field(default_factory=list)   # surviving evaluate_* bags
    smoke: list = field(default_factory=list)        # too short to train on
    dropped: dict = field(default_factory=dict)      # name -> reason
    weight: dict = field(default_factory=dict)       # name -> sample weight
    md5_alias: dict = field(default_factory=dict)    # duplicate name -> canonical name
    cfg: SplitConfig = field(default_factory=SplitConfig)

    def names(self, split):
        return {"train": self.train, "val": self.val, "test": self.test,
                "eval_extra": self.eval_extra, "smoke": self.smoke}[split]

    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def to_dict(self):
        return {
            "cfg": dataclasses.asdict(self.cfg),
            "train": sorted(self.train), "val": sorted(self.val),
            "test": sorted(self.test), "eval_extra": sorted(self.eval_extra),
            "smoke": sorted(self.smoke),
            "weight": {k: self.weight[k] for k in sorted(self.weight)},
            "md5_alias": {k: self.md5_alias[k] for k in sorted(self.md5_alias)},
            "dropped": {k: self.dropped[k] for k in sorted(self.dropped)},
        }


def _canonical(name_a, name_b):
    """Prefer the ``rosbag_*`` name (it is the one carrying a notes.txt label), then the
    lower numeric index."""
    a_is_bag = name_a.startswith("rosbag_")
    b_is_bag = name_b.startswith("rosbag_")
    if a_is_bag != b_is_bag:
        return name_a if a_is_bag else name_b
    def idx(n):
        try:
            return float(n.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return float("inf")
    return name_a if idx(name_a) <= idx(name_b) else name_b


def dedup_by_md5(trajs):
    """``(unique_trajs, alias)`` where ``alias`` maps a dropped name to its canonical one."""
    by_md5 = {}
    for tr in trajs:
        prev = by_md5.get(tr.md5)
        by_md5[tr.md5] = tr if prev is None else (
            tr if _canonical(tr.name, prev.name) == tr.name else prev)
    keep = {t.name for t in by_md5.values()}
    alias = {t.name: by_md5[t.md5].name for t in trajs if t.name not in keep}
    return [t for t in trajs if t.name in keep], alias


def make_splits(trajs, cfg=SplitConfig(), qcfg=quality.QualityConfig()):
    """Assign every bag to a split.  See the module docstring for the policy."""
    uniq, alias = dedup_by_md5(trajs)
    sp = Splits(cfg=cfg, md5_alias=alias)
    for dup, canon in alias.items():
        sp.dropped[dup] = f"duplicate .db3 of {canon}"

    pool = []
    for tr in uniq:
        ok, why = quality.bag_verdict(tr, qcfg)
        if tr.label.quality == "DNU":
            sp.dropped[tr.name] = "DNU label"
        elif not ok:
            sp.dropped[tr.name] = why
            sp.smoke.append(tr.name)
        else:
            pool.append(tr)

    by_name = {t.name: t for t in pool}

    # --- TEST: the notes' own [test] tags that survived filtering -----------
    sp.test = sorted(t.name for t in pool if t.label.is_test_tagged)
    # --- EVAL: unlabelled evaluate_* bags, reported separately --------------
    sp.eval_extra = sorted(t.name for t in pool
                           if t.name.startswith("evaluate_") and t.name not in sp.test)

    remaining = [t for t in pool if t.name not in sp.test and t.name not in sp.eval_extra]

    # Only bags with usable ground truth may train.
    trainable = [t for t in remaining if not t.label.corrupt_ground_truth]
    for t in remaining:
        if t.label.corrupt_ground_truth:
            sp.dropped[t.name] = f"corrupt ground truth ({t.label.reason})"
    if not cfg.use_bad_bags:
        for t in list(trainable):
            if t.label.quality == "Bad":
                sp.dropped[t.name] = "Bad label (use_bad_bags=False)"
        trainable = [t for t in trainable if t.label.quality != "Bad"]

    # --- VAL: session-grouped draw from the Good bags -----------------------
    rng = np.random.default_rng(cfg.seed)
    good = sorted((t.name for t in trainable if t.label.quality == "Good"))
    sessions = {}
    for n in good:
        sessions.setdefault(by_name[n].label.session, []).append(n)

    target = max(1, int(round(cfg.val_frac * len(good))))
    val = []
    for n in rng.permutation(good):
        if len(val) >= target:
            break
        s = by_name[n].label.session
        # never let a whole session go to val -- that would remove a condition entirely
        if sum(1 for v in val if by_name[v].label.session == s) >= len(sessions[s]) - 1:
            continue
        val.append(str(n))
    sp.val = sorted(val)

    sp.train = sorted(t.name for t in trainable if t.name not in sp.val)
    for n in sp.train + sp.val:
        q = by_name[n].label.quality
        sp.weight[n] = cfg.bad_bag_weight if q == "Bad" else cfg.good_bag_weight

    assert_no_leakage(sp, {t.name: t.md5 for t in trajs})
    return sp


def assert_no_leakage(sp, md5_of):
    """The one assert that catches all three duplicate classes in this corpus."""
    groups = {k: {md5_of[n] for n in sp.names(k) if n in md5_of}
              for k in ("train", "val", "test", "eval_extra")}
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = groups[a] & groups[b]
            if shared:
                names_a = [n for n in sp.names(a) if md5_of.get(n) in shared]
                names_b = [n for n in sp.names(b) if md5_of.get(n) in shared]
                raise AssertionError(
                    f"split leakage: {a} and {b} share the same .db3 -- "
                    f"{names_a} vs {names_b}")


def write_split(sp, path=SPLIT_FILE):
    """Write the split as YAML (the submodule .gitignore has a bare ``*.json``)."""
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(sp.to_dict(), sort_keys=True, default_flow_style=False))
    return path


def print_report(sp, trajs=None):
    n_valid = {t.name: t.n_valid for t in trajs} if trajs else {}
    print(f"\n=== splits (seed={sp.cfg.seed}, use_bad_bags={sp.cfg.use_bad_bags}, "
          f"bad_weight={sp.cfg.bad_bag_weight}) ===")
    for k in ("train", "val", "test", "eval_extra", "smoke"):
        names = sp.names(k)
        tot = sum(n_valid.get(n, 0) for n in names)
        extra = f", {tot} usable samples" if n_valid else ""
        print(f"  {k:11s} {len(names):3d} bags{extra}")
        if k in ("test", "val", "eval_extra"):
            print(f"              {', '.join(names)}")
    print(f"  dropped     {len(sp.dropped):3d} bags")
    if sp.md5_alias:
        print(f"  md5 aliases: " + ", ".join(f"{k}->{v}" for k, v in sorted(sp.md5_alias.items())))
    print(f"  split sha256: {sp.sha256()[:16]}")
