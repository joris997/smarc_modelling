"""Bag labels + trajectory objects: the clean interface everything downstream uses.

``parse_notes`` reads ``data/notes.txt`` (the hand-written experiment log) and
``read_bag`` turns one rosbag directory into a :class:`BagTrajectory`.  Quality
filtering lives in ``quality.py`` and is applied by ``load_dataset`` in ``cache.py``.

Run directly to build/refresh the cache and print the corpus report::

    python utils/robots/smarc_modelling/train/bags.py --rebuild-cache --report
"""
import argparse
import dataclasses
import pathlib
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

try:
    from . import _bootstrap, cdr
except ImportError:                                            # run as a script
    import _bootstrap, cdr                                     # noqa: F401

DATA_DIR = _bootstrap.DATA_DIR
NOTES = DATA_DIR / "notes.txt"

#: "From rosbag2_2025_05_22-15_37_01"
_RE_SESSION = re.compile(r"^From\s+(?P<session>\S+)\s*$")
#: "[150, 165] rosbag_85 - Good, low damping        [test]"
_RE_ENTRY = re.compile(
    r"^\[\s*(?P<t0>[\d.]+)\s*,\s*(?P<t1>[\d.]+|end)\s*\]\s+"
    r"(?P<name>rosbag_[\d.]+)\s*-\s*"
    r"(?P<quality>Good|Bad|DNU)\s*"
    r"(?:,\s*(?P<reason>.*?))?\s*(?P<test>\[test\])?\s*$")

#: Bad-bag reasons that mean the GROUND TRUTH is corrupt, not just that the run failed.
#: Everything else ("goes wrong direction", "drops rpm", "low speed") is perfectly good
#: dynamics data -- and the ~18 "wrong direction" bags are this corpus's only meaningful
#: reverse-thrust excitation, so throwing them away costs real coverage.
CORRUPT_REASON_PATTERNS = ("mocap", "turn rate", "bias in speeds")


def reason_is_corrupt(reason: str) -> bool:
    r = (reason or "").lower()
    return any(p in r for p in CORRUPT_REASON_PATTERNS)


@dataclass(frozen=True)
class BagLabel:
    """One ``notes.txt`` line."""
    name: str
    quality: str = "unlabelled"          # "Good" | "Bad" | "DNU" | "unlabelled"
    reason: str = ""
    is_test_tagged: bool = False
    session: str = ""                    # the "From rosbag2_..." header it sat under
    window: tuple = (None, None)         # (t0, t1) in the parent recording; t1=None => "end"

    @property
    def corrupt_ground_truth(self) -> bool:
        return self.quality == "DNU" or (self.quality == "Bad"
                                         and reason_is_corrupt(self.reason))


@dataclass
class BagTrajectory:
    """One recorded run, decoded and (optionally) quality-masked.

    Arrays are all length ``N`` and index-aligned.  ``valid`` / ``seg_id`` are filled by
    ``quality.apply_quality``; before that everything is valid and one segment.
    """
    name: str
    label: BagLabel
    md5: str
    t: np.ndarray                        # (N,)  seconds, t[0] == 0
    eta: np.ndarray                      # (N,7) [x,y,z, qw,qx,qy,qz]  SCALAR FIRST
    nu: np.ndarray                       # (N,6) body [u,v,w,p,q,r]
    u_cmd: np.ndarray                    # (N,6) [vbs_cmd, lcg_cmd, dS, dR, rpm1_cmd, rpm2_cmd]
    u_fb: np.ndarray                     # (N,6) [vbs_fb,  lcg_fb,  dS, dR, rpm1_cmd, rpm2_cmd]
    valid: np.ndarray = None             # (N,) bool
    seg_id: np.ndarray = None            # (N,) int, -1 where invalid
    reject: dict = field(default_factory=dict)   # rule name -> (N,) bool
    raw_n: int = 0                       # messages before any filtering

    def __post_init__(self):
        if self.valid is None:
            self.valid = np.ones(len(self.t), dtype=bool)
        if self.seg_id is None:
            self.seg_id = np.zeros(len(self.t), dtype=int)
        if not self.raw_n:
            self.raw_n = len(self.t)

    def __len__(self):
        return len(self.t)

    @property
    def n_valid(self):
        return int(self.valid.sum())

    def state19(self):
        """(N,19) ``[eta(7), nu(6), u_fb(6)]`` -- the state ``SAM.dynamics`` takes."""
        return np.concatenate([self.eta, self.nu, self.u_fb], axis=1)

    def state15(self):
        """(N,15) ``[eta(7), nu(6), vbs_fb, lcg_fb]`` -- the state ``SAMTorch`` takes."""
        return np.concatenate([self.eta, self.nu, self.u_fb[:, :2]], axis=1)

    def dt(self):
        return np.diff(self.t)

    def segments(self, min_len=2):
        """Contiguous runs of equal, non-negative ``seg_id``, as slices.

        Every downstream operation (derivatives, rollout windows, normalisation stats)
        runs per segment -- never across a dropped sample or a recording gap, where the
        time axis is meaningless.
        """
        out, i, n = [], 0, len(self.t)
        while i < n:
            if self.seg_id[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < n and self.seg_id[j + 1] == self.seg_id[i]:
                j += 1
            if j - i + 1 >= min_len:
                out.append(slice(i, j + 1))
            i = j + 1
        return out

    def windows(self, horizon, stride=1, min_len=None):
        """Start indices of every clean run of ``horizon + 1`` consecutive samples."""
        need = horizon + 1
        starts = []
        for sl in self.segments(min_len=min_len or need):
            starts.extend(range(sl.start, sl.stop - horizon, stride))
        return np.asarray(starts, dtype=int)


def parse_notes(path=NOTES):
    """``data/notes.txt`` -> ``{bag_name: BagLabel}``.

    The file's footer tally ("Good: 63 / Bad: 51 / DNU: 21") is STALE -- the actual
    per-line counts are 51/41/21 over 113 labelled bags.  We parse the lines and ignore
    the footer.  ``rosbag_113``, ``rosbag_114`` and the ``evaluate_*`` dirs carry no
    label at all and come back as ``"unlabelled"``.
    """
    labels, session = {}, ""
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.rstrip("\t ").strip()
        if not line:
            continue
        ms = _RE_SESSION.match(line)
        if ms:
            session = ms.group("session")
            continue
        me = _RE_ENTRY.match(line)
        if me:
            t1 = me.group("t1")
            labels[me.group("name")] = BagLabel(
                name=me.group("name"),
                quality=me.group("quality"),
                reason=(me.group("reason") or "").strip(),
                is_test_tagged=me.group("test") is not None,
                session=session,
                window=(float(me.group("t0")), None if t1 == "end" else float(t1)),
            )
    return labels


def list_bag_dirs(data_dir=DATA_DIR):
    """Every rosbag2 directory in ``data/``, sorted naturally (rosbag_2 before rosbag_10)."""
    def key(p):
        m = re.match(r"([a-z_]+)_([\d.]+)$", p.name)
        return (m.group(1), float(m.group(2))) if m else (p.name, 0.0)

    dirs = [p for p in pathlib.Path(data_dir).iterdir()
            if p.is_dir() and p.name != "cache" and list(p.glob("*.db3"))]
    return sorted(dirs, key=key)


def read_bag(bag_dir, labels=None, quat_order="wxyz"):
    """Decode one bag directory into a :class:`BagTrajectory` (no quality filtering).

    ``quat_order="xyzw"`` reproduces ``load_rosbag``'s scalar-last ordering bug-for-bug;
    it exists so the port can be shown faithful before the fix is switched on.
    """
    bag_dir = pathlib.Path(bag_dir)
    raw = cdr.read_bag_raw(bag_dir)
    labels = labels if labels is not None else {}

    q = raw["quat"]                                        # decoded scalar-first
    if quat_order == "xyzw":
        q = np.concatenate([q[:, 1:], q[:, :1]], axis=1)
    elif quat_order != "wxyz":
        raise ValueError(f"quat_order must be 'wxyz' or 'xyzw', got {quat_order!r}")

    eta = np.concatenate([raw["pos"], q], axis=1)
    nu = np.concatenate([raw["lin"], raw["ang"]], axis=1)
    fins = np.stack([raw["dS"], raw["dR"]], axis=1)
    # rpm FEEDBACK is unusable (measured range [-10378, 8594], 32% exact zeros), so both
    # control vectors carry rpm_cmd -- exactly what utility_functions.py:119 already does.
    rpm = np.stack([raw["rpm1_cmd"], raw["rpm2_cmd"]], axis=1)
    u_cmd = np.concatenate([raw["vbs_cmd"][:, None], raw["lcg_cmd"][:, None], fins, rpm], axis=1)
    u_fb = np.concatenate([raw["vbs_fb"][:, None], raw["lcg_fb"][:, None], fins, rpm], axis=1)

    t = raw["t"] - raw["t"][0] if len(raw["t"]) else raw["t"]
    return BagTrajectory(
        name=bag_dir.name,
        label=labels.get(bag_dir.name, BagLabel(name=bag_dir.name)),
        md5=cdr.bag_md5(bag_dir),
        t=t, eta=eta, nu=nu, u_cmd=u_cmd, u_fb=u_fb, raw_n=len(t),
    )


def read_all_bags(data_dir=DATA_DIR, quat_order="wxyz", verbose=True):
    """Decode every bag in ``data/``.  Warns about labels with no directory and vice versa."""
    labels = parse_notes()
    dirs = list_bag_dirs(data_dir)
    names = {d.name for d in dirs}

    missing = sorted(set(labels) - names)
    if missing:
        warnings.warn(f"notes.txt labels {len(missing)} bags with no directory: {missing}")
    unlabelled = sorted(names - set(labels))
    if unlabelled and verbose:
        print(f"  {len(unlabelled)} directories carry no notes.txt label: "
              f"{', '.join(unlabelled)}")

    trajs = []
    for d in dirs:
        trajs.append(read_bag(d, labels, quat_order=quat_order))
    return trajs


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    try:
        from . import cache, quality, splits
    except ImportError:
        import cache, quality, splits

    ds = cache.load_dataset(rebuild=args.rebuild_cache, verbose=True)
    if args.report:
        quality.print_report(ds)
        sp = splits.make_splits(ds)
        splits.print_report(sp, ds)
        print(f"  split written to {splits.write_split(sp)}")


if __name__ == "__main__":
    _main()
