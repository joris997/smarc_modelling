"""npz cache for the decoded + quality-filtered corpus.

Decoding all 120 bags takes a few seconds and md5-ing them a few more; the training loop
and the benchmark both want the result, so it is cached once.  The cache key embeds the
decoder version, the schema version and a hash of the :class:`~quality.QualityConfig`, so
changing a filter threshold silently invalidates the cache rather than silently reusing
stale masks.
"""
import hashlib
import json
import pathlib

import numpy as np

try:
    from . import _bootstrap, bags, quality
except ImportError:
    import _bootstrap, bags, quality

SCHEMA_VERSION = 1
CACHE_DIR = _bootstrap.CACHE_DIR
#: Arrays stored per bag, in BagTrajectory field order.
_ARRAYS = ("t", "eta", "nu", "u_cmd", "u_fb", "valid", "seg_id")


def cache_key(cfg, quat_order):
    payload = json.dumps({
        "schema": SCHEMA_VERSION,
        "decoder": bags.cdr.DECODER_VERSION,
        "quat_order": quat_order,
        "quality": [list(kv) for kv in cfg.key()],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(cfg, quat_order):
    return CACHE_DIR / f"bags_v{SCHEMA_VERSION}_{cache_key(cfg, quat_order)}.npz"


def _save(path, trajs, cfg, quat_order):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob, meta = {}, []
    for tr in trajs:
        for a in _ARRAYS:
            blob[f"{tr.name}/{a}"] = getattr(tr, a)
        for k, m in tr.reject.items():
            blob[f"{tr.name}/reject/{k}"] = m
        meta.append({
            "name": tr.name, "md5": tr.md5, "raw_n": tr.raw_n,
            "reject_rules": sorted(tr.reject),
            "label": {"name": tr.label.name, "quality": tr.label.quality,
                      "reason": tr.label.reason, "is_test_tagged": tr.label.is_test_tagged,
                      "session": tr.label.session, "window": list(tr.label.window)},
        })
    blob["_meta"] = np.frombuffer(
        json.dumps({"key": cache_key(cfg, quat_order), "quat_order": quat_order,
                    "bags": meta}).encode(), dtype=np.uint8)
    # Uncompressed on purpose: these are small (~4 MB) and load instantly.
    np.savez(path, **blob)


def _load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["_meta"]).decode())
    trajs = []
    for entry in meta["bags"]:
        n = entry["name"]
        lab = bags.BagLabel(
            name=entry["label"]["name"], quality=entry["label"]["quality"],
            reason=entry["label"]["reason"],
            is_test_tagged=bool(entry["label"]["is_test_tagged"]),
            session=entry["label"]["session"], window=tuple(entry["label"]["window"]))
        trajs.append(bags.BagTrajectory(
            name=n, label=lab, md5=entry["md5"], raw_n=int(entry["raw_n"]),
            **{a: z[f"{n}/{a}"] for a in _ARRAYS},
            reject={k: z[f"{n}/reject/{k}"] for k in entry["reject_rules"]},
        ))
    return trajs


def load_dataset(cfg=None, quat_order="wxyz", rebuild=False, verbose=False):
    """Decode + quality-filter every bag, cached.  Returns ``list[BagTrajectory]``."""
    cfg = cfg or quality.QualityConfig()
    path = cache_path(cfg, quat_order)
    if path.exists() and not rebuild:
        if verbose:
            print(f" loading cached corpus: {path.name}")
        return _load(path)

    if verbose:
        print(f" decoding rosbags from {_bootstrap.DATA_DIR} ...")
    trajs = bags.read_all_bags(quat_order=quat_order, verbose=verbose)
    # The Hampel floor is a CORPUS-wide statistic, so it is computed once over every bag
    # (box-filtered, so the p=-758 glitch cannot set it) and shared by all of them.
    scales = quality.channel_scales(trajs, cfg)
    for tr in trajs:
        quality.apply_quality(tr, cfg, scales)
    _save(path, trajs, cfg, quat_order)
    if verbose:
        tot = sum(len(t) for t in trajs)
        val = sum(t.n_valid for t in trajs)
        print(f" decoded {len(trajs)} bags, {tot} messages, {val} valid "
              f"-> cached {path.name}")
    return trajs


def as_dict(trajs):
    return {t.name: t for t in trajs}
