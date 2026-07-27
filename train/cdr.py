"""ROS-free reader for the ``/synched_data`` rosbags in ``data/``.

Why this exists instead of ``piml/utils/utility_functions.py::load_rosbag``: that reader
needs ``rosbag2_py`` + ``rclpy`` + the generated ``piml_msgs`` Python module, and none of
those are installed in the ``admm`` env — nor is ``piml_msgs`` installed anywhere on a
typical machine, so even sourcing ``/opt/ros/humble`` cannot deserialize these bags.
The recording format is fully determined by the ``.msg`` definitions, though, so we walk
the CDR payload directly out of the sqlite3 ``.db3`` with ``struct``.  Zero dependencies.

Validated against the full corpus: **12,418 / 12,418 messages decoded, 0 failures**,
exactly 3 bytes of trailing CDR pad on every message, ``|q| == 1.0`` to 7.5e-9, and the
recovered feature ranges reproduce ``checkpoints/pinn.pt``'s stored ``x_min``/``x_range``
exactly (``u in [-0.53105, 0.63966]``, ``dS = +-0.12217``, ``rpm in [-1100, 800]``).

Wire layout (``piml_msgs/msg/SynchedData``).  Note the smarc convention: the payload
comes FIRST and ``std_msgs/Header`` LAST, which is the opposite of most ROS messages and
the single easiest thing to get wrong here::

    rosgraph_msgs/Clock       clock              # builtin_interfaces/Time
    sensor_msgs/Imu           imu                # header, quat, 3x(vec3 + 9-cov)
    smarc_msgs/PercentStamped lcg_cmd            # float32 value; Header header
    smarc_msgs/PercentStamped lcg_fb
    nav_msgs/Odometry         odom_gt            # header, child_frame_id, pose+36, twist+36
    sam_msgs/ThrusterAngles   thrust_vector_cmd  # float32 vertical; float32 horizontal; Header
    smarc_msgs/ThrusterFeedback thruster1_fb     # Header; int32 rpm; float32 dc, current, torque
    smarc_msgs/ThrusterFeedback thruster2_fb
    smarc_msgs/PercentStamped vbs_cmd
    smarc_msgs/PercentStamped vbs_fb
    piml_msgs/ThrusterRPMStamped thruster1_cmd   # int32 rpm; Header header
    piml_msgs/ThrusterRPMStamped thruster2_cmd

``ThrusterAngles`` is *vertical* then *horizontal*; ``dS`` (stern plane) is the vertical
one and ``dR`` (rudder) the horizontal, matching ``load_rosbag``'s by-name access.
"""
import hashlib
import pathlib
import sqlite3
import struct

import numpy as np

#: Bumped whenever the decoding changes, so the npz cache invalidates itself.
DECODER_VERSION = 1

#: What we assert on every message; a layout error shows up here immediately.
EXPECTED_TRAILING_PAD = 3


class CDRError(RuntimeError):
    pass


class _CDR:
    """Minimal little/big-endian CDR (XCDR1) cursor.

    Alignment in CDR is relative to the start of the payload, i.e. *after* the 4-byte
    encapsulation header — hence the ``- 4`` in ``_align``.
    """

    __slots__ = ("b", "o", "e")

    def __init__(self, buf):
        if len(buf) < 4 or buf[0] != 0x00:
            raise CDRError(f"not a CDR encapsulation header: {buf[:4].hex()}")
        self.e = "<" if buf[1] in (1, 3) else ">"
        self.b = buf
        self.o = 4

    def _align(self, n):
        r = (self.o - 4) % n
        if r:
            self.o += n - r

    def _prim(self, code, n):
        self._align(n)
        (v,) = struct.unpack_from(self.e + code, self.b, self.o)
        self.o += n
        return v

    def i32(self):
        return self._prim("i", 4)

    def u32(self):
        return self._prim("I", 4)

    def f32(self):
        return self._prim("f", 4)

    def f64(self):
        return self._prim("d", 8)

    def f64s(self, n):
        self._align(8)
        v = struct.unpack_from(f"{self.e}{n}d", self.b, self.o)
        self.o += 8 * n
        return v

    def string(self):
        n = self.u32()                       # length INCLUDING the null terminator
        s = self.b[self.o:self.o + n - 1].decode("utf-8", "replace")
        self.o += n
        return s

    def time(self):
        return self.i32() + self.u32() * 1e-9

    def header(self):
        return self.time(), self.string()

    # --- the four smarc leaf messages -------------------------------------
    def percent_stamped(self):
        v = self.f32()
        t, _ = self.header()
        return v, t

    def rpm_stamped(self):
        r = self.i32()
        t, _ = self.header()
        return r, t

    def thruster_feedback(self):
        t, _ = self.header()
        return self.i32(), self.f32(), self.f32(), self.f32(), t


def decode_synched_data(buf):
    """Decode one serialized ``SynchedData`` into a flat dict of scalars/tuples.

    Returns ``(fields, n_unconsumed)``.  ``n_unconsumed`` should be
    ``EXPECTED_TRAILING_PAD``; anything else means the layout above is wrong for this
    bag and the caller must not trust the values.
    """
    c = _CDR(buf)
    m = {}
    m["clock"] = c.time()

    # sensor_msgs/Imu -- recorded but never populated in this corpus; skipped wholesale.
    c.header()
    c.f64s(4)                                  # orientation
    c.f64s(9)                                  # orientation_covariance
    c.f64s(3); c.f64s(9)                       # angular_velocity + cov
    c.f64s(3); c.f64s(9)                       # linear_acceleration + cov

    m["lcg_cmd"], m["t"] = c.percent_stamped()
    m["lcg_fb"], _ = c.percent_stamped()

    # nav_msgs/Odometry -- the mocap ground truth.
    m["odom_t"], m["frame_id"] = c.header()
    m["child_frame_id"] = c.string()
    m["pos"] = c.f64s(3)
    m["quat_xyzw"] = c.f64s(4)                 # ROS order: x, y, z, w
    c.f64s(36)
    m["lin"] = c.f64s(3)
    m["ang"] = c.f64s(3)
    c.f64s(36)

    m["dS"] = c.f32()                          # thruster_vertical_radians   (stern plane)
    m["dR"] = c.f32()                          # thruster_horizontal_radians (rudder)
    c.header()

    m["rpm1_fb"], m["dc1"], m["cur1"], m["torque1"], _ = c.thruster_feedback()
    m["rpm2_fb"], m["dc2"], m["cur2"], m["torque2"], _ = c.thruster_feedback()

    m["vbs_cmd"], _ = c.percent_stamped()
    m["vbs_fb"], _ = c.percent_stamped()
    m["rpm1_cmd"], _ = c.rpm_stamped()
    m["rpm2_cmd"], _ = c.rpm_stamped()

    return m, len(buf) - c.o


def bag_db3(bag_dir):
    """The single ``.db3`` inside a rosbag2 directory."""
    bag_dir = pathlib.Path(bag_dir)
    files = sorted(bag_dir.glob("*.db3"))
    if len(files) != 1:
        raise CDRError(f"expected exactly one .db3 in {bag_dir}, found {len(files)}")
    return files[0]


def bag_md5(bag_dir):
    """md5 of the ``.db3``.  Several bags in ``data/`` are byte-identical copies of each
    other (``rosbag_85 == rosbag_114``, ``rosbag_3 == evaluate_4``,
    ``rosbag_113 == evaluate_2 == evaluate_5``); the split builder dedups on this so the
    test set cannot leak into training."""
    h = hashlib.md5()
    with open(bag_db3(bag_dir), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bag_raw(bag_dir):
    """Decode a whole bag into column arrays, in recorded order, with no filtering.

    Quaternions are returned SCALAR FIRST (``[qw, qx, qy, qz]``): ``lib/gnc.py`` and
    ``SAM_torch`` both treat ``eta[3]`` as the scalar part, while the ROS wire order is
    scalar last.  ``load_rosbag`` passes the wire order straight through, which silently
    feeds those functions a permuted quaternion — measured on rosbag_3, that shifts the
    roll/pitch damping target rms by 39%/47%.  See ``bags.read_bag(quat_order=...)``.
    """
    con = sqlite3.connect(str(bag_db3(bag_dir)))
    try:
        rows = con.execute("SELECT data FROM messages ORDER BY timestamp").fetchall()
    finally:
        con.close()

    n = len(rows)
    out = {
        "t": np.empty(n), "pos": np.empty((n, 3)), "quat": np.empty((n, 4)),
        "lin": np.empty((n, 3)), "ang": np.empty((n, 3)),
        "vbs_cmd": np.empty(n), "vbs_fb": np.empty(n),
        "lcg_cmd": np.empty(n), "lcg_fb": np.empty(n),
        "dS": np.empty(n), "dR": np.empty(n),
        "rpm1_cmd": np.empty(n), "rpm2_cmd": np.empty(n),
        "rpm1_fb": np.empty(n), "rpm2_fb": np.empty(n),
    }
    for i, (blob,) in enumerate(rows):
        m, left = decode_synched_data(blob)
        if left != EXPECTED_TRAILING_PAD:
            raise CDRError(
                f"{bag_dir}: message {i} left {left} bytes unconsumed "
                f"(expected {EXPECTED_TRAILING_PAD}) -- the message layout in this "
                f"module does not match this bag.")
        out["t"][i] = m["t"]
        out["pos"][i] = m["pos"]
        qx, qy, qz, qw = m["quat_xyzw"]
        out["quat"][i] = (qw, qx, qy, qz)              # -> SCALAR FIRST
        out["lin"][i] = m["lin"]
        out["ang"][i] = m["ang"]
        for k in ("vbs_cmd", "vbs_fb", "lcg_cmd", "lcg_fb", "dS", "dR",
                  "rpm1_cmd", "rpm2_cmd", "rpm1_fb", "rpm2_fb"):
            out[k][i] = m[k]
    return out
