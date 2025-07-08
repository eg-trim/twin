"""
double_pendulum_logo.py
Animate a 2-link arm (L1 = L2 = 150 mm) drawing the uploaded SVG
"""

import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from svgpathtools import parse_path, Path, Line, CubicBezier
import torch  # type: ignore  # NEW: for saving dataset

# ---------- 1. parameters ----------
L1 = L2 = 150.0          # link lengths, mm
T_total = 6.0            # total draw time, s
fps = 60                 # frames per second
N = int(T_total * fps)   # number of samples
svg_file = "logo.svg"    # save your XML as logo.svg in the same folder

# ---------- 2. load & sample SVG ----------

# Helper: recursively walk the SVG tree, accumulate translate() transforms,
# and return a list of svgpathtools Path objects with those transforms applied.


def load_svg_paths_with_translate(svg_path):
    """Parse SVG and return list[Path] with group translate() transforms applied."""

    def _parse_translate(transform_str):
        """Extract (tx, ty) from a transform string containing translate(x [,y])."""
        if not transform_str or "translate" not in transform_str:
            return 0.0, 0.0
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", transform_str)
        if not nums:
            return 0.0, 0.0
        tx = float(nums[0])
        ty = float(nums[1]) if len(nums) > 1 else 0.0
        return tx, ty

    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns_strip = lambda tag: tag.split('}')[-1]  # Remove namespace if present

    paths_out = []

    def recurse(element, acc_tx=0.0, acc_ty=0.0):
        # Update transform from this element (if any)
        t_local = element.attrib.get("transform", "")
        dtx, dty = _parse_translate(t_local)
        acc_tx_new = acc_tx + dtx
        acc_ty_new = acc_ty + dty

        for child in element:
            tag = ns_strip(child.tag)
            if tag == "g":
                recurse(child, acc_tx_new, acc_ty_new)
            elif tag == "path":
                d_str = child.attrib.get("d", "")
                if not d_str:
                    continue
                p = parse_path(d_str).translated(acc_tx_new + 1j * acc_ty_new)
                paths_out.append(p)

    recurse(root)
    return paths_out


paths = load_svg_paths_with_translate(svg_file)


def sample_segment(seg, n_pts=100):
    ts = np.linspace(0.0, 1.0, n_pts)
    return np.array([seg.point(t) for t in ts])


pts = []
for path in paths:
    for seg in path:
        pts.append(sample_segment(seg, 100))

# (M, 2) complex → real, and flip Y-axis (SVG Y down, pyplot Y up)
logo_xy = np.concatenate(pts)
logo_xy = np.column_stack([logo_xy.real, -logo_xy.imag])

# ---------- 3. centre & scale ----------
# fit in a 300 mm × 300 mm box, keep aspect ratio
xy = logo_xy - logo_xy.mean(axis=0)
scale = 300.0 / np.max(np.ptp(xy, axis=0))
xy *= scale

# resample to N equally spaced points (by arc length)
dists = np.cumsum(np.r_[0, np.hypot(*np.diff(xy, axis=0).T)])
u = np.linspace(0, dists[-1], N)
sampled = np.vstack([np.interp(u, dists, xy[:, 0]),
                     np.interp(u, dists, xy[:, 1])]).T

# ---------- 4. analytic IK ----------
def two_link_IK(x, y, L1, L2):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)              # numerical safety
    theta2 = np.arccos(D)                  # elbow-down branch
    theta1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(theta2),
                                           L1 + L2*np.cos(theta2))
    return theta1, theta2

ths1, ths2 = two_link_IK(sampled[:, 0], sampled[:, 1], L1, L2)

# ---------- 5. prepare figure ----------
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-L1-L2-10, L1+L2+10)
ax.set_ylim(-L1-L2-10, L1+L2+10)
ax.axis('off')

(link1,) = ax.plot([], [], lw=3)
(link2,) = ax.plot([], [], lw=3)
(trace,) = ax.plot([], [], lw=1)

# ---------- 6. animation callback ----------
x0, y0 = 0, 0
trace_x, trace_y = [], []

def init():
    link1.set_data([], [])
    link2.set_data([], [])
    trace.set_data([], [])
    return link1, link2, trace

def animate(i):
    th1, th2 = ths1[i], ths2[i]
    x1 = x0 + L1*np.cos(th1)
    y1 = y0 + L1*np.sin(th1)
    x2 = x1 + L2*np.cos(th1+th2)
    y2 = y1 + L2*np.sin(th1+th2)

    link1.set_data([x0, x1], [y0, y1])
    link2.set_data([x1, x2], [y1, y2])
    trace_x.append(x2)
    trace_y.append(y2)
    trace.set_data(trace_x, trace_y)
    return link1, link2, trace

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=N, interval=1000/fps, blit=True)

# ---------- 7. save outputs ----------
ani.save("logo_arm.gif", writer="pillow", fps=fps)
plt.savefig("logo_finished.png", dpi=300, bbox_inches='tight')

print("Done!  • logo_arm.gif  • logo_finished.png")

# ---------- 3.5. save trajectory in training format ----------

def _normalise_xy(xy: np.ndarray) -> np.ndarray:
    """Normalise coordinates to [-1, 1] keeping aspect ratio."""
    max_abs = np.abs(xy).max()
    return xy / max_abs if max_abs > 0 else xy


def save_logo_dataset(sampled_xy: np.ndarray, out_path: str = "logo_trajectory_dataset.pt") -> None:
    """Save sampled XY trajectory to disk in (waveforms, params) format.

    The output dictionary follows the convention used by other datasets
    in this repository:
        waveforms: (N, T, 1, 1, 2)
        params   : (N, P)  – here a dummy zero scalar
    """
    xy_norm = _normalise_xy(sampled_xy).astype(np.float32)          # (T,2)
    waveforms = torch.from_numpy(xy_norm)[None, :, None, None, :]   # (1,T,1,1,2)
    params = torch.zeros((1, 1), dtype=torch.float32)               # dummy param
    torch.save({"waveforms": waveforms, "params": params}, out_path)
    print(f"Saved trajectory dataset to {out_path} → waveforms {waveforms.shape}, params {params.shape}")

# Call immediately after creating `sampled`
save_logo_dataset(sampled)
