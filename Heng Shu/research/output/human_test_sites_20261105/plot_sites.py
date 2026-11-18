"""Draw the planned field layout and calculate ideal array response similarity."""
from pathlib import Path
from math import cos, sin, radians

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent
RX_X = 3.65
RADIUS = 2.60
WAVELENGTH = 299792458 / 5.23e9
ANTENNA_Y = (np.arange(1, 13) - 6.5) * 0.028


def planned_point(angle):
    return RX_X - RADIUS * cos(radians(angle)), -RADIUS * sin(radians(angle))


def unit_response(point, indices):
    x, y = point
    paths = np.hypot(RX_X - x, ANTENNA_Y[indices] - y)
    return np.exp(-2j * np.pi * (paths - paths[0]) / WAVELENGTH) / np.sqrt(len(indices))


def main():
    angles = {"H1": -25, "H2": -37, "H3": 25, "H4": 37}
    points = {name: planned_point(angle) for name, angle in angles.items()}
    for first, second in [("H1", "H2"), ("H3", "H4")]:
        indices_all = np.arange(12)
        values = []
        for indices in [*np.split(indices_all, 4), indices_all]:
            values.append(abs(np.vdot(unit_response(points[first], indices),
                                      unit_response(points[second], indices))))
        print(first, second, "ideal G1, G2, G3, G4, coherent 12:", np.round(values, 4))

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8.0, 6.8), layout="constrained")
    ax.plot([0, RX_X], [0, 0], color="#9aa5ae", lw=1.5)
    arc = np.deg2rad(np.linspace(-43, 43, 160))
    ax.plot(RX_X - RADIUS*np.cos(arc), -RADIUS*np.sin(arc),
            ls="--", lw=1.2, color="#aeb9c1")
    for name, (x, y) in points.items():
        color = "#116761" if y > 0 else "#355792"
        ax.plot([x, RX_X], [y, 0], color=color, alpha=.23, lw=1.2)
        ax.scatter(x, y, s=85, color=color, zorder=4)
        shift = (-9, 4) if name in ("H1", "H3") else (5, 6 if y > 0 else -14)
        label = f"{name}  ({x:.2f}, {y:+.2f})"
        ax.annotate(label, (x, y), xytext=shift, textcoords="offset points",
                    ha="right" if shift[0] < 0 else "left", color=color, fontsize=10)
    for name, y in [("P1", 1), ("P2", -1)]:
        ax.scatter(1.64, y, s=50, facecolors="white", edgecolors="#7b8086", zorder=3)
        ax.annotate(f"{name} 试采", (1.64, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, color="#676c70")
    ax.scatter(0, 0, marker="D", s=72, color="#172e44")
    ax.text(0, -.17, "TX (0, 0)", ha="center", fontsize=10)
    ax.scatter(np.full(12, RX_X), ANTENNA_Y, s=10, color="#172e44", zorder=4)
    ax.text(RX_X, -.34, "RX (3.65, 0)", ha="center", fontsize=10)
    ax.scatter(1.825, 0, marker="*", s=100, color="#af5961", zorder=4)
    ax.text(1.825, -.16, "人体面向中点", ha="center", fontsize=9, color="#943d48")
    ax.set(xlim=(-.45, 4.22), ylim=(-2.06, 2.05), aspect="equal",
           xlabel="全局 X / m", ylabel="全局 Y / m")
    ax.set_title("静止人体测试站位｜每次只站一个点\n实心为四个正式点，空心为两个原站位；门位于 RX 一侧，具体门位待实测。",
                 fontsize=11, pad=10)
    ax.grid(alpha=.16)
    fig.savefig(OUTPUT/"人体测试点位.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
