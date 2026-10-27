"""KCS 초록 그림 1 — (a) 보이드 위치별 온도 상승, (b) 새 조건 160건 예측 대 실제

  python app/abstract_figure.py   →  figures/abstract_fig1.png (300 dpi), data/abstract_fig1.json

(a) 기준 조건(5 W, 1 mm 핫스팟, TIM 0.05 mm / 3 W/m·K)에서 보이드 중심을 핫스팟 중심에서 0~4 mm 옮기며
    실제 2D 해석(model.run_2d)으로 최고 온도 상승을 계산한다. 폭 0.5 / 1 / 2 mm.
(b) data/fresh_test.csv (학습에 쓰지 않은 160건)에서 v1·v2 예측과 실제 해석값을 비교한다.
"""

import json
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import model as M
import fresh_test as F

FIG = os.path.join(M.ROOT, "figures", "abstract_fig1.png")
OUT = os.path.join(M.ROOT, "data", "abstract_fig1.json")
OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
WIDTHS = [0.5, 1.0, 2.0]


def panel_a():
    base = M.run_2d(dict(M.DEFAULT, void_mm=0.0))["t_max"]
    out = {"t_no_void": base, "offsets": OFFSETS, "rise": {}}
    for w in WIDTHS:
        rise = []
        for off in OFFSETS:
            t = M.run_2d(dict(M.DEFAULT, void_mm=w, void_center_mm=M.SPOT_CENTER_MM + off))["t_max"]
            rise.append(t - base)
        out["rise"][str(w)] = rise
        print("  보이드 {:.1f} mm: ".format(w) + ", ".join("{:+.2f}".format(r) for r in rise), flush=True)
    return out


def panel_b():
    rows = F.load()
    actual = np.array([r["t_max"] for r in rows])
    conds = [M.snap(r) for r in rows]
    out = {"actual": actual.tolist(),
           "mae_1d": float(np.mean(np.abs(np.array([r["t_1d"] for r in rows]) - actual)))}
    for v in ("v1", "v2"):
        mean, half, _ = M.Surrogate(v).predict_many(conds, [r["t_1d"] for r in rows])
        out[v] = {"pred": mean.tolist(), "mae": float(np.mean(np.abs(mean - actual))),
                  "coverage": float(np.mean(np.abs(mean - actual) <= half))}
    print("  새 데이터 평균 오차: 1D만 {:.2f}℃, v1 {:.2f}℃, v2 {:.2f}℃".format(
        out["mae_1d"], out["v1"]["mae"], out["v2"]["mae"]))
    return out


def plot(a, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#6b6a66", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a7a1"
    shades = {"0.5": "#86b6ef", "1.0": "#3987e5", "2.0": "#0d366b"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7), dpi=300)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(length=0, labelcolor=INK2, labelsize=6.5)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    # (a) 위치별 상승
    offs = a["offsets"]
    ax1.axvspan(0, 0.5, color=ORANGE, alpha=0.12, lw=0)
    ax1.text(0.25, 15.6, "핫스팟", ha="center", fontsize=6, color=INK2)
    for w in ["2.0", "1.0", "0.5"]:
        ax1.plot(offs, a["rise"][w], "-o", color=shades[w], lw=1.2, ms=2.5, mec="white", mew=0.4, zorder=3,
                 label="보이드 폭 {:g} mm".format(float(w)))
    ax1.legend(frameon=False, fontsize=6, loc="upper right", labelcolor=INK2, handlelength=1.6)
    i2 = offs.index(2.0)
    ax1.annotate("+{:.1f}℃".format(a["rise"]["2.0"][0]), xy=(0, a["rise"]["2.0"][0]), xytext=(0.45, 14.3),
                 fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax1.annotate("+{:.1f}℃".format(a["rise"]["2.0"][i2]), xy=(2.0, a["rise"]["2.0"][i2]), xytext=(2.25, 3.0),
                 fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax1.set_xlim(-0.1, 4.1)
    ax1.set_ylim(-0.5, 16.5)
    ax1.set_xlabel("핫스팟 중심에서 보이드 중심까지 거리 (mm)", fontsize=6.8, color=INK2)
    ax1.set_ylabel("최고 온도 상승 (℃)", fontsize=6.8, color=INK2)
    ax1.text(-0.12, 1.04, "(a)", transform=ax1.transAxes, fontsize=8, fontweight="bold", color=INK)

    # (b) 예측 대 실제
    act = np.array(b["actual"])
    lim = (25, float(np.ceil(act.max() / 10) * 10))
    ax2.plot(lim, lim, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax2.scatter(act, b["v1"]["pred"], s=5, facecolor="none", edgecolor=GRAY, lw=0.5, zorder=2, label="v1 (겹침 정보 없음)")
    ax2.scatter(act, b["v2"]["pred"], s=5, color=BLUE, lw=0, zorder=3, label="v2 (겹침 정보 포함)")
    ax2.set_xlim(*lim)
    ax2.set_ylim(*lim)
    ax2.set_xlabel("실제 최고 온도, 2D 해석 (℃)", fontsize=6.8, color=INK2)
    ax2.set_ylabel("대리모델 예측 (℃)", fontsize=6.8, color=INK2)
    ax2.legend(frameon=False, fontsize=6, loc="upper left", labelcolor=INK2, handletextpad=0.2)
    ax2.text(0.97, 0.05, "평균 오차\nv1 {:.2f}℃\nv2 {:.2f}℃".format(b["v1"]["mae"], b["v2"]["mae"]),
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK, linespacing=1.4)
    ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes, fontsize=8, fontweight="bold", color=INK)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG, dpi=300, facecolor="white")
    print("저장:", FIG)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        with open(OUT, encoding="utf-8") as f:
            saved = json.load(f)
        plot(saved["a"], saved["b"])
        sys.exit(0)
    print("(a) 보이드 위치별 2D 해석 ({}회)".format(len(OFFSETS) * len(WIDTHS) + 1))
    a = panel_a()
    print("(b) 새 데이터 예측")
    b = panel_b()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"a": a, "b": b}, f, ensure_ascii=False)
    plot(a, b)
