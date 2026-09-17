"""1단계 스윕 결과 그림.

  python scripts/plot_sweep_stage1.py   →  figures/sweep_stage1.png

왼쪽 : TIM 두께에 따른 칩 최고 온도. 핫스팟 크기별 선 + 균일 발열(= 1D) 기준선.
오른쪽: TIM 을 0.05 → 0.25mm 로 바꿨을 때 최고 온도가 얼마나 오르는가, 핫스팟 크기별.

숫자 전체는 data/sweep_stage1.csv 에 있다 (그림의 표 버전).
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "sweep_stage1.csv")
OUT = os.path.join(ROOT, "figures", "sweep_stage1.png")

# 색 — dataviz 기본 팔레트
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
REF_GRAY = "#898781"                 # 균일 발열 = 1D 기준. 강조하지 않는 색
# 핫스팟 크기는 순서가 있는 값이라 파랑 한 가지를 밝음→진함으로 쓴다 (작을수록 진하게)
SPOT_COLOR = {5.0: "#86b6ef", 1.0: "#2a78d6", 0.25: "#104281"}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def load():
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in r:
            r[key] = float(r[key])
    return rows


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=0, labelcolor=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.8, ls="-")
    ax.set_axisbelow(True)


def series(rows, spot):
    pts = sorted((r["tim_mm"], r["t_max"]) for r in rows if r["spot_mm"] == spot)
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    rows = load()
    tims = sorted({r["tim_mm"] for r in rows})
    spots = sorted({r["spot_mm"] for r in rows}, reverse=True)   # 10 → 0.25
    t_thin, t_thick = tims[0], tims[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)

    # ── 왼쪽: 최고 온도 vs TIM 두께 ─────────────────────────
    style(ax1)
    shown = [10.0, 5.0, 1.0, 0.25]
    handles = []
    for spot in shown:
        xs, ys = series(rows, spot)
        color = REF_GRAY if spot == 10.0 else SPOT_COLOR[spot]
        label = "균일 발열 (= 1D)" if spot == 10.0 else "핫스팟 {:g}mm".format(spot)
        ax1.plot(xs, ys, color=color, lw=2.0, solid_capstyle="round",
                 marker="o", ms=5.5, mec=SURFACE, mew=1.5, zorder=3)
        # 끝점 직접 라벨 — 글자는 잉크색, 옆의 선이 정체를 알려준다
        ax1.text(xs[-1] + 0.006, ys[-1], "{:.1f}℃".format(ys[-1]),
                 va="center", ha="left", fontsize=9, color=INK2)
        handles.append(Line2D([0], [0], color=color, lw=2.0, marker="o", ms=5.5,
                              mec=SURFACE, mew=1.5, label=label))

    ax1.set_xlim(t_thin - 0.01, t_thick + 0.04)
    ax1.set_xlabel("TIM 두께 [mm]", color=INK2, fontsize=9.5)
    ax1.set_ylabel("칩 최고 온도 [℃]", color=INK2, fontsize=9.5)
    ax1.set_title("TIM이 두꺼워질수록 핫스팟은 더 가파르게 뜨거워진다",
                  loc="left", fontsize=11, color=INK, pad=22)
    ax1.text(0, 1.02, "발열 5W · TIM 열전도율 3 W/m·K · 방열판 1 K/W 고정",
             transform=ax1.transAxes, fontsize=8.5, color=MUTED)
    ax1.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5,
               labelcolor=INK2)

    # ── 오른쪽: TIM 0.05 → 0.25mm 에 따른 최고 온도 상승 ────
    style(ax2)
    rise = []
    for spot in spots:
        _, ys = series(rows, spot)
        rise.append(ys[-1] - ys[0])

    x = list(range(len(spots)))
    colors = [REF_GRAY if s == 10.0 else SERIES_1 for s in spots]
    ax2.bar(x, rise, width=0.26, color=colors, zorder=3)
    for xi, v in zip(x, rise):
        ax2.text(xi, v + 0.18, "+{:.1f}".format(v), ha="center", va="bottom",
                 fontsize=9, color=INK2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["{:g}".format(s) for s in spots])
    ax2.set_xlabel("핫스팟 크기 [mm]   (10mm = 다이 전체 = 균일 발열)",
                   color=INK2, fontsize=9.5)
    ax2.set_ylabel("최고 온도 상승 [℃]", color=INK2, fontsize=9.5)
    ax2.set_ylim(0, max(rise) * 1.22)
    ratio = rise[-1] / rise[0]
    ax2.set_title("같은 TIM 변경이 핫스팟에서는 {:.1f}배 크게 나타난다".format(ratio),
                  loc="left", fontsize=11, color=INK, pad=22)
    ax2.text(0, 1.02, "TIM 두께 {:g} → {:g}mm 로 바꿨을 때".format(t_thin, t_thick),
             transform=ax2.transAxes, fontsize=8.5, color=MUTED)
    ax2.legend(handles=[Patch(color=SERIES_1, label="핫스팟"),
                        Patch(color=REF_GRAY, label="균일 발열 (= 1D 가 보는 값)")],
               loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)

    fig.tight_layout(w_pad=3.0)
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print("저장:", OUT)

    print("\nTIM {:g} → {:g}mm 일 때 최고 온도 상승".format(t_thin, t_thick))
    for s, v in zip(spots, rise):
        print("  핫스팟 {:>5g}mm : +{:.2f}℃  (균일 대비 {:.2f}배)".format(s, v, v / rise[0]))


if __name__ == "__main__":
    main()
