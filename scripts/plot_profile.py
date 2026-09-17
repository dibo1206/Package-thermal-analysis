"""세로 방향 온도 프로파일 — 두께를 따라 온도가 어디서 떨어지는가.

  python scripts/plot_profile.py   →  figures/2d_profile.png

왼쪽 : 균일 발열. 칩 바닥에서 패키지 윗면까지 온도가 내려가는 모양.
가운데: 핫스팟. 발열 바로 위(중앙)와 멀리 떨어진 곳(가장자리)을 비교.
오른쪽: 층별 온도 강하 막대. 방열판까지 포함해서 어디서 떨어지는지 한눈에.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import thermal_1d as m1
import thermal_2d as t2

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures", "2d_profile.png")

Y_SI = t2.T_SI * 1e3
Y_TIM = (t2.T_SI + t2.T_TIM) * 1e3
Y_TOP = (t2.T_SI + t2.T_TIM + t2.T_CU) * 1e3

BANDS = [(0.0, Y_SI, "실리콘", "#f2c14e"), (Y_SI, Y_TIM, "TIM", "#e8705f"),
         (Y_TIM, Y_TOP, "구리", "#7fa9c9")]


def draw_bands(ax):
    for lo, hi, name, color in BANDS:
        ax.axhspan(lo, hi, color=color, alpha=0.16, lw=0)
        ax.axhline(hi, color="0.55", lw=0.6, ls="--")
    ax.set_ylim(0, Y_TOP)
    ax.set_ylabel("패키지 바닥에서의 높이 [mm]")


def label_bands(ax):
    x = ax.get_xlim()[1]
    for lo, hi, name, _ in BANDS:
        ax.text(x, (lo + hi) / 2, " " + name, va="center", ha="left",
                fontsize=8, color="0.25", clip_on=False)


def main():
    k = t2.build_grid()
    T_uni, _, _ = t2.solve(t2.source_uniform(k), k)
    T_hot, _, _ = t2.solve(t2.source_hotspot(k), k)

    y = (np.arange(k.shape[0]) + 0.5) * t2.H_GRID * 1e3
    center = k.shape[1] // 2
    edge = int(round(0.5e-3 / t2.H_GRID))        # 가장자리에서 0.5mm 지점

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6))

    # ── 왼쪽: 균일 발열 ─────────────────────────────────────
    ax = axes[0]
    draw_bands(ax)
    ax.plot(T_uni[:, center], y, color="#1f77b4", lw=2.2, marker="o", ms=2.5)
    ax.set_xlabel("온도 [℃]")
    ax.set_title("균일 발열 — 어디서 떨어지나", fontsize=10)
    ax.grid(alpha=0.25)

    # TIM '층 전체' 의 온도 강하를 화살표로 표시.
    # 격자점 두 개의 차이가 아니라 층의 아랫면~윗면 차이여야 한다.
    # 계면 온도는 1D 저항으로 위에서부터 쌓아 내려오면 정확히 나온다.
    parts, _, _ = m1.solve()
    t_surface = t2.T_AIR + t2.Q * parts[3][1]       # 패키지 윗면
    t_tim_top = t_surface + t2.Q * parts[2][1]      # TIM 윗면 (= 구리 아랫면)
    t_tim_bot = t_tim_top + t2.Q * parts[1][1]      # TIM 아랫면 (= 실리콘 윗면)
    d_tim = t2.Q * parts[1][1]
    share = d_tim / (t2.Q * sum(p[1] for p in parts[:-1])) * 100

    ax.annotate("", xy=(t_tim_top, Y_TIM), xytext=(t_tim_bot, Y_SI),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.6))
    ax.text(30.04, 0.30,
            "TIM 0.05mm 에서 {:.2f}℃ 강하\n패키지 내부 강하의 {:.0f}%".format(d_tim, share),
            fontsize=8.5, color="#c0392b", va="bottom",
            bbox=dict(fc="white", ec="#c0392b", alpha=0.9, lw=0.6))
    label_bands(ax)

    # ── 가운데: 핫스팟 ──────────────────────────────────────
    ax = axes[1]
    draw_bands(ax)
    ax.plot(T_hot[:, center], y, color="#e67e22", lw=2.2,
            label="발열 바로 위 (5mm)")
    ax.plot(T_hot[:, edge], y, color="0.35", lw=1.8, ls="--",
            label="멀리 떨어진 곳 (0.5mm)")
    ax.set_xlabel("온도 [℃]")
    ax.set_title("핫스팟 — 위로 갈수록 차이가 사라진다", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    lo = min(float(T_hot[:, edge].min()), float(T_hot[:, center].min()))
    hi = float(T_hot[:, center].max())
    ax.set_xlim(lo - 0.6, hi + 0.6)                # 가장자리 선이 축에 붙지 않게

    gap_lo = float(T_hot[0, center] - T_hot[0, edge])
    gap_hi = float(T_hot[-1, center] - T_hot[-1, edge])
    ax.text(0.30, 0.62, "바닥 차이 {:.1f}℃\n윗면 차이 {:.1f}℃".format(gap_lo, gap_hi),
            transform=ax.transAxes, fontsize=9, color="#7f3f00",
            bbox=dict(fc="white", ec="0.8", alpha=0.9))
    label_bands(ax)

    # ── 오른쪽: 층별 온도 강하 ──────────────────────────────
    ax = axes[2]
    parts, r_total, _ = m1.solve()
    names = [p[0] for p in parts]
    drops = [t2.Q * p[1] for p in parts]
    colors = ["#f2c14e", "#e8705f", "#7fa9c9", "#9b59b6"]
    bars = ax.barh(names, drops, color=colors, edgecolor="0.4", lw=0.6)
    for b, d in zip(bars, drops):
        ax.text(d + 0.08, b.get_y() + b.get_height() / 2,
                "{:.2f}℃".format(d), va="center", fontsize=9)
    ax.set_xlabel("온도 강하 [℃]")
    ax.set_xlim(0, max(drops) * 1.25)
    ax.set_title("어디서 온도가 떨어지나 (총 {:.1f}℃)".format(sum(drops)), fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.97, 0.96,
            "방열판이 전체의 {:.0f}%\n패키지 안에서만 보면 TIM 이 {:.0f}%".format(
                drops[-1] / sum(drops) * 100,
                drops[1] / sum(drops[:-1]) * 100),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.25",
            bbox=dict(fc="white", ec="0.8", alpha=0.9))

    fig.suptitle("두께 방향 온도 프로파일 — 칩에서 공기까지", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, bbox_inches="tight")
    print("저장:", OUT)
    print("균일: 바닥 {:.3f}℃ → 윗면 {:.3f}℃".format(
        float(T_uni[0, center]), float(T_uni[-1, center])))
    print("핫스팟 중앙: 바닥 {:.3f}℃ → 윗면 {:.3f}℃".format(
        float(T_hot[0, center]), float(T_hot[-1, center])))


if __name__ == "__main__":
    main()
