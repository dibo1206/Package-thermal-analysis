"""2D 온도 분포 그림.

  python scripts/plot_2d.py   →  figures/2d_temperature.png

위: 발열을 다이 전체에 고르게 준 경우 (1D 로 계산할 수 있는 상황)
중: 같은 5W 를 다이 가운데 1mm 에만 준 경우 (1D 로는 불가능한 상황)
아래: 두 경우의 다이 바닥 온도를 가로 위치별로 비교
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

import thermal_2d as t2

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False      # 맑은 고딕에 U+2212 가 없다
plt.rcParams["figure.dpi"] = 130

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures", "2d_temperature.png")

# 층 경계 위치 [mm], 아래에서부터
Y_SI = t2.T_SI * 1e3
Y_TIM = (t2.T_SI + t2.T_TIM) * 1e3
Y_TOP = (t2.T_SI + t2.T_TIM + t2.T_CU) * 1e3


def draw_map(ax, T, title):
    nx = T.shape[1]
    extent = [0.0, nx * t2.H_GRID * 1e3, 0.0, Y_TOP]
    im = ax.imshow(T, origin="lower", extent=extent, aspect="auto", cmap="inferno")

    for y in (Y_SI, Y_TIM):
        ax.axhline(y, color="white", lw=0.8, ls="--", alpha=0.65)
    for y, name in ((Y_SI / 2, "Si"), ((Y_SI + Y_TIM) / 2, "TIM"),
                    ((Y_TIM + Y_TOP) / 2, "Cu")):
        ax.text(10.15, y, name, va="center", ha="left", fontsize=8, color="0.3")

    ax.set_title("{}   최고 {:.1f}℃".format(title, T.max()), fontsize=10)
    ax.set_ylabel("두께 [mm]")
    plt.colorbar(im, ax=ax, pad=0.06, label="온도 [℃]")


def main():
    k = t2.build_grid()
    T_uni, _, _ = t2.solve(t2.source_uniform(k), k)
    T_hot, it, res = t2.solve(t2.source_hotspot(k), k)
    print("핫스팟 {}회 반복, 상대 잔차 {:.1e}".format(it, res))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0),
                             gridspec_kw={"height_ratios": [1, 1, 1.1]})

    draw_map(axes[0], T_uni, "발열 5W 를 다이 전체에 고르게")
    draw_map(axes[1], T_hot, "같은 5W 를 가운데 1mm 에 집중")

    x = (np.arange(k.shape[1]) + 0.5) * t2.H_GRID * 1e3
    ax = axes[2]
    ax.plot(x, T_uni[0], label="균일 발열", lw=1.8)
    ax.plot(x, T_hot[0], label="핫스팟", lw=1.8)
    ax.plot(x, T_hot[-1], label="핫스팟, 패키지 윗면", lw=1.2, ls="--", color="0.45")
    _, r_total = t2.expected_1d()
    ax.axhline(t2.T_AIR + t2.Q * r_total, color="crimson", lw=1.0, ls=":",
               label="1D 모델 예측 {:.1f}℃".format(t2.T_AIR + t2.Q * r_total))
    ax.set_xlabel("가로 위치 [mm]")
    ax.set_ylabel("온도 [℃]")
    ax.set_title("다이 바닥 온도 — 같은 발열량, 다른 분포", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 10)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
