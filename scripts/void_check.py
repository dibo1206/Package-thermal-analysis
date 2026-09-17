"""TIM 보이드가 칩 온도를 얼마나 올리나 — 크기만 먼저 재보는 중간 점검.

  python scripts/void_check.py   →  data/void_check.csv, figures/void_check.png

보이드 = 리드를 덮어 누를 때 TIM 안에 갇힌 공기 방울 (공정 불량).
모델에서는 TIM 층의 칸 몇 개를 공기(열전도율 0.026 W/m·K)로 바꾼다.

이 값이 크면 본격적인 연구(A-1)로 가고, 작으면 여기서 정리한다.

온도는 표면 기준으로 읽는다 (외부 검증 v2 에서 고친 방식, docs/validation-log.md).
"""

import csv
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import thermal_2d as t2

K_AIR = 0.026                 # 공기 열전도율 [W/m·K]
H = t2.H_GRID                 # 격자 한 칸 0.025mm
SPOT_MM = 1.0                 # 핫스팟 폭
SPOT_CENTER_MM = 5.0          # 핫스팟은 가운데
LAYERS = [(t2.T_SI, t2.K_SI), (t2.T_TIM, t2.K_TIM), (t2.T_CU, t2.K_CU)]

VOID_WIDTHS_MM = [0.5, 1.0, 2.0]
VOID_CENTERS_MM = [5.0, 3.0, 1.0]      # 핫스팟 바로 위 / 2mm 옆 / 4mm 옆

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_OUT = os.path.join(ROOT, "data", "void_check.csv")
FIG_OUT = os.path.join(ROOT, "figures", "void_check.png")


def build_with_void(void_mm, center_mm, h=H):
    """TIM 층의 일부 칸을 공기로 바꾼 격자."""
    k = t2.build_grid(h=h, layers=LAYERS).copy()
    if void_mm <= 0:
        return k
    n_si = int(round(t2.T_SI / h))
    n_tim = int(round(t2.T_TIM / h))
    nx = k.shape[1]
    w = max(1, int(round(void_mm * 1e-3 / h)))
    lo = int(round((center_mm - void_mm / 2.0) * 1e-3 / h))
    lo = max(0, min(lo, nx - w))
    k[n_si:n_si + n_tim, lo:lo + w] = K_AIR
    return k


def run(void_mm, center_mm):
    k = build_with_void(void_mm, center_mm)
    S = t2.source_hotspot(k, q=t2.Q, spot_mm=SPOT_MM, h=H)
    T, it, res = t2.solve(S, k, h=H)
    surface = t2.surface_temperature(T, S, k)
    return {
        "void_mm": void_mm,
        "center_mm": center_mm,
        "t_max": float(surface.max()),
        "t_die_mean": float(surface.mean()),
        "t_top_max": float(T[-1].max()),
        "iters": int(it),
        "residual": float(res),
        "q_out": t2.heat_out(T, k, H),
    }


def main():
    print("기준 조건: 발열 {:.0f}W, 핫스팟 {:g}mm, TIM {:g}mm (k={:g}), 방열판 {:g} K/W\n".format(
        t2.Q, SPOT_MM, t2.T_TIM * 1e3, t2.K_TIM, t2.R_SINK))

    base = run(0.0, 0.0)
    print("보이드 없음: 칩 최고 온도 {:.2f}℃\n".format(base["t_max"]))

    rows = [base]
    print("{:>10s} {:>12s} {:>10s} {:>9s}".format("보이드 폭", "보이드 위치", "최고 온도", "상승"))
    for center in VOID_CENTERS_MM:
        where = "핫스팟 바로 위" if center == SPOT_CENTER_MM else "{:.0f}mm 옆".format(
            abs(center - SPOT_CENTER_MM))
        for width in VOID_WIDTHS_MM:
            r = run(width, center)
            r["rise"] = r["t_max"] - base["t_max"]
            rows.append(r)
            print("{:9.1f}mm {:>12s} {:9.2f}℃ {:8.2f}℃".format(
                width, where, r["t_max"], r["rise"]))
        print()

    # 검산
    print("검산")
    worst_q = max(abs(r["q_out"] - t2.Q) for r in rows)
    print("  [{}] 에너지 보존 (최대 오차 {:.1e} W)".format("OK" if worst_q < 1e-4 else "실패", worst_q))
    conv = all(r["residual"] < 1e-8 for r in rows)
    print("  [{}] 전부 수렴".format("OK" if conv else "실패"))
    near = [r for r in rows if r.get("rise") is not None and r["center_mm"] == 5.0]
    far = [r for r in rows if r.get("rise") is not None and r["center_mm"] == 1.0]
    ok_far = all(f["rise"] < n["rise"] for n, f in zip(near, far))
    print("  [{}] 멀리 있는 보이드가 덜 뜨겁다".format("OK" if ok_far else "실패"))

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        cols = ["void_mm", "center_mm", "t_max", "t_die_mean", "t_top_max",
                "rise", "iters", "residual", "q_out"]
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({c: r.get(c, "") for c in cols})
    print("\n저장:", CSV_OUT)
    plot(rows, base)


def plot(rows, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    x = np.arange(len(VOID_WIDTHS_MM))
    bw = 0.26
    for i, center in enumerate(VOID_CENTERS_MM):
        vals = [next(r["rise"] for r in rows
                     if r.get("rise") is not None
                     and r["center_mm"] == center and r["void_mm"] == v)
                for v in VOID_WIDTHS_MM]
        label = "핫스팟 바로 위" if center == SPOT_CENTER_MM else "{:.0f}mm 옆".format(
            abs(center - SPOT_CENTER_MM))
        ax.bar(x + (i - 1) * bw, vals, width=bw, color=colors[i], label=label, zorder=3)
        for xi, v in zip(x + (i - 1) * bw, vals):
            ax.text(xi, v + 0.06, "{:.1f}".format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels(["{:g}mm".format(v) for v in VOID_WIDTHS_MM])
    ax.set_xlabel("보이드 폭", color=INK2, fontsize=9.5)
    ax.set_ylabel("칩 최고 온도 상승 [℃]", color=INK2, fontsize=9.5)
    ax.set_title("TIM 보이드가 칩 온도를 얼마나 올리나", loc="left", fontsize=11, color=INK, pad=22)
    ax.text(0, 1.02, "보이드 없을 때 {:.1f}℃ 기준 · 발열 5W · 핫스팟 1mm".format(base["t_max"]),
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)

    fig.tight_layout()
    fig.savefig(FIG_OUT, bbox_inches="tight", facecolor=SURFACE)
    print("저장:", FIG_OUT)


if __name__ == "__main__":
    main()
