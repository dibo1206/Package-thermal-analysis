"""대리모델이 어떤 조건에서 크게 틀리나 — 보완 분석.

  python scripts/error_analysis.py   →  figures/error_analysis.png

surrogate_void.py 에서 고른 설정을 그대로 써서 검증 160건을 다시 예측하고,
오차가 큰 조건이 어떤 특징을 갖는지 본다. 초록의 한계 문장을 근거 있게 쓰기 위한 분석.
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

import surrogate_void as sv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures", "error_analysis.png")
SETTINGS = os.path.join(ROOT, "data", "surrogate_void.json")


def gap(r):
    """보이드 가장자리와 핫스팟 가장자리 사이 거리 [mm]. 0 이면 핫스팟 위를 덮는다."""
    if r["void_mm"] <= 0:
        return np.inf
    d = abs(r["void_center_mm"] - sv.SPOT_CENTER_MM)
    return max(0.0, d - (r["void_mm"] + r["spot_mm"]) / 2.0)


def main():
    rows = sv.load()
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    with open(SETTINGS, encoding="utf-8") as f:
        chosen = json.load(f)["chosen"]

    name = "1D + AI"
    ell, noise = chosen[name]["ell"], chosen[name]["noise"]
    pred = sv.method_hybrid(train, test, ell, noise)
    actual = np.array([r["t_max"] for r in test])
    err = pred - actual
    ae = np.abs(err)

    print("{} (거리 {}, 잡음 {:g}) — 검증 {}건\n".format(name, ell, noise, len(test)))
    print("평균 오차 {:.2f}℃, 최대 {:.2f}℃\n".format(ae.mean(), ae.max()))

    order = np.argsort(-ae)[:10]
    print("오차가 큰 10건")
    print("  {:>7s} {:>7s} {:>8s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
        "실제℃", "예측℃", "오차℃", "발열W", "핫스팟", "보이드", "간격mm", "TIMmm", "TIM k"))
    for i in order:
        r = test[i]
        g = gap(r)
        print("  {:7.1f} {:7.1f} {:+8.1f} {:7.1f} {:7.2f} {:7.2f} {:>7s} {:7.3f} {:7.1f}".format(
            actual[i], pred[i], err[i], r["q_w"], r["spot_mm"], r["void_mm"],
            "없음" if np.isinf(g) else "{:.2f}".format(g), r["tim_mm"], r["tim_k"]))

    def group(label, mask):
        if mask.sum() == 0:
            return
        print("  {:22s} {:3d}건  평균 {:5.2f}℃  최대 {:6.2f}℃".format(
            label, int(mask.sum()), ae[mask].mean(), ae[mask].max()))

    gaps = np.array([gap(r) for r in test])
    voids = np.array([r["void_mm"] for r in test])
    print("\n보이드 위치별")
    group("보이드 없음", voids == 0)
    group("핫스팟을 덮음 (간격 0)", (voids > 0) & (gaps == 0))
    group("가까움 (0~1mm)", (voids > 0) & (gaps > 0) & (gaps <= 1))
    group("멀리 (1mm 초과)", (voids > 0) & (gaps > 1))

    print("\n실제 최고 온도 구간별")
    for lo, hi in [(0, 50), (50, 80), (80, 120), (120, 1e9)]:
        m = (actual >= lo) & (actual < hi)
        group("{:.0f} ~ {:.0f}℃".format(lo, min(hi, actual.max())), m)

    print("\n상대 오차로 보면 평균 {:.2f}%, 최대 {:.2f}%".format(
        float(np.mean(ae / (actual - 25.0)) * 100), float(np.max(ae / (actual - 25.0)) * 100)))

    plot(actual, ae, gaps, voids)


def plot(actual, ae, gaps, voids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(length=0, labelcolor=INK2, labelsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel("예측 오차 [℃]", color=INK2, fontsize=9.5)

    covers = (voids > 0) & (gaps == 0)
    ax1.scatter(actual[~covers], ae[~covers], s=20, color="#898781", edgecolor=SURFACE, lw=0.5,
                label="그 외", zorder=3)
    ax1.scatter(actual[covers], ae[covers], s=26, color="#eb6834", edgecolor=SURFACE, lw=0.5,
                label="보이드가 핫스팟을 덮음", zorder=4)
    ax1.set_xlabel("실제 최고 온도 [℃]", color=INK2, fontsize=9.5)
    ax1.set_title("뜨거운 조건일수록 오차가 크다", loc="left", fontsize=10.5, color=INK)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK2)

    finite = np.isfinite(gaps) & (voids > 0)
    ax2.scatter(gaps[finite], ae[finite], s=22, color="#2a78d6", edgecolor=SURFACE, lw=0.5, zorder=3)
    ax2.set_xlabel("보이드와 핫스팟 사이 간격 [mm]  (0 = 덮음)", color=INK2, fontsize=9.5)
    ax2.set_title("보이드가 핫스팟을 덮을 때 오차가 몰린다", loc="left", fontsize=10.5, color=INK)

    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight", facecolor=SURFACE)
    print("\n저장:", FIG)


if __name__ == "__main__":
    main()
