"""학습 곡선 — 데이터를 줄여도 버티나? 1D 를 넣으면 몇 개까지 줄일 수 있나?

  python scripts/learning_curve.py   →  data/learning_curve.json, figures/learning_curve.png

이 프로젝트의 핵심 질문:
  "1D 공식으로 계산할 수 있는 부분을 분리하면, 같은 정확도에 필요한 시뮬레이션이 몇 개로 줄어드는가?"

방법
  학습 데이터를 25, 50, 100, 200, 400, 640 개로 늘려가며 두 방법(AI 혼자 / 1D + AI)의
  검증셋 평균 오차를 잰다. 같은 크기에서 무작위로 5번 뽑아 평균과 범위를 함께 본다.

미리 정해둔 목표선 (결과 보기 전에 정함)
  평균 오차 2.0℃ 와 1.5℃. 각 방법이 이 선에 도달하는 데 필요한 데이터 수를 읽는다.

솔직히 기록할 두 가지
  1. surrogate_void.py 1차 실행에서 고른 설정값이 후보 범위의 끝이었다. 그래서 범위를 넓혔다.
     (그 사실은 검증셋이 아니라 교차검증 결과를 보고 알 수 있었다. 다만 1차 채점 결과를 이미 본 뒤에 바꿨다.)
  2. 학습 곡선을 그리려면 검증셋을 크기마다 여러 번 채점하게 된다.
     설정을 고르는 데는 학습 데이터 안의 교차검증만 쓰고, 검증셋은 점수를 읽는 데만 쓴다.
"""

import json
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import surrogate_void as sv

sv.ELLS = [0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0]      # 범위를 넓혔다

SIZES = [25, 50, 100, 200, 400, 640]
REPEATS = 5
SEED = 42
TARGETS = [2.0, 1.5]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(ROOT, "data", "learning_curve.json")
OUT_FIG = os.path.join(ROOT, "figures", "learning_curve.png")


def mae(pred, rows):
    return float(np.mean(np.abs(pred - np.array([r["t_max"] for r in rows]))))


def run():
    rows = sv.load()
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    baseline = mae(np.array([r["t_1d"] for r in test]), test)
    print("학습 {}개 / 검증 {}개, 1D 공식만 쓸 때 평균 오차 {:.2f}℃\n".format(
        len(train), len(test), baseline))

    results = {name: {} for name in sv.METHODS}
    t_start = time.time()
    for size in SIZES:
        reps = 1 if size >= len(train) else REPEATS
        rng = np.random.default_rng(SEED + size)
        subsets = [[train[i] for i in rng.choice(len(train), size=size, replace=False)]
                   for _ in range(reps)]
        for name, method in sv.METHODS.items():
            scores = []
            for sub in subsets:
                _, ell, noise = sv.tune(method, sub)
                scores.append(mae(method(sub, test, ell, noise), test))
            results[name][size] = {"mean": float(np.mean(scores)),
                                   "min": float(np.min(scores)),
                                   "max": float(np.max(scores)),
                                   "runs": [float(s) for s in scores]}
            print("  {:8s} 데이터 {:4d}개 → 평균 오차 {:.3f}℃ (범위 {:.3f}~{:.3f}, {}회)  경과 {:.0f}초".format(
                name, size, results[name][size]["mean"], results[name][size]["min"],
                results[name][size]["max"], reps, time.time() - t_start), flush=True)
    return results, baseline, len(test)


def needed(results, target):
    """목표 평균 오차에 처음 도달하는 데이터 수. 도달 못 하면 None."""
    out = {}
    for name, curve in results.items():
        hit = [s for s in SIZES if curve[s]["mean"] <= target]
        out[name] = min(hit) if hit else None
    return out


def report(results, baseline):
    print("\n목표선까지 필요한 데이터 수\n")
    for target in TARGETS:
        need = needed(results, target)
        parts = []
        for name in sv.METHODS:
            n = need[name]
            parts.append("{} {}".format(name, "{}개".format(n) if n else "도달 못 함"))
        line = "  평균 오차 {:.1f}℃ 이내 : ".format(target) + " / ".join(parts)
        a, b = need.get("AI 혼자"), need.get("1D + AI")
        if a and b and b > 0:
            line += "   → {:.1f}배 차이".format(a / b)
        print(line)
    print("\n  (참고) 1D 공식만 쓰면 {:.2f}℃".format(baseline))


def plot(results, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = {"AI 혼자": "#2a78d6", "1D + AI": "#eb6834"}

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0, labelcolor=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.8, which="both")
    ax.set_axisbelow(True)

    for name, curve in results.items():
        xs = SIZES
        ys = [curve[s]["mean"] for s in xs]
        lo = [curve[s]["min"] for s in xs]
        hi = [curve[s]["max"] for s in xs]
        ax.fill_between(xs, lo, hi, color=colors[name], alpha=0.15, lw=0)
        ax.plot(xs, ys, color=colors[name], lw=2, marker="o", ms=6,
                mec=SURFACE, mew=1.5, label=name, zorder=3)

    for target in TARGETS:
        ax.axhline(target, color=MUTED, lw=1, ls="--", zorder=2)
        ax.text(25, target * 1.03, "목표 {:.1f}℃".format(target), fontsize=8.5, color=INK2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_xticks(SIZES)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # 로그 세로축은 기본 눈금이 1, 10 뿐이라 숫자가 거의 안 보인다. 직접 지정한다.
    ax.set_yticks([1, 1.5, 2, 3, 5, 7])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("학습에 쓴 시뮬레이션 수", color=INK2, fontsize=9.5)
    ax.set_ylabel("검증셋 평균 오차 [℃]", color=INK2, fontsize=9.5)
    ax.set_title("데이터를 줄이면 어디서 무너지나", loc="left", fontsize=11, color=INK, pad=22)
    ax.text(0, 1.02, "검증용 160개로 채점 · 같은 크기에서 5번 뽑은 평균, 색칠한 띠는 최소~최대 · 1D 공식만 쓰면 {:.1f}℃".format(baseline),
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight", facecolor=SURFACE)
    print("저장:", OUT_FIG)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:          # 계산은 13분 걸린다. 그림만 다시 그릴 때
        with open(OUT_JSON, encoding="utf-8") as f:
            saved = json.load(f)
        results = {name: {int(k): v for k, v in curve.items()}
                   for name, curve in saved["results"].items()}
        report(results, saved["baseline_1d_mae"])
        plot(results, saved["baseline_1d_mae"])
        sys.exit(0)

    results, baseline, n_test = run()
    report(results, baseline)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sizes": SIZES, "repeats": REPEATS, "targets": TARGETS,
                   "baseline_1d_mae": baseline, "n_test": n_test,
                   "ells": sv.ELLS, "noises": sv.NOISES, "results": results},
                  f, ensure_ascii=False, indent=2)
    print("저장:", OUT_JSON)
    plot(results, baseline)
