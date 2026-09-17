"""1단계 파라미터 스윕 — TIM 두께 × 핫스팟 크기.

  python scripts/sweep_stage1.py   →  data/sweep_stage1.csv

손잡이 2개만 격자로 돌린다. 목적은 학습 데이터가 아니라 '눈으로 경향 보기'.
나머지 조건(발열 5W, TIM k=3, 방열판 1 K/W, 공기 25℃)은 전부 고정.

값을 고른 기준
  TIM 두께  : 격자 한 칸이 0.025mm 라서 그 배수만 쓸 수 있다.
              TIM 을 최소 2칸으로 쪼갠다는 원칙 때문에 0.05mm 부터 시작한다.
  핫스팟 크기: 10mm = 다이 전체 = 균일 발열. 1D 와 같은 상황이라 기준점이 된다.

기록하는 값
  최고 온도      고장을 결정하는 값
  다이 평균      발열 분포와 무관하게 같아야 하는 값 (검산에 쓴다)
  핫스팟 페널티  최고 − 다이 평균. 2D 에서만 나오는 값
"""

import csv
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import thermal_1d as m1
import thermal_2d as t2

TIM_MM = [0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250]
SPOT_MM = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "sweep_stage1.csv")

COLUMNS = ["tim_mm", "spot_mm", "t_max", "t_die_mean", "t_top_max", "penalty",
           "t_1d", "iters", "residual", "q_out", "seconds"]


def layers_for(tim_mm):
    """TIM 두께만 바꾼 층 구성. 1D 용과 2D 용을 같이 돌려준다."""
    t_tim = tim_mm * 1e-3
    cells = t_tim / t2.H_GRID
    if abs(cells - round(cells)) > 1e-9 or round(cells) < 2:
        raise ValueError("TIM {}mm 는 격자 0.025mm 의 2칸 이상 배수가 아니다".format(tim_mm))
    layers_2d = [(t2.T_SI, t2.K_SI), (t_tim, t2.K_TIM), (t2.T_CU, t2.K_CU)]
    layers_1d = [("다이 (Si)", t2.T_SI, t2.K_SI), ("TIM", t_tim, t2.K_TIM),
                 ("리드 (Cu)", t2.T_CU, t2.K_CU)]
    return layers_2d, layers_1d


def run_one(tim_mm, spot_mm):
    layers_2d, layers_1d = layers_for(tim_mm)
    k = t2.build_grid(layers=layers_2d)
    S = t2.source_hotspot(k, spot_mm=spot_mm)

    t0 = time.time()
    T, it, res = t2.solve(S, k)
    sec = time.time() - t0

    _, _, t_1d = m1.solve(layers=layers_1d)
    return {
        "tim_mm": tim_mm,
        "spot_mm": spot_mm,
        "t_max": float(T.max()),
        "t_die_mean": float(T[0].mean()),
        "t_top_max": float(T[-1].max()),
        "penalty": float(T.max() - T[0].mean()),
        "t_1d": float(t_1d),
        "iters": int(it),
        "residual": float(res),
        "q_out": t2.heat_out(T, k, t2.H_GRID),
        "seconds": sec,
    }


def verify(rows):
    """스윕 결과로 할 수 있는 검산 3가지."""
    print("\n검산\n")
    ok_all = True
    r_half = (t2.H_GRID / 2.0) / (t2.K_SI * t2.W_CROSS * t2.DEPTH)

    # 1) 모든 경우에 넣은 열 5W 가 다 나갔나
    worst_q = max(abs(r["q_out"] - t2.Q) for r in rows)
    ok = worst_q < 1e-4
    ok_all &= ok
    print("  [{}] 에너지 보존 ({}건 중 최대 오차 {:.1e} W)".format(
        "OK" if ok else "실패", len(rows), worst_q))

    # 2) 핫스팟 10mm(=균일 발열) 는 TIM 두께마다 1D 와 일치해야 한다
    worst_1d = max(abs(r["t_max"] - (r["t_1d"] - t2.Q * r_half))
                   for r in rows if r["spot_mm"] == 10.0)
    ok = worst_1d < 1e-4
    ok_all &= ok
    print("  [{}] 균일 발열 = 1D ({}가지 TIM 두께 중 최대 오차 {:.1e} ℃)".format(
        "OK" if ok else "실패", len(TIM_MM), worst_1d))

    # 3) 같은 TIM 두께면 핫스팟 크기와 상관없이 다이 평균이 같아야 한다
    worst_mean = 0.0
    for tim in TIM_MM:
        means = [r["t_die_mean"] for r in rows if r["tim_mm"] == tim]
        worst_mean = max(worst_mean, max(means) - min(means))
    ok = worst_mean < 1e-3
    ok_all &= ok
    print("  [{}] 발열 분포가 달라도 다이 평균은 같다 (최대 편차 {:.1e} ℃)".format(
        "OK" if ok else "실패", worst_mean))

    return ok_all


def main():
    total = len(TIM_MM) * len(SPOT_MM)
    print("스윕: TIM {}가지 × 핫스팟 {}가지 = {}회\n".format(len(TIM_MM), len(SPOT_MM), total))
    print("{:>4s} {:>8s} {:>8s} {:>9s} {:>9s} {:>7s} {:>6s}".format(
        "#", "TIM mm", "spot mm", "최고 ℃", "페널티 ℃", "반복", "초"))

    rows = []
    t_start = time.time()
    n = 0
    for tim in TIM_MM:
        for spot in SPOT_MM:
            n += 1
            r = run_one(tim, spot)
            rows.append(r)
            print("{:4d} {:8.3f} {:8.2f} {:9.2f} {:9.2f} {:7d} {:6.1f}".format(
                n, tim, spot, r["t_max"], r["penalty"], r["iters"], r["seconds"]))

    print("\n총 {:.0f}초".format(time.time() - t_start))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("저장:", OUT)

    if not verify(rows):
        sys.exit("검산 실패")


if __name__ == "__main__":
    main()
