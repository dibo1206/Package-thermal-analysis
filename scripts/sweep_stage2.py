"""2단계 파라미터 스윕 — 손잡이 5개를 무작위로 800개. 대리모델 학습 데이터.

  python scripts/sweep_stage2.py --check   검산만 (약 1분)
  python scripts/sweep_stage2.py           검산 → 전체 실행. 중간에 끊겨도 다시 돌리면 이어서 한다

  → data/sweep_stage2.csv

손잡이와 범위
  TIM 두께      0.05 ~ 0.25mm    격자 때문에 0.025mm 배수만 가능 (2~10칸)
  TIM 열전도율  1 ~ 10 W/m·K     로그 균등
  핫스팟 크기   0.25 ~ 10mm      로그 균등, 가운데 정렬을 위해 짝수 칸으로 반올림
  총 발열       1 ~ 15 W         균등
  방열판 저항   0.3 ~ 3 K/W      로그 균등

로그 균등이란: 1→2 와 5→10 을 같은 '2배 변화' 로 보고 고르게 뽑는 것.
그냥 균등하게 뽑으면 1~10 중 절반 이상이 5 위쪽에 몰려서 작은 값 쪽이 부실해진다.

검증셋: 800개를 뽑는 그 순간에 20% 를 test 로 정해서 CSV 에 적어둔다.
결과를 보기 전에 정한다 — 특허 프로젝트에서 사람 라벨을 먼저 백업한 것과 같은 원칙.
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

N_SAMPLES = 800
SEED = 42
TEST_FRACTION = 0.2

TIM_CELLS = (2, 10)
TIM_K = (1.0, 10.0)
SPOT_MM = (0.25, 10.0)
Q_W = (1.0, 15.0)
R_SINK = (0.3, 3.0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "sweep_stage2.csv")

INPUTS = ["idx", "split", "tim_mm", "tim_k", "spot_mm", "q_w", "r_sink"]
OUTPUTS = ["t_max", "t_die_mean", "t_top_max", "penalty", "t_1d",
           "iters", "residual", "q_out", "seconds"]
COLUMNS = INPUTS + OUTPUTS


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample(n=N_SAMPLES, seed=SEED):
    """조건 n개를 뽑고, 그중 20% 를 검증셋으로 고정한다."""
    rng = np.random.default_rng(seed)
    nx = int(round(t2.W_CROSS / t2.H_GRID))
    rows = []
    for i in range(n):
        tim_cells = int(rng.integers(TIM_CELLS[0], TIM_CELLS[1] + 1))
        tim_k = log_uniform(rng, *TIM_K)
        spot = log_uniform(rng, *SPOT_MM)
        w = int(round(spot * 1e-3 / t2.H_GRID / 2.0)) * 2
        w = min(max(w, 2), nx)
        q = float(rng.uniform(*Q_W))
        r_sink = log_uniform(rng, *R_SINK)
        rows.append({
            "idx": i,
            "tim_mm": tim_cells * t2.H_GRID * 1e3,
            "tim_k": tim_k,
            "spot_mm": w * t2.H_GRID * 1e3,
            "q_w": q,
            "r_sink": r_sink,
        })
    test = set(rng.choice(n, size=int(n * TEST_FRACTION), replace=False).tolist())
    for r in rows:
        r["split"] = "test" if r["idx"] in test else "train"
    return rows


def model_for(p):
    """조건 하나에 대한 격자, 발열, 방열판 계수, 1D 층 구성."""
    t_tim = p["tim_mm"] * 1e-3
    k = t2.build_grid(layers=[(t2.T_SI, t2.K_SI), (t_tim, p["tim_k"]), (t2.T_CU, t2.K_CU)])
    h_sink = 1.0 / (p["r_sink"] * t2.W_CROSS * t2.DEPTH)
    layers_1d = [("다이 (Si)", t2.T_SI, t2.K_SI), ("TIM", t_tim, p["tim_k"]),
                 ("리드 (Cu)", t2.T_CU, t2.K_CU)]
    return k, h_sink, layers_1d


def t1d_cell(p, layers_1d):
    """1D 칩 온도에서 맨 아래 반 칸을 뺀 값 = 2D 균일 발열의 다이 바닥 온도."""
    _, _, t_1d = m1.solve(layers=layers_1d, Q=p["q_w"], r_sink=p["r_sink"])
    r_half = (t2.H_GRID / 2.0) / (t2.K_SI * t2.W_CROSS * t2.DEPTH)
    return t_1d, t_1d - p["q_w"] * r_half


def run_one(p):
    k, h_sink, layers_1d = model_for(p)
    S = t2.source_hotspot(k, q=p["q_w"], spot_mm=p["spot_mm"])

    t0 = time.time()
    T, it, res = t2.solve(S, k, h_sink=h_sink)
    sec = time.time() - t0

    t_1d, _ = t1d_cell(p, layers_1d)
    out = dict(p)
    out.update({
        "t_max": float(T.max()),
        "t_die_mean": float(T[0].mean()),
        "t_top_max": float(T[-1].max()),
        "penalty": float(T.max() - T[0].mean()),
        "t_1d": float(t_1d),
        "iters": int(it),
        "residual": float(res),
        "q_out": t2.heat_out(T, k, t2.H_GRID, h_sink=h_sink),
        "seconds": sec,
    })
    return out


# ── 검산: 새로 추가한 손잡이(TIM k, 방열판, 발열량)가 제대로 연결됐는지 ──

def check(samples):
    print("검산 — 새 손잡이 연결 확인\n")
    ok_all = True
    rng = np.random.default_rng(7)

    # 1) 무작위 조건 5개를 '균일 발열' 로 바꿔서, 흐트러진 출발점에서 1D 값으로 돌아오는지.
    #    방열판·TIM k·발열량이 코드에 잘못 전달되면 여기서 1D 와 어긋난다.
    worst = 0.0
    for p in samples[:5]:
        k, h_sink, layers_1d = model_for(p)
        S = t2.source_uniform(k, q=p["q_w"])
        T0 = t2.guess_1d(k, p["q_w"], h_sink=h_sink) + rng.uniform(0.0, 10.0, k.shape)
        T, it, _ = t2.solve(S, k, h_sink=h_sink, T0=T0, tol=1e-10)
        _, want = t1d_cell(p, layers_1d)
        err = abs(float(T[0, 0]) - want)
        worst = max(worst, err)
        print("    TIM {:.3f}mm k={:4.1f}  Q={:4.1f}W  R={:.2f}K/W  →  2D {:.4f} / 1D {:.4f}  ({}회)".format(
            p["tim_mm"], p["tim_k"], p["q_w"], p["r_sink"], float(T[0, 0]), want, it))
    ok = worst < 1e-4
    ok_all &= ok
    print("  [{}] 균일 발열 = 1D, 조건 5개 (최대 오차 {:.1e} ℃)\n".format("OK" if ok else "실패", worst))

    # 2) 선형성: 핫스팟 조건에서 발열 2배 → 온도 상승 2배
    p = dict(samples[0])
    r1 = run_one(p)
    p["q_w"] *= 2.0
    r2 = run_one(p)
    err = abs((r2["t_max"] - t2.T_AIR) - 2.0 * (r1["t_max"] - t2.T_AIR))
    ok = err < 1e-4
    ok_all &= ok
    print("  [{}] 핫스팟 조건에서 발열 2배 → 상승 2배 (오차 {:.1e} ℃)".format("OK" if ok else "실패", err))

    # 3) 에너지 보존
    err = abs(r1["q_out"] - samples[0]["q_w"])
    ok = err < 1e-4
    ok_all &= ok
    print("  [{}] 에너지 보존 (오차 {:.1e} W)".format("OK" if ok else "실패", err))
    return ok_all


def verify_all(rows):
    """전체 결과에 대한 검산. 모든 행에 대해 할 수 있는 것들."""
    print("\n전체 검산 ({}건)\n".format(len(rows)))
    ok_all = True
    r_half = (t2.H_GRID / 2.0) / (t2.K_SI * t2.W_CROSS * t2.DEPTH)

    worst_q = max(abs(r["q_out"] - r["q_w"]) / r["q_w"] for r in rows)
    ok = worst_q < 1e-6
    ok_all &= ok
    print("  [{}] 에너지 보존 (최대 상대오차 {:.1e})".format("OK" if ok else "실패", worst_q))

    # 상반성: 다이 바닥 평균은 발열 분포와 무관하게 균일 발열 값(= 1D − 반 칸)과 같아야 한다
    worst_m = max(abs(r["t_die_mean"] - (r["t_1d"] - r["q_w"] * r_half)) for r in rows)
    ok = worst_m < 1e-3
    ok_all &= ok
    print("  [{}] 모든 행에서 다이 평균 = 1D (최대 오차 {:.1e} ℃)".format("OK" if ok else "실패", worst_m))

    stuck = [r for r in rows if r["residual"] >= 1e-8]
    ok = not stuck
    ok_all &= ok
    print("  [{}] 전부 수렴 (수렴 못 한 행 {}개)".format("OK" if ok else "실패", len(stuck)))

    n_test = sum(1 for r in rows if r["split"] == "test")
    print("  학습 {}개 / 검증 {}개".format(len(rows) - n_test, n_test))
    return ok_all


def load_done():
    if not os.path.exists(OUT):
        return {}
    done = {}
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for key in r:
                if key != "split":
                    r[key] = float(r[key])
            done[int(r["idx"])] = r
    return done


def main():
    samples = sample()
    if not check(samples):
        sys.exit("검산 실패 — 스윕을 시작하지 않는다")
    if "--check" in sys.argv:
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = load_done()
    todo = [p for p in samples if p["idx"] not in done]
    print("\n스윕: 전체 {}개, 이미 끝난 것 {}개, 남은 것 {}개\n".format(
        len(samples), len(done), len(todo)))

    new_file = not os.path.exists(OUT)
    t_start = time.time()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for n, p in enumerate(todo, 1):
            r = run_one(p)
            w.writerow({c: r[c] for c in COLUMNS})
            f.flush()
            done[p["idx"]] = r
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t_start
                eta = el / n * (len(todo) - n)
                print("  {:4d}/{}  경과 {:5.0f}초  남은 시간 약 {:4.0f}초".format(n, len(todo), el, eta),
                      flush=True)

    rows = [done[i] for i in sorted(done)]
    if len(rows) == len(samples) and not verify_all(rows):
        sys.exit("전체 검산 실패")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
