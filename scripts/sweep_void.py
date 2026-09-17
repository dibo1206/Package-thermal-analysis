"""보이드가 있는 조건으로 데이터 만들기 — 대리모델 학습용.

  python scripts/sweep_void.py --check   검산만 (약 1분)
  python scripts/sweep_void.py           검산 → 전체 실행. 중간에 끊겨도 다시 돌리면 이어서 한다

  → data/sweep_void.csv

손잡이 7개 (지난 5개 + 보이드 2개)
  TIM 두께      0.05 ~ 0.25mm    격자 때문에 0.025mm 배수
  TIM 열전도율  1 ~ 10 W/m·K     로그 균등
  핫스팟 크기   0.25 ~ 10mm      로그 균등
  총 발열       1 ~ 15 W         균등
  방열판 저항   0.3 ~ 3 K/W      로그 균등
  보이드 폭     0 또는 0.25 ~ 4mm  20% 는 보이드 없음(정상 제품)
  보이드 위치   다이 안 아무 곳    균등

온도는 표면 기준 (외부 검증 v2 에서 고친 방식).
검증셋 20% 는 뽑는 순간 정해서 CSV 에 적어둔다. 결과를 보기 전에 정한다.
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
NO_VOID_FRACTION = 0.2

K_AIR = 0.026
H = t2.H_GRID

TIM_CELLS = (2, 10)
TIM_K = (1.0, 10.0)
SPOT_MM = (0.25, 10.0)
Q_W = (1.0, 15.0)
R_SINK = (0.3, 3.0)
VOID_MM = (0.25, 4.0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "sweep_void.csv")

INPUTS = ["idx", "split", "tim_mm", "tim_k", "spot_mm", "q_w", "r_sink",
          "void_mm", "void_center_mm"]
OUTPUTS = ["t_max", "t_die_mean", "penalty", "t_1d", "iters", "residual", "q_out", "seconds"]
COLUMNS = INPUTS + OUTPUTS


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample(n=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    nx = int(round(t2.W_CROSS / H))
    width_mm = t2.W_CROSS * 1e3
    rows = []
    for i in range(n):
        tim_cells = int(rng.integers(TIM_CELLS[0], TIM_CELLS[1] + 1))
        tim_k = log_uniform(rng, *TIM_K)
        spot = log_uniform(rng, *SPOT_MM)
        w_spot = min(max(int(round(spot * 1e-3 / H / 2.0)) * 2, 2), nx)
        q = float(rng.uniform(*Q_W))
        r_sink = log_uniform(rng, *R_SINK)

        if rng.random() < NO_VOID_FRACTION:
            void_mm, void_center = 0.0, 0.0
        else:
            void = log_uniform(rng, *VOID_MM)
            w_void = min(max(int(round(void * 1e-3 / H)), 1), nx)
            void_mm = w_void * H * 1e3
            void_center = float(rng.uniform(void_mm / 2.0, width_mm - void_mm / 2.0))

        rows.append({
            "idx": i,
            "tim_mm": tim_cells * H * 1e3,
            "tim_k": tim_k,
            "spot_mm": w_spot * H * 1e3,
            "q_w": q,
            "r_sink": r_sink,
            "void_mm": void_mm,
            "void_center_mm": void_center,
        })
    test = set(rng.choice(n, size=int(n * TEST_FRACTION), replace=False).tolist())
    for r in rows:
        r["split"] = "test" if r["idx"] in test else "train"
    return rows


def build(p):
    """조건 하나에 대한 격자(보이드 포함), 방열판 계수, 1D 층 구성."""
    t_tim = p["tim_mm"] * 1e-3
    layers = [(t2.T_SI, t2.K_SI), (t_tim, p["tim_k"]), (t2.T_CU, t2.K_CU)]
    k = t2.build_grid(h=H, layers=layers).copy()

    if p["void_mm"] > 0:
        n_si = int(round(t2.T_SI / H))
        n_tim = int(round(t_tim / H))
        nx = k.shape[1]
        w = min(max(int(round(p["void_mm"] * 1e-3 / H)), 1), nx)
        lo = int(round((p["void_center_mm"] - p["void_mm"] / 2.0) * 1e-3 / H))
        lo = max(0, min(lo, nx - w))
        k[n_si:n_si + n_tim, lo:lo + w] = K_AIR

    h_sink = 1.0 / (p["r_sink"] * t2.W_CROSS * t2.DEPTH)
    layers_1d = [("다이 (Si)", t2.T_SI, t2.K_SI), ("TIM", t_tim, p["tim_k"]),
                 ("리드 (Cu)", t2.T_CU, t2.K_CU)]
    return k, h_sink, layers_1d


def run_one(p):
    k, h_sink, layers_1d = build(p)
    S = t2.source_hotspot(k, q=p["q_w"], spot_mm=p["spot_mm"], h=H)

    t0 = time.time()
    T, it, res = t2.solve(S, k, h=H, h_sink=h_sink)
    sec = time.time() - t0

    surface = t2.surface_temperature(T, S, k)
    _, _, t_1d = m1.solve(layers=layers_1d, Q=p["q_w"], r_sink=p["r_sink"])

    out = dict(p)
    out.update({
        "t_max": float(surface.max()),
        "t_die_mean": float(surface.mean()),
        "penalty": float(surface.max() - surface.mean()),
        "t_1d": float(t_1d),
        "iters": int(it),
        "residual": float(res),
        "q_out": t2.heat_out(T, k, H, h_sink=h_sink),
        "seconds": sec,
    })
    return out


def check(samples):
    print("검산 — 보이드를 넣어도 기본이 유지되나\n")
    ok = []

    # 1) 보이드 없음 + 균일 발열이면 표면 온도가 1D 칩 온도와 같아야 한다
    worst = 0.0
    for p in samples[:5]:
        q = dict(p, void_mm=0.0, void_center_mm=0.0, spot_mm=t2.W_CROSS * 1e3)
        r = run_one(q)
        worst = max(worst, abs(r["t_max"] - r["t_1d"]))
    ok.append(worst < 1e-4)
    print("  [{}] 보이드 없음 + 균일 발열 = 1D (조건 5개, 최대 오차 {:.1e} ℃)".format(
        "OK" if ok[-1] else "실패", worst))

    # 2) 보이드가 있어도 넣은 열은 전부 나가야 한다
    p = next(s for s in samples if s["void_mm"] > 0)
    r = run_one(p)
    err = abs(r["q_out"] - p["q_w"])
    ok.append(err < 1e-4)
    print("  [{}] 보이드 있는 조건에서 에너지 보존 (오차 {:.1e} W)".format(
        "OK" if ok[-1] else "실패", err))

    # 3) 발열 2배 → 온도 상승 2배 (재료가 온도에 안 변하므로 성립해야 한다)
    r2 = run_one(dict(p, q_w=p["q_w"] * 2))
    err = abs((r2["t_max"] - t2.T_AIR) - 2 * (r["t_max"] - t2.T_AIR))
    ok.append(err < 1e-4)
    print("  [{}] 발열 2배 → 상승 2배 (오차 {:.1e} ℃)".format("OK" if ok[-1] else "실패", err))

    # 4) 보이드는 온도를 낮출 수 없다
    r0 = run_one(dict(p, void_mm=0.0, void_center_mm=0.0))
    ok.append(r["t_max"] >= r0["t_max"] - 1e-9)
    print("  [{}] 보이드가 있으면 더 뜨겁거나 같다 ({:.2f}℃ vs {:.2f}℃)".format(
        "OK" if ok[-1] else "실패", r["t_max"], r0["t_max"]))

    print("\n  {}/{} 통과\n".format(sum(ok), len(ok)))
    return all(ok)


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


def verify_all(rows):
    print("\n전체 검산 ({}건)\n".format(len(rows)))
    ok = []
    worst_q = max(abs(r["q_out"] - r["q_w"]) / r["q_w"] for r in rows)
    ok.append(worst_q < 1e-6)
    print("  [{}] 에너지 보존 (최대 상대오차 {:.1e})".format("OK" if ok[-1] else "실패", worst_q))

    stuck = [r for r in rows if r["residual"] >= 1e-8]
    ok.append(not stuck)
    print("  [{}] 전부 수렴 (못 한 행 {}개)".format("OK" if ok[-1] else "실패", len(stuck)))

    hotter = [r for r in rows if r["t_max"] < r["t_1d"] - 1e-6]
    ok.append(not hotter)
    print("  [{}] 2D 최고 온도 >= 1D 예측 ({}건 위반)".format("OK" if ok[-1] else "실패", len(hotter)))

    n_void = sum(1 for r in rows if r["void_mm"] > 0)
    n_test = sum(1 for r in rows if r["split"] == "test")
    print("  보이드 있음 {}개 / 없음 {}개, 학습 {}개 / 검증 {}개".format(
        n_void, len(rows) - n_void, len(rows) - n_test, n_test))
    return all(ok)


def main():
    samples = sample()
    if not check(samples):
        sys.exit("검산 실패 — 데이터 생성을 시작하지 않는다")
    if "--check" in sys.argv:
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = load_done()
    todo = [p for p in samples if p["idx"] not in done]
    print("전체 {}개, 이미 끝난 것 {}개, 남은 것 {}개\n".format(len(samples), len(done), len(todo)))

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
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t_start
                print("  {:4d}/{}  경과 {:5.0f}초  남은 시간 약 {:4.0f}초".format(
                    n, len(todo), el, el / n * (len(todo) - n)), flush=True)

    rows = [done[i] for i in sorted(done)]
    if len(rows) == len(samples) and not verify_all(rows):
        sys.exit("전체 검산 실패")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
