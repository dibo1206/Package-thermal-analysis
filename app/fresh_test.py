"""새 검증 데이터 — 한 번도 본 적 없는 조건 160건으로 v1·v2 를 다시 채점한다.

  python app/fresh_test.py            해석 (약 10분, 끊겨도 다시 돌리면 이어서) → 채점
  python app/fresh_test.py --score    해석은 건너뛰고 채점만

  → data/fresh_test.csv, data/fresh_test.json, data/fresh_test.log

사전 등록: docs/validation-log.md 의 "새 검증 데이터" 절 (기준 F1~F3).
규칙: 모델은 원래 학습 640건으로만 학습한다. 새 데이터로는 아무 설정도 고르지 않는다.
"""

import csv
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

import model as M
import sweep_void as sw

SEED = 20261027          # 원래 데이터는 42
N = 160
OUT_CSV = os.path.join(M.ROOT, "data", "fresh_test.csv")
OUT_JSON = os.path.join(M.ROOT, "data", "fresh_test.json")
OUT_LOG = os.path.join(M.ROOT, "data", "fresh_test.log")
COLUMNS = ["idx"] + M.KEYS + ["t_max", "t_1d", "iters", "residual", "q_out", "seconds"]

_lines = []


def say(text=""):
    print(text, flush=True)
    _lines.append(text)


def generate():
    samples = sw.sample(n=N, seed=SEED)
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8") as f:
            done = {int(r["idx"]) for r in csv.DictReader(f)}
    todo = [p for p in samples if p["idx"] not in done]
    print("새 조건 {}건, 이미 끝난 것 {}건, 남은 것 {}건".format(N, len(done), len(todo)), flush=True)

    new_file = not os.path.exists(OUT_CSV)
    t0 = time.time()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for n, p in enumerate(todo, 1):
            r = sw.run_one(p)
            w.writerow({c: r[c] for c in COLUMNS})
            f.flush()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print("  {:3d}/{}  경과 {:4.0f}초  남은 시간 약 {:4.0f}초".format(
                    n, len(todo), el, el / n * (len(todo) - n)), flush=True)


def load():
    with open(OUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in r:
            r[key] = float(r[key])
    return sorted(rows, key=lambda r: r["idx"])


def check_data(rows):
    """해석 결과가 원래 데이터와 같은 품질인지."""
    worst_q = max(abs(r["q_out"] - r["q_w"]) / r["q_w"] for r in rows)
    stuck = sum(r["residual"] >= 1e-8 for r in rows)
    below = sum(r["t_max"] < r["t_1d"] - 1e-6 for r in rows)
    say("데이터 검산: 에너지 보존 최대 상대오차 {:.1e}, 수렴 못 한 행 {}개, 2D < 1D 위반 {}건".format(
        worst_q, stuck, below))
    return worst_q < 1e-6 and stuck == 0 and below == 0


def score(rows):
    actual = np.array([r["t_max"] for r in rows])
    conds = [M.snap(r) for r in rows]
    covers = np.array([M.gap_mm(c) == 0.0 for c in conds])
    out = {}
    for version in ("v1", "v2"):
        s = M.Surrogate(version)
        mean, half, _ = s.predict_many(conds, [r["t_1d"] for r in rows])
        ae = np.abs(mean - actual)
        inside = ae <= half
        out[version] = {
            "mae": float(ae.mean()), "max": float(ae.max()), "bias": float(np.mean(mean - actual)),
            "coverage": float(inside.mean()),
            "covers_n": int(covers.sum()), "covers_mae": float(ae[covers].mean()),
            "covers_max": float(ae[covers].max()), "other_mae": float(ae[~covers].mean()),
        }
        o = out[version]
        say("  {}  평균 {:.3f}℃  최대 {:.2f}℃  95% 범위 적중 {:.1%} | 핫스팟 덮음({}건) 평균 {:.2f}℃ 최대 {:.2f}℃ | 그 외 평균 {:.2f}℃".format(
            version, o["mae"], o["max"], o["coverage"], o["covers_n"], o["covers_mae"], o["covers_max"], o["other_mae"]))
    return out


def main():
    if "--score" not in sys.argv:
        generate()
    rows = load()
    say("새 검증 데이터 {}건 (seed {})\n".format(len(rows), SEED))
    ok_data = check_data(rows)
    n_void = sum(r["void_mm"] > 0 for r in rows)
    say("보이드 있음 {}건 / 없음 {}건\n".format(n_void, len(rows) - n_void))

    say("채점 (모델은 원래 학습 640건으로만 학습)")
    res = score(rows)

    with open(M.SETTINGS_V2, encoding="utf-8") as f:
        orig = json.load(f)["test"]
    say("\n  참고: 원래 검증셋에서는 v1 {:.3f}℃, v2 {:.3f}℃".format(orig["v1 (논문 설정)"]["mae"], orig["v2"]["mae"]))

    v1, v2 = res["v1"], res["v2"]
    checks = [
        ("F1 v2 평균 오차 < v1 평균 오차", v2["mae"] < v1["mae"], "{:.3f} vs {:.3f}℃".format(v2["mae"], v1["mae"])),
        ("F2 v2 평균 오차 ≤ 0.45℃", v2["mae"] <= 0.45, "{:.3f}℃".format(v2["mae"])),
        ("F3 v2 95% 범위 적중 90~98%", 0.90 <= v2["coverage"] <= 0.98, "{:.1%}".format(v2["coverage"])),
    ]
    say("\n사전 등록 기준")
    for name, ok, detail in checks:
        say("  [{}] {}  ({})".format("합격" if ok else "불합격", name, detail))
    say("\n{}/3 합격".format(sum(ok for _, ok, _ in checks)))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n": len(rows), "n_void": n_void, "data_ok": ok_data,
                   "results": res, "original_test": {"v1": orig["v1 (논문 설정)"]["mae"], "v2": orig["v2"]["mae"]},
                   "checks": [{"name": n, "ok": bool(o), "detail": d} for n, o, d in checks],
                   "date": time.strftime("%Y-%m-%d")}, f, ensure_ascii=False, indent=2)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    say("저장: " + OUT_JSON)


if __name__ == "__main__":
    main()
