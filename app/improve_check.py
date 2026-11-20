"""'온도를 낮추는 방법' 순위가 실제 2D 해석과 맞나

  python app/improve_check.py   →  data/improve_check.json

방법
  새 조건(fresh_test.csv) 160건 중 30건을 무작위로 고른다 (시드 42).
  조건마다 model.Surrogate.improvements 가 낸 손잡이별 온도 변화(AI)를,
  같은 조건 변화로 2D 해석을 돌려 얻은 온도 변화(실제)와 비교한다.

합격 기준 (결과를 보기 전에 정함, 2026-11-20)
  1. 온도 변화의 평균 절대 오차 0.3℃ 이하
  2. AI 1등 손잡이가 실제 1등과 30건 중 24건(80%) 이상 같다
  둘 다 맞으면 메인 화면에 넣는다.
참고로 함께 적는 값 (기준 아님)
  AI 1등을 골랐을 때 실제로 내려간 온도 / 실제 1등을 골랐을 때 내려간 온도  (1 이면 손해 없음)
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

import model as M
import fresh_test as F

OUT = os.path.join(M.ROOT, "data", "improve_check.json")
N_PICK = 30

if __name__ == "__main__":
    rows = F.load()
    pick = np.random.default_rng(42).choice(len(rows), N_PICK, replace=False)
    s = M.Surrogate("v2")
    records, errs, same, ratio = [], [], 0, []
    for n, i in enumerate(pick, 1):
        base = M.prepare({k: rows[i][k] for k in M.KEYS})
        ai = s.improvements(base)
        t0 = M.run_2d(base)["t_max"]
        for r in ai:
            r["true_delta"] = M.run_2d(r["cond"])["t_max"] - t0
            errs.append(abs(r["delta"] - r["true_delta"]))
        ai_best = ai[0]
        true_best = min(ai, key=lambda r: r["true_delta"])
        same += ai_best["name"] == true_best["name"]
        ratio.append(ai_best["true_delta"] / true_best["true_delta"] if true_best["true_delta"] < 0 else 1.0)
        records.append({"idx": int(i), "base": base, "t0_2d": t0,
                        "actions": [{k: r[k] for k in ("name", "delta", "true_delta", "half95")} for r in ai],
                        "ai_best": ai_best["name"], "true_best": true_best["name"]})
        print("{:2d}. 보이드 {:.2f}mm | AI 1등 {} ({:+.2f}℃) | 실제 1등 {} ({:+.2f}℃)".format(
            n, base["void_mm"], ai_best["name"], ai_best["delta"], true_best["name"], true_best["true_delta"]),
            flush=True)

    mae = float(np.mean(errs))
    pass1, pass2 = mae <= 0.3, same >= 24
    print("\n온도 변화 평균 오차 {:.3f}℃ (최대 {:.2f}℃, {}개 비교) → 기준 1 {}".format(
        mae, max(errs), len(errs), "통과" if pass1 else "실패"))
    print("1등 일치 {}/{} → 기준 2 {}".format(same, N_PICK, "통과" if pass2 else "실패"))
    print("참고: AI 1등을 따랐을 때 얻은 효과 / 최선의 효과 = 평균 {:.0%}, 최저 {:.0%}".format(np.mean(ratio), min(ratio)))
    print("→ 메인 화면에 {}".format("넣는다" if pass1 and pass2 else "넣지 않는다"))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n": N_PICK, "seed": 42, "mae_delta": mae, "max_err": float(max(errs)), "n_compared": len(errs),
                   "top1_same": int(same), "effect_ratio_mean": float(np.mean(ratio)),
                   "effect_ratio_min": float(min(ratio)), "pass1": bool(pass1), "pass2": bool(pass2),
                   "records": records}, f, ensure_ascii=False, indent=2)
    print("저장:", OUT)
