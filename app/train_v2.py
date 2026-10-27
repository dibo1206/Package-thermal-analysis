"""v2 모델 설정 고르기 — 보이드와 핫스팟의 겹침 정보를 추가한 1D + AI.

  python app/train_v2.py   →  data/surrogate_v2.json

왜 만들었나 (솔직한 경위)
  1. 웹 도구 검증 7번(위험 지도 대 실제 해석)에서 v1 이 30칸 중 17칸만 95% 범위에 들었다.
     핫스팟 바로 위는 3~4℃ 낮게, 핫스팟 가장자리의 큰 보이드는 3~4℃ 높게, 먼 곳은 음수로 예측했다.
  2. 그보다 앞서 오차 분석(scripts/error_analysis.py)이 검증셋에서 '보이드가 핫스팟을 덮을 때' 오차가 몰린다는 것을
     보여줬다. 즉 이 착안은 검증셋을 본 뒤에 나왔다. 그 사실을 여기 기록한다.

규칙 (v1 과 같다)
  - 설정(거리, 잡음)은 학습 640건 안의 5겹 교차검증으로만 고른다.
  - 검증 160건은 설정을 정한 뒤 이 스크립트에서 한 번만 채점한다.
  - 후보 범위는 learning_curve.py 에서 넓힌 범위를 그대로 쓴다.
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

import model as M
import surrogate_void as sv

ELLS = [0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0]


def make_method(feat):
    def method(tr, te, ell, noise):
        mean, _, _ = M._hybrid(tr, te, [r["t_1d"] for r in te], ell, noise, feat)
        return mean
    return method


def tune(method, rows):
    sv.ELLS = ELLS
    return sv.tune(method, rows)


def score(pred, rows):
    actual = np.array([r["t_max"] for r in rows])
    ae = np.abs(pred - actual)
    covers = np.array([M.gap_mm(r) == 0.0 and r["void_mm"] > 0 for r in rows])
    return {"mae": float(ae.mean()), "max": float(ae.max()), "bias": float(np.mean(pred - actual)),
            "covers_mae": float(ae[covers].mean()), "covers_max": float(ae[covers].max()),
            "other_mae": float(ae[~covers].mean()), "other_max": float(ae[~covers].max())}


def main():
    rows = sv.load()
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    t0 = time.time()

    out = {"note": "보이드-핫스팟 겹침 특징 추가. 착안은 검증셋 오차 분석 이후. 설정은 학습 데이터 교차검증으로만 선택.",
           "ells": ELLS, "noises": sv.NOISES, "cv": {}, "test": {}}
    for name, feat in [("v1", M.features_v1), ("v2", M.features_v2)]:
        method = make_method(feat)
        cv, ell, noise = tune(method, train)
        out["cv"][name] = {"mae": cv, "ell": ell, "noise": noise}
        print("{}  교차검증 평균 오차 {:.3f}℃  (거리 {}, 잡음 {:g})".format(name, cv, ell, noise), flush=True)
        if name == "v2":
            out["chosen"] = {"ell": ell, "noise": noise}

    print("\n검증 160건 채점 (한 번)")
    v1 = json.load(open(M.SETTINGS, encoding="utf-8"))["chosen"]["1D + AI"]
    for name, feat, cfg in [("v1 (논문 설정)", M.features_v1, v1), ("v2", M.features_v2, out["chosen"])]:
        pred = make_method(feat)(train, test, cfg["ell"], cfg["noise"])
        out["test"][name] = score(pred, test)
        s = out["test"][name]
        print("  {:14s} 평균 {:.3f}℃ 최대 {:.2f}℃ | 핫스팟 덮음 평균 {:.2f}℃ 최대 {:.2f}℃ | 그 외 평균 {:.2f}℃ 최대 {:.2f}℃".format(
            name, s["mae"], s["max"], s["covers_mae"], s["covers_max"], s["other_mae"], s["other_max"]))

    out["seconds"] = time.time() - t0
    with open(M.SETTINGS_V2, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장:", M.SETTINGS_V2)


if __name__ == "__main__":
    main()
