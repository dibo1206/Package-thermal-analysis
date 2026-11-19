"""2단계 — 민감도 분석 (설명 가능한 AI, 순열 중요도)

  python app/sensitivity.py   →  data/sensitivity.json

방법
  새 조건 160건에서 입력 하나만 행끼리 무작위로 섞는다. 나머지 입력은 그대로 둔다.
  모델이 다시 예측하게 하고, 평균 오차가 얼마나 늘었는지 잰다.
  많이 늘수록 모델이 그 입력에 많이 기대고 있다는 뜻이다.
  섞기는 입력마다 10번 하고 평균을 낸다 (시드 42 고정).
  섞는 단위는 사람이 정하는 원래 조건 7개다. 겹침 특징은 섞인 조건으로 다시 계산한다.

두 가지 보기
  전체   : 섞인 조건으로 1D 값도 다시 계산한다 → '1D 식 + 머신러닝' 전체가 무엇에 기대나
  ML만  : 1D 값은 원래 값으로 고정한다     → 머신러닝이 배운 '추가 온도' 가 무엇에 기대나
  첫 실행에서 두 모델의 '전체' 숫자가 거의 같았다. 둘 다 같은 1D 식을 쓰니 1D 식의 순위가 나온 것이라
  'ML만' 보기를 추가했다.

나눠 보기
  모든 160건 / 보이드가 핫스팟과 겹친 68건 / 나머지 92건 (떨어진 보이드 63건 + 보이드 없음 29건)
  섞기는 항상 160건 전체에서 하고, 오차만 그룹별로 잰다.

두 모델에 똑같이 한다.
  A. 가우시안 과정 + 1D (v2)
  B. 신경망 + 1D (nn_compare.py 에서 교차검증으로 고른 설정, 시드 0)

주의
  이것은 '모델이 무엇에 기대는가' 이지 '실제 물리에서 무엇이 중요한가' 를 직접 잰 것은 아니다.
  중요도는 학습 데이터의 범위(예: 발열 범위, 보이드 폭 0.25~2.5mm)에 따라 달라진다.
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
import nn_compare as N

OUT = os.path.join(M.ROOT, "data", "sensitivity.json")
REPEATS = 10
NAMES = {"q_w": "발열량", "spot_mm": "핫스팟 크기", "tim_mm": "TIM 두께", "tim_k": "TIM 열전도율",
         "r_sink": "방열판 저항", "void_mm": "보이드 폭", "void_center_mm": "보이드 위치"}
MODES = {"full": "전체", "ml_only": "ML만"}


def rebuild(rows, originals=None):
    """섞인 조건을 격자에 맞추고 1D 값을 계산한다. 정답(t_max)은 원래 값 그대로.
    originals 를 주면 1D 값은 원래 행의 값으로 고정한다 ('ML만' 보기)."""
    out = []
    for i, r in enumerate(rows):
        c = M.snap(r)
        c["t_1d"] = originals[i]["t_1d"] if originals else M.t_1d(c)
        c["t_max"] = r["t_max"]
        out.append(c)
    return out


def errors(rows, t_hat):
    return np.abs(np.asarray(t_hat) - np.array([r["t_max"] for r in rows]))


def importance(predict, rows, groups):
    """입력마다, 그룹마다 '섞었을 때 늘어난 평균 오차' [℃]."""
    base_err = errors(rows, predict(rows))
    result = {}
    for mode in MODES:
        rng = np.random.default_rng(42)
        result[mode] = {}
        for key in M.KEYS:
            rises = {g: [] for g in groups}
            for _ in range(REPEATS):
                perm = rng.permutation(len(rows))
                shuffled = [dict(r, **{key: rows[j][key]}) for r, j in zip(rows, perm)]
                new = rebuild(shuffled, rows if mode == "ml_only" else None)
                err = errors(rows, predict(new))
                for g, mask in groups.items():
                    rises[g].append(float(err[mask].mean() - base_err[mask].mean()))
            result[mode][key] = {g: float(np.mean(v)) for g, v in rises.items()}
    return {"base_mae": {g: float(base_err[m].mean()) for g, m in groups.items()}, "rise": result}


def show(name, res, groups):
    for mode, label in MODES.items():
        print("  [{}] {}".format(label, name))
        print("    {:10s}".format("") + "".join("{:>14s}".format(g) for g in groups))
        order = sorted(M.KEYS, key=lambda k: -res["rise"][mode][k]["모든 160건"])
        for k in order:
            print("    {:10s}".format(NAMES[k]) + "".join(
                "{:>13.2f}℃".format(res["rise"][mode][k][g]) for g in groups))


if __name__ == "__main__":
    fresh = rebuild(F.load())
    overlap = np.array([M.gap_mm(r) == 0.0 for r in fresh])
    groups = {"모든 160건": np.ones(len(fresh), bool),
              "겹침 {}건".format(overlap.sum()): overlap,
              "나머지 {}건".format((~overlap).sum()): ~overlap}

    sur = M.Surrogate("v2")
    with open(N.OUT, encoding="utf-8") as f:
        setting = json.load(f)["nn_with_1d"]["setting"]
    net = N.NN(tuple(setting["hidden"]), setting["alpha"], N.SEEDS[0]).fit(
        N.x_with_baseline(sur.train), N.y_with_baseline(sur.train))

    models = {
        "A 가우시안 과정 + 1D": lambda rows: sur.predict_many(rows, [r["t_1d"] for r in rows])[0],
        "B 신경망 + 1D": lambda rows: N.to_t_max(rows, net.predict(N.x_with_baseline(rows)), True),
    }
    out = {"names": NAMES, "repeats": REPEATS, "groups": {g: int(m.sum()) for g, m in groups.items()}}
    for name, predict in models.items():
        res = importance(predict, fresh, groups)
        print("\n{} (원래 평균 오차: {})".format(name, ", ".join(
            "{} {:.2f}℃".format(g, v) for g, v in res["base_mae"].items())))
        show(name, res, groups)
        out[name] = res

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("저장:", OUT)
