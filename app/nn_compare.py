"""1단계 — 딥러닝(심층 신경망) vs 머신러닝(가우시안 과정) 공정 비교

  python app/nn_compare.py   →  data/nn_compare.json

공정한 비교를 위해 세 가지를 똑같이 맞춘다.
  1. 학습 데이터: sweep_void.csv 의 학습 640건
  2. 입력: v2 조건 정보 (features_v2, 9개)
  3. 채점: 학습에 한 번도 안 쓴 새 조건 160건 (fresh_test.csv)

비교하는 모델 세 개
  A. 가우시안 과정 + 1D 기준선   (지금 웹 도구가 쓰는 v2)
  B. 신경망 + 1D 기준선          (A 와 똑같이 '1W당 추가 온도' 를 배운다)
  C. 신경망 단독                 (1D 기준선 없이 최고 온도를 바로 배운다)

B 와 C 를 비교하면 '물리 기준선이 딥러닝에도 도움이 되나' 를 알 수 있다.

신경망 설정(층 크기, 규제 세기)은 학습 640건 안의 5겹 교차검증으로만 고른다.
새 조건 160건은 설정을 고르는 데 쓰지 않고, 마지막 채점에만 쓴다.
신경망은 시작값이 무작위라 결과가 흔들리므로, 시드 5개로 학습해 평균과 범위를 함께 적는다.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import model as M
import fresh_test as F

OUT = os.path.join(M.ROOT, "data", "nn_compare.json")

HIDDEN = [(32, 32, 32), (64, 64, 64)]      # 은닉층 3개 → 심층 신경망
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0]   # 규제: 클수록 학습 데이터를 덜 외운다 (0.1 이 끝이라 위로 넓힘)
SEEDS = [0, 1, 2, 3, 4]


# ── 입력과 정답 ──────────────────────────────────────────────

def x_with_baseline(rows):
    return M.features_v2(rows)


def y_with_baseline(rows):
    """A·B 가 배우는 값: 1W당 추가 온도 = (2D 최고 온도 − 1D 온도) / 발열"""
    return np.array([(r["t_max"] - r["t_1d"]) / r["q_w"] for r in rows])


def x_direct(rows):
    """C 는 1D 값을 모르므로 발열량도 입력으로 줘야 한다."""
    q = np.array([[r["q_w"]] for r in rows])
    return np.hstack([M.features_v2(rows), q])


def y_direct(rows):
    return np.array([r["t_max"] for r in rows])


# ── 신경망 ────────────────────────────────────────────────────

class NN:
    """입력과 정답을 평균 0, 표준편차 1 로 맞춘 뒤 학습한다 (신경망은 숫자 크기가 제각각이면 잘 못 배운다)."""

    def __init__(self, hidden, alpha, seed):
        self.net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="lbfgs",
                                alpha=alpha, max_iter=5000, random_state=seed)
        self.sx, self.sy = StandardScaler(), StandardScaler()

    def fit(self, X, y):
        self.net.fit(self.sx.fit_transform(X), self.sy.fit_transform(y.reshape(-1, 1)).ravel())
        return self

    def predict(self, X):
        return self.sy.inverse_transform(self.net.predict(self.sx.transform(X)).reshape(-1, 1)).ravel()


def to_t_max(rows, y_hat, baseline):
    """모델 출력을 최고 온도로 바꾼다."""
    if baseline:
        return np.array([r["t_1d"] + r["q_w"] * g for r, g in zip(rows, y_hat)])
    return y_hat


def mae(rows, t_hat):
    return float(np.mean(np.abs(t_hat - np.array([r["t_max"] for r in rows]))))


def pick_setting(train, baseline):
    """학습 640건 안의 5겹 교차검증 (가우시안 과정과 같은 나누기)으로 설정을 고른다."""
    fx, fy = (x_with_baseline, y_with_baseline) if baseline else (x_direct, y_direct)
    idx = np.random.default_rng(M.sv.SEED).permutation(len(train))
    folds = np.array_split(idx, M.sv.N_FOLDS)
    results = []
    for hidden in HIDDEN:
        for alpha in ALPHAS:
            errs = []
            for fold in folds:
                hold = set(fold.tolist())
                tr = [train[i] for i in idx if i not in hold]
                va = [train[i] for i in fold]
                net = NN(hidden, alpha, SEEDS[0]).fit(fx(tr), fy(tr))
                errs.append(mae(va, to_t_max(va, net.predict(fx(va)), baseline)))
            results.append({"hidden": list(hidden), "alpha": alpha, "cv_mae": float(np.mean(errs))})
            print("    층 {} 규제 {:g}: 교차검증 오차 {:.3f}℃".format(hidden, alpha, results[-1]["cv_mae"]), flush=True)
    return min(results, key=lambda r: r["cv_mae"]), results


def run_nn(train, fresh, baseline):
    fx, fy = (x_with_baseline, y_with_baseline) if baseline else (x_direct, y_direct)
    best, table = pick_setting(train, baseline)
    maes, secs = [], []
    for seed in SEEDS:
        t0 = time.time()
        net = NN(tuple(best["hidden"]), best["alpha"], seed).fit(fx(train), fy(train))
        secs.append(time.time() - t0)
        maes.append(mae(fresh, to_t_max(fresh, net.predict(fx(fresh)), baseline)))
    return {"setting": best, "cv_table": table, "fresh_mae_by_seed": maes,
            "fresh_mae_mean": float(np.mean(maes)), "fresh_mae_min": float(min(maes)),
            "fresh_mae_max": float(max(maes)), "train_seconds_mean": float(np.mean(secs))}


# ── 실행 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    train = M.Surrogate("v2").train
    fresh = F.load()
    print("학습 {}건, 새 조건 {}건".format(len(train), len(fresh)))

    print("A. 가우시안 과정 + 1D (v2)")
    t0 = time.time()
    sur = M.Surrogate("v2")
    gp_secs = time.time() - t0
    mean, _, _ = sur.predict_many([M.snap(r) for r in fresh], [r["t_1d"] for r in fresh])
    gp = {"fresh_mae": mae(fresh, mean), "train_seconds_incl_calibration": gp_secs}
    print("  새 조건 오차 {:.3f}℃".format(gp["fresh_mae"]))

    print("B. 신경망 + 1D")
    nn_b = run_nn(train, fresh, baseline=True)
    print("C. 신경망 단독")
    nn_c = run_nn(train, fresh, baseline=False)

    print("\n결과 (새 조건 160건, 평균 절대 오차)")
    print("  A 가우시안 과정 + 1D : {:.3f}℃".format(gp["fresh_mae"]))
    for name, r in (("B 신경망 + 1D       ", nn_b), ("C 신경망 단독       ", nn_c)):
        print("  {}: {:.3f}℃ (시드 5개 범위 {:.3f}~{:.3f}, 설정 층{} 규제{:g})".format(
            name, r["fresh_mae_mean"], r["fresh_mae_min"], r["fresh_mae_max"],
            tuple(r["setting"]["hidden"]), r["setting"]["alpha"]))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n_train": len(train), "n_fresh": len(fresh), "gp_v2": gp,
                   "nn_with_1d": nn_b, "nn_direct": nn_c}, f, ensure_ascii=False, indent=2)
    print("저장:", OUT)
