"""보이드 데이터로 대리모델 학습 — 1D 공식 / AI 혼자 / 1D + AI 비교.

  python scripts/surrogate_void.py   →  data/surrogate_void.json, figures/surrogate_void.png

데이터: data/sweep_void.csv (800개, 학습 640 / 검증 160 은 데이터를 만들 때 이미 나눠둠)

세 가지 방법
  1D 공식만   AI 없음. 1D 모델이 계산한 칩 온도를 그대로 답으로 쓴다.
  AI 혼자     조건 7개 → 최고 온도를 통째로 배운다.
  1D + AI     최고 온도 = 1D 값 + 발열 × (AI 가 배운 1W 당 추가 온도)
              1D 가 못 보는 핫스팟·보이드 몫만 AI 가 배운다.
              '발열 2배 → 온도 상승 2배' 를 검산으로 확인했으므로 발열로 나눠도 된다.

AI 방법: 가우시안 프로세스 회귀 (numpy 만으로 구현)
  "비슷한 조건이면 비슷한 온도" — 학습 데이터 중 가까운 조건들의 값을 거리에 따라 섞어서 예측한다.

규칙
  - 설정(얼마나 멀리까지 섞을지 등)은 학습용 640개 안에서 5겹 교차검증으로만 고른다.
  - 검증용 160개는 설정을 다 정한 뒤 마지막에 한 번만 채점에 쓴다.
  - 두 AI 방법에 넣는 조건 정보는 같다. 다른 건 '무엇을 배우느냐' 뿐이다.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "sweep_void.csv")
OUT_JSON = os.path.join(ROOT, "data", "surrogate_void.json")
OUT_FIG = os.path.join(ROOT, "figures", "surrogate_void.png")

SEED = 42
N_FOLDS = 5
SPOT_CENTER_MM = 5.0
ELLS = [0.25, 0.4, 0.6, 1.0, 1.6, 2.5]
NOISES = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]


# ── 데이터 ────────────────────────────────────────────────────

def load():
    with open(DATA, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in r:
            if key != "split":
                r[key] = float(r[key])
    return rows


def features(rows, with_q):
    """두 AI 방법에 똑같이 들어가는 조건 정보.

    보이드 위치는 '핫스팟 중심에서 얼마나 떨어졌나' 로 바꿔 넣는다 (핫스팟이 항상 가운데이므로).
    보이드가 없으면 가장 먼 거리(5mm)로 둔다.
    로그 균등으로 뽑은 값은 로그를 씌운다.
    """
    X = []
    for r in rows:
        dist = abs(r["void_center_mm"] - SPOT_CENTER_MM) if r["void_mm"] > 0 else SPOT_CENTER_MM
        x = [r["tim_mm"], np.log(r["tim_k"]), np.log(r["spot_mm"]), np.log(r["r_sink"]),
             r["void_mm"], dist]
        if with_q:
            x.append(r["q_w"])
        X.append(x)
    return np.array(X)


# ── 가우시안 프로세스 ─────────────────────────────────────────

class GP:
    def __init__(self, ell, noise):
        self.ell, self.noise = ell, noise

    def _k(self, A, B):
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        return np.exp(-np.maximum(d2, 0.0) / (2.0 * self.ell ** 2))

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.Z = (X - self.mu) / self.sd
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        t = (y - self.ym) / self.ys
        K = self._k(self.Z, self.Z) + self.noise * np.eye(len(t))
        L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))
        return self

    def predict(self, X):
        Zs = (X - self.mu) / self.sd
        return self._k(Zs, self.Z) @ self.alpha * self.ys + self.ym


# ── 두 AI 방법 ────────────────────────────────────────────────

def method_alone(tr, te, ell, noise):
    """AI 혼자: 조건 7개 → 최고 온도."""
    gp = GP(ell, noise).fit(features(tr, True), np.array([r["t_max"] for r in tr]))
    return gp.predict(features(te, True))


def method_hybrid(tr, te, ell, noise):
    """1D + AI: AI 는 1W 당 추가 온도만 배운다."""
    g = np.array([(r["t_max"] - r["t_1d"]) / r["q_w"] for r in tr])
    gp = GP(ell, noise).fit(features(tr, False), g)
    g_hat = gp.predict(features(te, False))
    return np.array([r["t_1d"] + r["q_w"] * gh for r, gh in zip(te, g_hat)])


METHODS = {"AI 혼자": method_alone, "1D + AI": method_hybrid}


def cv_mae(method, rows, ell, noise, rng_seed=SEED):
    """학습 데이터 안에서만 5겹 교차검증. 최고 온도 기준 평균 오차 [℃]."""
    idx = np.random.default_rng(rng_seed).permutation(len(rows))
    folds = np.array_split(idx, N_FOLDS)
    errs = []
    for f in folds:
        hold = set(f.tolist())
        tr = [rows[i] for i in idx if i not in hold]
        va = [rows[i] for i in f]
        pred = method(tr, va, ell, noise)
        errs.extend(np.abs(pred - np.array([r["t_max"] for r in va])))
    return float(np.mean(errs))


def tune(method, rows):
    best = None
    for ell in ELLS:
        for noise in NOISES:
            try:
                m = cv_mae(method, rows, ell, noise)
            except np.linalg.LinAlgError:
                continue
            if best is None or m < best[0]:
                best = (m, ell, noise)
    return best


# ── 실행 ──────────────────────────────────────────────────────

def summary(pred, rows):
    err = pred - np.array([r["t_max"] for r in rows])
    return {"mae": float(np.mean(np.abs(err))), "max": float(np.max(np.abs(err))),
            "bias": float(np.mean(err))}


def main():
    rows = load()
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print("학습 {}개 / 검증 {}개\n".format(len(train), len(test)))

    print("1) 설정 고르기 — 학습용 640개 안에서만 5겹 교차검증")
    chosen = {}
    for name, method in METHODS.items():
        t0 = time.time()
        m, ell, noise = tune(method, train)
        chosen[name] = (ell, noise)
        print("  {:8s} 교차검증 평균 오차 {:.3f}℃  (거리 {}, 잡음 {:g})  {:.0f}초".format(
            name, m, ell, noise, time.time() - t0))

    print("\n2) 검증용 160개로 한 번만 채점\n")
    results = {}
    pred_1d = np.array([r["t_1d"] for r in test])
    results["1D 공식만"] = summary(pred_1d, test)
    preds = {"1D 공식만": pred_1d}
    for name, method in METHODS.items():
        ell, noise = chosen[name]
        t0 = time.time()
        p = method(train, test, ell, noise)
        dt = (time.time() - t0) / len(test)
        preds[name] = p
        results[name] = summary(p, test)
        results[name]["seconds_per_sample_incl_fit"] = dt

    sim_sec = float(np.mean([r["seconds"] for r in test]))
    print("  {:10s} {:>12s} {:>12s} {:>10s}".format("방법", "평균 오차", "최대 오차", "치우침"))
    for name in ["1D 공식만", "AI 혼자", "1D + AI"]:
        s = results[name]
        print("  {:10s} {:11.3f}℃ {:11.3f}℃ {:+9.3f}℃".format(name, s["mae"], s["max"], s["bias"]))
    print("\n  시뮬레이션 1건 평균 {:.2f}초".format(sim_sec))

    t_range = [min(r["t_max"] for r in test), max(r["t_max"] for r in test)]
    print("  검증셋 최고 온도 범위 {:.1f} ~ {:.1f}℃".format(*t_range))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"chosen": {k: {"ell": v[0], "noise": v[1]} for k, v in chosen.items()},
                   "test": results, "sim_seconds_mean": sim_sec,
                   "n_train": len(train), "n_test": len(test)}, f, ensure_ascii=False, indent=2)
    print("\n저장:", OUT_JSON)
    plot(preds, test, results)


def plot(preds, test, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = {"1D 공식만": "#898781", "AI 혼자": "#2a78d6", "1D + AI": "#eb6834"}
    names = ["1D 공식만", "AI 혼자", "1D + AI"]

    actual = np.array([r["t_max"] for r in test])
    # 예측값까지 포함해서 범위를 잡는다. 실제값만 쓰면 가장 크게 틀린 점이 그림 밖으로 잘린다.
    everything = np.concatenate([actual] + [np.asarray(preds[n]) for n in names])
    lo, hi = everything.min() - 3, everything.max() + 3

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes, names):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(length=0, labelcolor=INK2, labelsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(actual, preds[name], s=18, color=colors[name], edgecolor=SURFACE, lw=0.6, zorder=3)
        s = results[name]
        ax.set_title("{}\n평균 오차 {:.2f}℃ · 최대 {:.1f}℃".format(name, s["mae"], s["max"]),
                     loc="left", fontsize=10, color=INK)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("실제 최고 온도 (2D 계산) [℃]", color=INK2, fontsize=9)
    axes[0].set_ylabel("예측한 최고 온도 [℃]", color=INK2, fontsize=9)
    fig.suptitle("검증용 160개 채점 — 점선 위에 붙을수록 정확", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG, bbox_inches="tight", facecolor=SURFACE)
    print("저장:", OUT_FIG)


if __name__ == "__main__":
    main()
