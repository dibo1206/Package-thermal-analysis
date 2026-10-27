"""웹 도구의 계산 부분 — 대리모델 예측, 불확실성, 위험 지도, 허용 보이드 역산, 2D 해석 호출.

화면(streamlit_app.py)과 검증(validate_backend.py)이 이 파일 하나를 같이 쓴다.
기존 scripts/ 의 코드는 고치지 않고 가져다 쓴다.

불확실성
  가우시안 과정은 예측값과 함께 '이 예측이 얼마나 흔들릴 수 있나(표준편차)' 도 계산한다.
  다만 그 크기는 설정에 따라 실제보다 크거나 작게 나올 수 있다.
  그래서 학습 데이터 안의 5겹 교차검증으로 '실제 오차의 95% 가 들어오려면 몇 배를 곱해야 하나' 를 정한다.
  검증셋 160건은 이 배수를 정하는 데 쓰지 않는다. 맞는지 채점하는 데만 쓴다.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import surrogate_void as sv   # noqa: E402
import sweep_void as sw       # noqa: E402
import thermal_1d as m1       # noqa: E402
import thermal_2d as t2       # noqa: E402

SETTINGS = os.path.join(ROOT, "data", "surrogate_void.json")          # v1 설정 (논문 모델)
SETTINGS_V2 = os.path.join(ROOT, "data", "surrogate_v2.json")         # v2 설정 (train_v2.py)
VERSION = "v2"                                                        # 화면이 쓰는 모델


def validation_path(version):
    return os.path.join(ROOT, "data", "web_validation_{}.json".format(version))

H_MM = t2.H_GRID * 1e3                 # 격자 한 칸 0.025mm
WIDTH_MM = t2.W_CROSS * 1e3            # 다이 폭 10mm
SPOT_CENTER_MM = sv.SPOT_CENTER_MM     # 핫스팟은 항상 가운데
T_AIR = t2.T_AIR
Z95 = 1.959964

# 학습 데이터를 만든 범위 (scripts/sweep_void.py). 이 밖은 예측하지 않는다.
RANGES = {
    "q_w":     (sw.Q_W[0], sw.Q_W[1]),
    "spot_mm": (sw.SPOT_MM[0], sw.SPOT_MM[1]),
    "tim_mm":  (sw.TIM_CELLS[0] * H_MM, sw.TIM_CELLS[1] * H_MM),
    "tim_k":   (sw.TIM_K[0], sw.TIM_K[1]),
    "r_sink":  (sw.R_SINK[0], sw.R_SINK[1]),
    "void_mm": (sw.VOID_MM[0], sw.VOID_MM[1]),      # 0 은 '보이드 없음' 으로 따로 허용
}

# 지금까지 써온 기준 조건 (thermal_2d.py 의 기본값)
DEFAULT = {
    "q_w": t2.Q,
    "spot_mm": 1.0,
    "tim_mm": t2.T_TIM * 1e3,
    "tim_k": t2.K_TIM,
    "r_sink": t2.R_SINK,
    "void_mm": 0.0,
    "void_center_mm": SPOT_CENTER_MM,
}

KEYS = ["q_w", "spot_mm", "tim_mm", "tim_k", "r_sink", "void_mm", "void_center_mm"]
_EPS = 1e-9


# ── 입력 정리 ─────────────────────────────────────────────────

def snap(p):
    """입력을 격자에 맞춘다. 학습 데이터도 같은 방식으로 맞춰져 있다 (sweep_void.sample).

    - TIM 두께: 0.025mm 배수
    - 핫스팟 폭: 0.05mm 배수 (가운데 정렬을 위해 칸 수가 짝수)
    - 보이드 폭: 0.025mm 배수, 0 이면 보이드 없음
    - 보이드 위치: 보이드가 다이 밖으로 나가지 않게 자르기만 한다.
      학습 데이터도 위치는 연속값 그대로 저장했고, 격자 맞춤은 해석기(sweep_void.build)가 한다.
      여기서 위치까지 격자에 맞추면 학습 때와 다른 값이 모델에 들어간다 (검증 3번에서 확인).
    """
    q = {key: float(p[key]) for key in KEYS}
    q["tim_mm"] = round(q["tim_mm"] / H_MM) * H_MM
    q["spot_mm"] = max(int(round(q["spot_mm"] / H_MM / 2.0)) * 2, 2) * H_MM
    if q["void_mm"] <= 0:
        q["void_mm"], q["void_center_mm"] = 0.0, 0.0
    else:
        q["void_mm"] = max(int(round(q["void_mm"] / H_MM)), 1) * H_MM
        half = q["void_mm"] / 2.0
        q["void_center_mm"] = min(max(q["void_center_mm"], half), WIDTH_MM - half)
    return q


def validate(p):
    """학습 범위 밖이면 ValueError. 메시지는 화면에 그대로 보여줄 수 있게 한국어로."""
    problems = []
    names = {"q_w": "총 발열", "spot_mm": "핫스팟 크기", "tim_mm": "TIM 두께",
             "tim_k": "TIM 열전도율", "r_sink": "방열판 저항", "void_mm": "보이드 폭"}
    for key, (lo, hi) in RANGES.items():
        v = float(p[key])
        if key == "void_mm" and v == 0:
            continue
        if not (lo - _EPS <= v <= hi + _EPS):
            problems.append("{} {:g} 이(가) 학습 범위 {:g}~{:g} 밖입니다".format(names[key], v, lo, hi))
    if p["void_mm"] > 0:
        half = p["void_mm"] / 2.0
        if not (half - _EPS <= p["void_center_mm"] <= WIDTH_MM - half + _EPS):
            problems.append("보이드가 다이 밖으로 나갑니다")
    if problems:
        raise ValueError(" / ".join(problems))
    return p


def prepare(p):
    return validate(snap(p))


def gap_mm(p):
    """보이드 가장자리와 핫스팟 가장자리 사이 거리. 0 이면 보이드가 핫스팟을 (일부라도) 덮는다."""
    if p["void_mm"] <= 0:
        return np.inf
    d = abs(p["void_center_mm"] - SPOT_CENTER_MM)
    return max(0.0, d - (p["void_mm"] + p["spot_mm"]) / 2.0)


def t_1d(p):
    """1D 공식 칩 온도. sweep_void.build 와 같은 층 구성."""
    layers = [("다이 (Si)", t2.T_SI, t2.K_SI), ("TIM", p["tim_mm"] * 1e-3, p["tim_k"]),
              ("리드 (Cu)", t2.T_CU, t2.K_CU)]
    _, _, t = m1.solve(layers=layers, Q=p["q_w"], r_sink=p["r_sink"])
    return float(t)


# ── 모델에 넣는 조건 정보 ────────────────────────────────────

def overlap_gap(r):
    """보이드와 핫스팟이 가로로 겹치는 길이, 떨어진 거리 [mm]. 보이드가 없으면 (0, 5)."""
    if r["void_mm"] <= 0:
        return 0.0, SPOT_CENTER_MM
    a0 = r["void_center_mm"] - r["void_mm"] / 2.0
    a1 = r["void_center_mm"] + r["void_mm"] / 2.0
    b0 = SPOT_CENTER_MM - r["spot_mm"] / 2.0
    b1 = SPOT_CENTER_MM + r["spot_mm"] / 2.0
    return max(0.0, min(a1, b1) - max(a0, b0)), max(0.0, max(a0, b0) - min(a1, b1))


def features_v1(rows):
    """논문 모델 (surrogate_void.features 그대로)."""
    return sv.features(rows, False)


def features_v2(rows):
    """v1 + 보이드와 핫스팟의 관계 3개: 핫스팟 중 덮인 비율, 겹친 길이, 떨어진 거리.

    v1 은 보이드 폭과 '중심 사이 거리' 만 알아서, 보이드가 핫스팟에 닿는 순간 온도가 꺾여 오르는 것을
    스스로 찾아내야 했다. 매끄럽게 섞는 모델이라 그 꺾임을 뭉갰다 (검증 7번 실패).
    v2 는 그 꺾임을 조건 정보로 직접 준다.
    """
    extra = []
    for r in rows:
        ov, gap = overlap_gap(r)
        extra.append([ov / r["spot_mm"], ov, gap])
    return np.hstack([sv.features(rows, False), np.array(extra)])


FEATURES = {"v1": features_v1, "v2": features_v2}


# ── 가우시안 과정 + 표준편차 ─────────────────────────────────

class GPVar(sv.GP):
    """surrogate_void.GP 와 예측값은 똑같고, 표준편차를 추가로 계산한다."""

    def fit(self, X, y):
        super().fit(X, y)
        K = self._k(self.Z, self.Z) + self.noise * np.eye(len(self.Z))
        self.L_inv = np.linalg.inv(np.linalg.cholesky(K))
        return self

    def predict_std(self, X):
        Zs = (X - self.mu) / self.sd
        Ks = self._k(Zs, self.Z)
        mean = Ks @ self.alpha * self.ys + self.ym
        v = self.L_inv @ Ks.T
        var = np.maximum(1.0 - np.sum(v * v, axis=0), 0.0)
        return mean, np.sqrt(var) * self.ys


def _hybrid(train, conds, t1d, ell, noise, feat=features_v1):
    """1D + AI. 돌려주는 값: 예측 최고 온도, 그 표준편차(보정 전)."""
    g = np.array([(r["t_max"] - r["t_1d"]) / r["q_w"] for r in train])
    gp = GPVar(ell, noise).fit(feat(train), g)
    g_hat, g_std = gp.predict_std(feat(conds))
    q = np.array([c["q_w"] for c in conds])
    return np.asarray(t1d) + q * g_hat, q * g_std, gp


# ── 대리모델 ─────────────────────────────────────────────────

class Surrogate:
    def __init__(self, version=VERSION):
        rows = sv.load()
        self.version = version
        self.feat = FEATURES[version]
        self.train = [r for r in rows if r["split"] == "train"]
        self.test = [r for r in rows if r["split"] == "test"]
        with open(SETTINGS, encoding="utf-8") as f:
            settings = json.load(f)
        self.sim_seconds = settings["sim_seconds_mean"]
        if version == "v1":
            chosen = settings["chosen"]["1D + AI"]
        else:
            with open(SETTINGS_V2, encoding="utf-8") as f:
                chosen = json.load(f)["chosen"]
        self.ell, self.noise = chosen["ell"], chosen["noise"]

        g = np.array([(r["t_max"] - r["t_1d"]) / r["q_w"] for r in self.train])
        self.gp = GPVar(self.ell, self.noise).fit(self.feat(self.train), g)
        self._calibrate()

    def _calibrate(self):
        """학습 640건 안의 5겹 교차검증만으로 두 숫자를 정한다.

        k95       실제 오차 / 표준편차 의 95번째 백분위. 95% 범위 = 예측 ± k95 × 표준편차
        flag_half 교차검증에서 나온 95% 범위 폭의 90번째 백분위. 이보다 넓으면 '자신 없음' 경고
        """
        rows = self.train
        idx = np.random.default_rng(sv.SEED).permutation(len(rows))
        ratios, halves_raw = [], []
        for fold in np.array_split(idx, sv.N_FOLDS):
            hold = set(fold.tolist())
            tr = [rows[i] for i in idx if i not in hold]
            va = [rows[i] for i in fold]
            mean, std, _ = _hybrid(tr, va, [r["t_1d"] for r in va], self.ell, self.noise, self.feat)
            err = np.abs(mean - np.array([r["t_max"] for r in va]))
            ratios.extend(err / np.maximum(std, 1e-12))
            halves_raw.extend(std)
        self.k95 = float(np.quantile(ratios, 0.95))
        self.flag_half = float(np.quantile(np.array(halves_raw) * self.k95, 0.90))

    def predict_many(self, conds, t1d=None):
        """conds: prepare() 를 거친 조건 dict 목록. 돌려주는 값: 예측, 95% 반폭, 1D 값."""
        if t1d is None:
            t1d = [t_1d(c) for c in conds]
        t1d = np.broadcast_to(np.asarray(t1d, dtype=float), (len(conds),))
        g_hat, g_std = self.gp.predict_std(self.feat(conds))
        q = np.array([c["q_w"] for c in conds])
        return t1d + q * g_hat, self.k95 * q * g_std, np.array(t1d)

    def predict(self, p):
        p = prepare(p)
        mean, half, base = self.predict_many([p])
        return {
            "cond": p,
            "t_max": float(mean[0]),
            "half95": float(half[0]),
            "t_1d": float(base[0]),
            "extra": float(mean[0] - base[0]),
            "uncertain": bool(half[0] > self.flag_half),
            "covers_hotspot": gap_mm(p) == 0.0,
        }

    # ── 위험 지도 ──

    def risk_map(self, base, n_pos=81, n_size=50):
        """보이드 위치 × 크기마다 '보이드 때문에 오른 온도' 를 예측한다.

        base 의 보이드 값은 무시한다. 다이 밖으로 나가는 칸은 NaN.
        """
        base = prepare(dict(base, void_mm=0.0))
        lo, hi = RANGES["void_mm"]
        cells = np.unique(np.round(np.linspace(lo, hi, n_size) / H_MM).astype(int))
        sizes = cells * H_MM
        positions = np.linspace(0.0, WIDTH_MM, n_pos)

        conds, where = [], []
        for i, s in enumerate(sizes):
            for j, x in enumerate(positions):
                if s / 2.0 - _EPS <= x <= WIDTH_MM - s / 2.0 + _EPS:
                    conds.append(snap(dict(base, void_mm=s, void_center_mm=x)))
                    where.append((i, j))

        t1d = t_1d(base)                       # 1D 값은 보이드와 상관없다
        mean, half, _ = self.predict_many(conds + [base], t1d)
        no_void = mean[-1]
        dT = np.full((len(sizes), len(positions)), np.nan)
        hw = np.full_like(dT, np.nan)
        for (i, j), m, h in zip(where, mean[:-1], half[:-1]):
            dT[i, j] = m - no_void
            hw[i, j] = h
        return {"sizes": sizes, "positions": positions, "dT": dT, "half95": hw,
                "t_no_void": float(no_void), "n_cells": len(conds),
                "uncertain": hw > self.flag_half}

    # ── 허용 보이드 역산 ──

    def allowable_void(self, base, center_mm, limit_c):
        """center_mm 위치에서 '95% 범위의 위쪽 끝' 이 limit_c 를 넘지 않는 가장 큰 보이드 폭.

        status
          "none"      보이드가 없어도 위쪽 끝이 한계를 넘는다 → 답할 수 없음
          "below_min" 학습 범위의 가장 작은 보이드(0.25mm)부터 넘는다
          "size"      size_mm 까지 허용, 그다음 칸부터 넘는다
          "all"       이 위치에 넣을 수 있는 가장 큰 보이드(최대 4mm)까지 한계 이내
        """
        base = prepare(dict(base, void_mm=0.0))
        t1d = t_1d(base)
        lo, hi = RANGES["void_mm"]
        room = 2.0 * min(center_mm, WIDTH_MM - center_mm)       # 다이 밖으로 안 나가는 최대 폭
        top = min(hi, room)
        result = {"limit": float(limit_c), "center_mm": float(center_mm),
                  "max_checked_mm": float(top)}

        m0, h0, _ = self.predict_many([base], t1d)
        result["no_void"] = {"t_max": float(m0[0]), "upper": float(m0[0] + h0[0])}
        if top < lo - _EPS:
            status = "none" if m0[0] + h0[0] > limit_c else "all"
            return dict(result, status=status, size_mm=0.0)

        cells = np.arange(int(round(lo / H_MM)), int(np.floor(top / H_MM + _EPS)) + 1)
        conds = [snap(dict(base, void_mm=c * H_MM, void_center_mm=center_mm)) for c in cells]
        mean, half, _ = self.predict_many(conds, t1d)
        upper = mean + half
        result["curve"] = {"sizes": [c["void_mm"] for c in conds],
                           "t_max": mean.tolist(), "upper": upper.tolist()}
        if m0[0] + h0[0] > limit_c:
            return dict(result, status="none", size_mm=0.0)
        over = np.nonzero(upper > limit_c)[0]
        if len(over) == 0:
            k = len(conds) - 1
            status = "all"
        elif over[0] == 0:
            return dict(result, status="below_min", size_mm=0.0,
                        first_over={"size_mm": conds[0]["void_mm"], "upper": float(upper[0])})
        else:
            k = over[0] - 1
            status = "size"
        out = dict(result, status=status, size_mm=conds[k]["void_mm"], cond=conds[k],
                   t_max=float(mean[k]), upper=float(upper[k]))
        if status == "size":
            out["first_over"] = {"size_mm": conds[k + 1]["void_mm"], "upper": float(upper[k + 1])}
        return out


# ── 2D 해석 (정답 확인) ──────────────────────────────────────

def run_2d(p):
    """데이터를 만들 때와 같은 코드(sweep_void.build, thermal_2d.solve)로 해석한다."""
    p = prepare(p)
    k, h_sink, _ = sw.build(p)
    S = t2.source_hotspot(k, q=p["q_w"], spot_mm=p["spot_mm"], h=t2.H_GRID)
    import time
    t0 = time.time()
    T, it, res = t2.solve(S, k, h=t2.H_GRID, h_sink=h_sink)
    sec = time.time() - t0
    surface = t2.surface_temperature(T, S, k)
    return {
        "cond": p,
        "t_max": float(surface.max()),
        "surface": surface,
        "T": T,
        "k": k,
        "iters": int(it),
        "residual": float(res),
        "seconds": sec,
        "q_out": float(t2.heat_out(T, k, t2.H_GRID, h_sink=h_sink)),
    }


def load_validation(version=VERSION):
    """validate_backend.py 가 남긴 측정값. 없으면 None."""
    path = validation_path(version)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
