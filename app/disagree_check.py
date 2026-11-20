"""두 AI 의견 차이가 '이 예측 믿어도 되나' 신호로 쓸모 있나

  python app/disagree_check.py   →  data/disagree_check.json

질문
  가우시안 과정(화면의 큰 숫자)과 신경망의 예측이 많이 다를 때, 가우시안 과정의 실제 오차도 큰가?
  그리고 그 신호가 가우시안 과정이 원래 가진 95% 범위 폭보다 나은가, 같이 쓰면 나은가?

방법 (새 조건 160건, 학습에 안 쓴 데이터)
  실제 오차   e = |가우시안 과정 예측 − 2D 해석|
  의견 차이   d = |가우시안 과정 예측 − 신경망 예측|   (신경망: nn_compare.py 설정, 시드 0 — 화면과 같음)
  범위 폭     h = 가우시안 과정 95% 반폭
  '큰 오차' = e 상위 10% (16건)
  각 신호로 160건 중 상위 20%(32건)를 '확인 필요' 로 표시했을 때 큰 오차 16건 중 몇 건을 잡나.
  같은 표시 비율로 비교하므로 문턱값을 새 데이터로 고르지 않는다.

합격 기준 (결과를 보기 전에 정함, 2026-11-20)
  1. d 상위 20% 가 큰 오차 16건 중 8건 이상을 잡는다 (무작위로 표시하면 기대값 3.2건)
  2. 그리고 d 가 h 보다 더 많이 잡거나, 'd 또는 h' (각 상위 20%) 가 h 혼자보다 더 많이 잡는다
  둘 다 맞으면 메인 화면에 넣는다. 아니면 넣지 않는다.

결과를 본 뒤 찾은 기준의 허점 (2026-11-20)
  'd 또는 h' 는 32건이 아니라 41건을 표시한다. 더 많이 표시하면 더 많이 잡는 게 당연하다.
  그래서 h 도 같은 41건을 표시해 비교했더니 15건으로 'd 또는 h' 와 같았다.
  → 기준은 형식상 통과했지만, 신경망 의견 차이는 가우시안 과정의 95% 범위에 새 정보를 더하지 않는다.
  → 메인 화면에 넣지 않는다. (아래 '같은 수 비교' 줄이 이 확인이다)
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

OUT = os.path.join(M.ROOT, "data", "disagree_check.json")
FLAG_RATE = 0.20
BIG_RATE = 0.10


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def top(x, rate):
    k = int(round(len(x) * rate))
    mask = np.zeros(len(x), bool)
    mask[np.argsort(-np.asarray(x))[:k]] = True
    return mask


if __name__ == "__main__":
    rows = [dict(M.snap(r), t_1d=r["t_1d"], t_max=r["t_max"]) for r in F.load()]
    true = np.array([r["t_max"] for r in rows])

    sur = M.Surrogate("v2")
    gp, half, _ = sur.predict_many(rows, [r["t_1d"] for r in rows])

    with open(N.OUT, encoding="utf-8") as f:
        setting = json.load(f)["nn_with_1d"]["setting"]
    net = N.NN(tuple(setting["hidden"]), setting["alpha"], N.SEEDS[0]).fit(
        N.x_with_baseline(sur.train), N.y_with_baseline(sur.train))
    nn = N.to_t_max(rows, net.predict(N.x_with_baseline(rows)), True)

    e = np.abs(gp - true)
    d = np.abs(gp - nn)
    big = top(e, BIG_RATE)
    flags = {"의견 차이 d": top(d, FLAG_RATE), "범위 폭 h": top(half, FLAG_RATE)}
    flags["d 또는 h"] = flags["의견 차이 d"] | flags["범위 폭 h"]

    print("새 조건 {}건, 큰 오차 {}건 (오차 {:.2f}℃ 이상)".format(len(rows), big.sum(), e[big].min()))
    print("순위 상관 (스피어만): 실제 오차와 d {:.2f}, h {:.2f}".format(spearman(d, e), spearman(half, e)))
    caught = {}
    for name, m in flags.items():
        caught[name] = {"flagged": int(m.sum()), "caught_big": int((m & big).sum()),
                        "mae_flagged": float(e[m].mean()), "mae_rest": float(e[~m].mean())}
        print("  {:8s}: {}건 표시 → 큰 오차 {}/{}건, 표시한 곳 평균 오차 {:.2f}℃ / 나머지 {:.2f}℃".format(
            name, m.sum(), (m & big).sum(), big.sum(), e[m].mean(), e[~m].mean()))

    c_d, c_h, c_u = (caught[k]["caught_big"] for k in ("의견 차이 d", "범위 폭 h", "d 또는 h"))
    pass1 = c_d >= 8
    pass2 = c_d > c_h or c_u > c_h
    print("\n기준 1 (d 가 8건 이상): {} ({}건)".format("통과" if pass1 else "실패", c_d))
    print("기준 2 (d > h 또는 d∪h > h): {} (d {}, h {}, d∪h {})".format("통과" if pass2 else "실패", c_d, c_h, c_u))
    n_union = int(flags["d 또는 h"].sum())
    c_h_same = int((top(half, n_union / len(rows)) & big).sum())
    print("같은 수 비교 (결과를 본 뒤 추가): h 혼자 {}건 표시 → {}건, d 또는 h → {}건".format(n_union, c_h_same, c_u))
    adds = c_u > c_h_same
    print("→ 신경망 의견 차이가 새 정보를 {} → 메인 화면에 {}".format(
        "더한다" if adds else "더하지 않는다", "넣는다" if pass1 and pass2 and adds else "넣지 않는다"))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "n_big": int(big.sum()), "big_threshold": float(e[big].min()),
                   "flag_rate": FLAG_RATE, "spearman_d": spearman(d, e), "spearman_h": spearman(half, e),
                   "caught": caught, "pass1": bool(pass1), "pass2": bool(pass2),
                   "posthoc_same_count": {"n_flagged": n_union, "h_alone": c_h_same, "d_or_h": c_u,
                                          "adds_information": bool(adds)},
                   "d_flag_threshold": float(np.sort(d)[-int(round(len(d) * FLAG_RATE))])},
                  f, ensure_ascii=False, indent=2)
    print("저장:", OUT)
