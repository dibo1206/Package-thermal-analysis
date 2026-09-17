"""1D 열저항 모델 — 2D 계산의 검산 기준.

  python scripts/thermal_1d.py

패키지를 직렬 저항 회로로 본다. 열은 칩에서 나와 층을 차례로 통과해 공기로 빠진다.

    칩(발열) ─ R_다이 ─ R_TIM ─ R_리드 ─ R_방열판 ─ 공기

    R = t / (k * A)        층 하나의 열저항 [K/W]
    dT = Q * R_total       옴의 법칙과 같은 모양 (V = I * R)

가정: 열은 위로만, 똑바로 흐른다. 옆으로 퍼지지 않는다.
      이 가정이 틀린 만큼이 2D 계산이 필요한 이유다.
"""

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def r_cond(t, k, A):
    """전도 열저항 [K/W]. t 두께[m], k 열전도율[W/m·K], A 면적[m²]."""
    return t / (k * A)


def r_conv(h, A):
    """대류 열저항 [K/W]. 표면에서 공기로 나갈 때. h [W/m²·K]."""
    return 1.0 / (h * A)


# 기준 패키지. 다이 10x10mm, 발열 5W.
A_DIE = 10e-3 * 10e-3
Q = 5.0
T_AIR = 25.0

# (이름, 두께 m, 열전도율 W/m·K)
LAYERS = [
    ("다이 (Si)",   0.10e-3, 150.0),
    ("TIM",         0.05e-3,   3.0),
    ("리드 (Cu)",   1.00e-3, 400.0),
]
# 리드 위에 방열판을 얹었다고 두고, 방열판 전체를 저항 하나로 본다.
# 팬 달린 소형 방열판은 대략 1 K/W 수준이다.
R_HEATSINK = 1.0


def solve(layers=LAYERS, Q=Q, A=A_DIE, r_sink=R_HEATSINK, T_air=T_AIR):
    """각 층의 저항과 칩 온도를 돌려준다."""
    parts = [(name, r_cond(t, k, A)) for name, t, k in layers]
    parts.append(("방열판→공기", r_sink))
    R_total = sum(r for _, r in parts)
    T_chip = T_air + Q * R_total
    return parts, R_total, T_chip


# ─────────────────────────────────────────────────────────────
# 자체 검증 — 답을 아는 경우로 먼저 검산한다
# ─────────────────────────────────────────────────────────────

def _check(name, ok):
    print("  [{}] {}".format("OK" if ok else "실패", name))
    return ok


def self_test():
    print("자체 검증\n")
    res = []
    A = 1e-4

    # 1) 층 하나: dT = Q * t/(kA) 를 손으로 계산한 값과 같아야 한다
    _, R, T = solve([("x", 1e-3, 10.0)], Q=2.0, A=A, r_sink=0.0, T_air=0.0)
    res.append(_check("층 1개: dT = Q·t/(kA) = 2 K", abs(T - 2.0) < 1e-12))

    # 2) 직렬이면 저항이 더해진다 — 같은 층 2개 = 두께 2배 1개
    _, R2, _ = solve([("a", 1e-3, 10.0), ("b", 1e-3, 10.0)], A=A, r_sink=0.0)
    _, R1, _ = solve([("a", 2e-3, 10.0)], A=A, r_sink=0.0)
    res.append(_check("직렬 합: 1mm+1mm == 2mm", abs(R2 - R1) < 1e-12))

    # 3) 발열 2배 -> 온도 상승 2배 (선형)
    _, _, T1 = solve(Q=5.0, T_air=0.0)
    _, _, T2 = solve(Q=10.0, T_air=0.0)
    res.append(_check("Q 2배 -> dT 2배", abs(T2 - 2 * T1) < 1e-9))

    # 4) 열전도율이 무한대면 그 층 저항은 0
    res.append(_check("k -> 무한대 이면 R -> 0", r_cond(1e-3, 1e12, A) < 1e-10))

    # 5) 발열이 없으면 칩은 공기 온도
    _, _, T0 = solve(Q=0.0)
    res.append(_check("Q = 0 -> 칩 = 공기 온도", abs(T0 - T_AIR) < 1e-12))

    print("\n{}/{} 통과".format(sum(res), len(res)))
    if not all(res):
        sys.exit("검증 실패")


def report():
    parts, R_total, T_chip = solve()
    print("\n" + "=" * 56)
    print("기준 패키지: 다이 10x10mm, 발열 {:.0f}W, 공기 {:.0f}℃".format(Q, T_AIR))
    print("=" * 56)
    print("\n{:14s} {:>10s} {:>10s} {:>8s}".format("층", "R [K/W]", "dT [K]", "비중"))
    for name, r in parts:
        print("{:14s} {:10.4f} {:10.2f} {:7.1f}%".format(name, r, Q * r, r / R_total * 100))
    print("{:14s} {:10.4f} {:10.2f}".format("합계", R_total, Q * R_total))
    print("\n칩 온도 = {:.0f} + {:.1f} × {:.3f} = {:.1f} ℃".format(T_AIR, Q, R_total, T_chip))

    # 패키지 안쪽만 보면 누가 병목인가
    inside = parts[:-1]
    worst = max(inside, key=lambda p: p[1])
    print("\n패키지 안쪽 병목: {}  ({:.0f}%)".format(
        worst[0], worst[1] / sum(r for _, r in inside) * 100))


if __name__ == "__main__":
    self_test()
    report()
