"""2D 유한차분 열해석 — 패키지 단면의 온도 분포.

  python scripts/thermal_2d.py

1D 모델(thermal_1d.py)은 "열은 위로만 똑바로 흐른다"고 가정했다.
여기서는 단면을 격자로 잘라서 옆으로 퍼지는 것까지 계산한다.

    한 점의 온도 = 이웃들의 (저항으로 가중한) 평균 + 자기 발열 몫

    ┌──────────────────────────────┐  ← 윗면: 반칸 전도 + 방열판 → 공기
    │      리드 (Cu)    1.00mm     │
    ├──────────────────────────────┤
    │      TIM          0.05mm     │
    ├──────────────────────────────┤
    │      다이 (Si)    0.10mm     │  ← 맨 아래 줄에서 발열
    └──────────────────────────────┘  ← 바닥·옆면: 단열
                 가로 10mm

기준점: 발열을 다이 전체에 고르게 주면 1D와 같은 상황이 된다.
        그때 답이 1D와 맞는지가 이 코드의 검산이다.
"""

import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import thermal_1d as m1

# ── 재료와 치수 (1D 모델과 같은 값) ────────────────────────────
K_SI, K_TIM, K_CU = 150.0, 3.0, 400.0          # W/m·K
T_SI, T_TIM, T_CU = 0.10e-3, 0.05e-3, 1.00e-3  # m, 아래→위

W_CROSS = 10e-3   # 단면 가로 [m]
DEPTH   = 10e-3   # 화면 안쪽 방향 깊이 [m] (다이가 10x10mm 이므로)
Q       = 5.0     # 총 발열 [W]
T_AIR   = 25.0    # 공기 온도 [℃]
R_SINK  = 1.0     # 방열판→공기 열저항 [K/W], 윗면 10x10mm 전체에 대한 값
H_GRID  = 0.025e-3  # 격자 한 칸 [m] — 가장 얇은 층(TIM)이 2칸 되게

# 방열판을 '면적당' 값으로 바꿔둔다. 이렇게 해두면 단면 폭을 바꿔도 조건이 그대로다.
H_SINK  = 1.0 / (R_SINK * W_CROSS * DEPTH)   # 등가 대류계수 [W/m²·K]

LAYERS = [(T_SI, K_SI), (T_TIM, K_TIM), (T_CU, K_CU)]  # 아래→위


# ── 격자 만들기 ────────────────────────────────────────────────

def build_grid(h=H_GRID, width=W_CROSS, layers=LAYERS):
    """각 칸의 열전도율 배열 k[j, i] 를 만든다. j=0 이 맨 아래 줄."""
    cols = [np.full(int(round(t / h)), k) for t, k in layers]
    k_col = np.concatenate(cols)                 # 세로 방향 재료 분포
    nx = int(round(width / h))
    return np.repeat(k_col[:, None], nx, axis=1)


def conductances(k, h, depth=DEPTH, h_sink=H_SINK):
    """이웃 사이의 열전도도(=1/열저항) [W/K].

    정사각 격자에서는 C = k_면 · A/h = k_면 · (h·depth)/h = k_면 · depth 다.
    격자를 촘촘히 해도 이웃 간 전도도는 그대로라는 뜻.
    """
    harm = lambda a, b: 2.0 * a * b / (a + b)    # 조화평균 = 반칸 2개 직렬

    Cx = harm(k[:, :-1], k[:, 1:]) * depth       # 좌우 이웃
    Cy = harm(k[:-1, :], k[1:, :]) * depth       # 위아래 이웃

    # 윗면: 맨 윗칸 중심 → 표면(반칸 전도) → 방열판·공기(대류) 직렬
    a_cell = h * depth
    r_half = (h / 2.0) / (k[-1, :] * a_cell)
    r_conv = 1.0 / (h_sink * a_cell)
    Ctop = 1.0 / (r_half + r_conv)
    return Cx, Cy, Ctop


# ── 1D 해로 만든 출발점 ────────────────────────────────────────

def guess_1d(k, q, h=H_GRID, t_air=T_AIR, h_sink=H_SINK, depth=DEPTH):
    """1D 답으로 채운 초기 온도장. 반복법의 출발점으로 쓴다.

    각 줄의 온도 = 공기 온도 + 열유속 × (그 칸 중심에서 공기까지의 단위면적 저항)
    """
    kcol = k[:, 0]
    width = k.shape[1] * h
    flux = q / (width * depth)                   # 열유속 [W/m²]

    x = h / kcol                                 # 칸 하나의 단위면적 저항 [m²K/W]
    r_above = np.zeros(kcol.size)                # 자기 위의 칸들을 다 통과하는 저항
    r_above[:-1] = np.cumsum(x[::-1])[::-1][1:]
    r_up = (h / 2.0) / kcol + r_above + 1.0 / h_sink
    return np.repeat((t_air + flux * r_up)[:, None], k.shape[1], axis=1)


# ── 반복법으로 풀기 ────────────────────────────────────────────

def solve(S, k=None, h=H_GRID, t_air=T_AIR, omega=1.95, tol=1e-8,
          max_iter=60000, T0=None, h_sink=H_SINK):
    """S[j,i] = 그 칸에서 나는 열 [W]. 온도 배열과 반복 횟수를 돌려준다.

    각 칸에 대해 '들어온 열 = 나간 열' 을 쓰면
        T = (Σ C_이웃·T_이웃 + C_top·T_공기 + S) / (Σ C_이웃 + C_top)
    이 나온다. 이걸 값이 안 변할 때까지 반복한다.

    omega 는 과다완화(SOR) 계수다. 계산한 변화량을 omega 배로 밀어주면
    같은 답에 훨씬 빨리 도달한다. omega=1 이면 보통 가우스-자이델.

    그냥 반복하면 지독하게 느리다. 패키지 안은 서로 잘 연결돼 있는데(구리 k=400)
    공기로 나가는 출구가 1 K/W 라는 큰 병목이라 '전체 온도 수준' 이 아주 천천히
    올라온다. 그래서 두 가지를 쓴다.

      1) 1D 해에서 출발한다 (guess_1d) — 온도 수준을 처음부터 맞춰놓는다
      2) 매 스윕 뒤 전역 레벨 보정   — 나간 열이 넣은 열과 같아지게 전체를 들어올린다

    수렴 판정도 '변화량' 이 아니라 '열 균형이 안 맞는 양(잔차)' 으로 한다.
    변화량은 답에서 5℃ 떨어져 있어도 작을 수 있다 (실제로 그랬다).
    """
    if k is None:
        k = build_grid(h)
    ny, nx = k.shape
    Cx, Cy, Ctop = conductances(k, h, h_sink=h_sink)

    denom = np.zeros((ny, nx))                   # Σ C
    denom[:, :-1] += Cx; denom[:, 1:] += Cx
    denom[:-1, :] += Cy; denom[1:, :] += Cy
    denom[-1, :] += Ctop
    src = np.zeros((ny, nx))                     # 공기로 나가는 항 + 발열
    src[-1, :] += Ctop * t_air
    src += S

    q_in = float(S.sum())
    T = guess_1d(k, q_in, h, t_air, h_sink) if T0 is None else np.array(T0, dtype=float)

    jj, ii = np.indices((ny, nx))
    red = ((ii + jj) % 2 == 0)                   # 체커보드: 이웃과 번갈아 갱신

    def neighbors(field):
        nb = np.zeros((ny, nx))
        nb[:, :-1] += Cx * field[:, 1:]
        nb[:, 1:]  += Cx * field[:, :-1]
        nb[:-1, :] += Cy * field[1:, :]
        nb[1:, :]  += Cy * field[:-1, :]
        return nb

    c_top_sum = float(Ctop.sum())
    scale = max(q_in, 1e-12)                     # 잔차를 총 발열로 나눠 무차원화

    res = np.inf
    for it in range(1, max_iter + 1):
        for mask in (red, ~red):
            T += omega * np.where(mask, (neighbors(T) + src) / denom - T, 0.0)

        # 전역 레벨 보정: 윗면으로 나간 열 = 넣은 열 이 되도록 전체를 들어올린다
        q_out = float(np.sum(Ctop * (T[-1, :] - t_air)))
        T += (q_in - q_out) / c_top_sum

        res = float(np.abs(neighbors(T) + src - denom * T).max()) / scale
        if res < tol:
            break
    return T, it, res


def surface_temperature(T, S, k, depth=DEPTH):
    """다이 바닥 '표면' 온도 [℃]. 트랜지스터가 실제로 있는 면이다.

    격자 온도는 칸 중심 값이라 표면에서 반 칸 위를 잰 셈이다.
    열이 들어오는 칸은 표면 → 중심 사이 반 칸을 열이 지나가므로 표면이 그만큼 더 뜨겁다.
        T_표면 = T_중심 + 열유속 × (h/2)/k,   열유속 = S/(h·depth)
              = T_중심 + S / (2·depth·k)          (h 가 약분된다)
    열이 안 들어오는 칸은 바닥이 단열이라 표면과 중심 온도를 같게 본다.

    외부 검증 v1 에서 이 반 칸 차이가 오차의 원인으로 추정됐다 (docs/validation-log.md).
    """
    return T[0] + S[0] / (2.0 * depth * k[0])


def heat_out(T, k, h, t_air=T_AIR, h_sink=H_SINK):
    """윗면으로 빠져나가는 총 열 [W]. 에너지 보존 검산용."""
    _, _, Ctop = conductances(k, h, h_sink=h_sink)
    return float(np.sum(Ctop * (T[-1, :] - t_air)))


# ── 발열 분포 2가지 ────────────────────────────────────────────

def source_uniform(k, q=Q):
    """다이 맨 아래 줄에 고르게. 1D 모델과 같은 상황."""
    S = np.zeros(k.shape)
    S[0, :] = q / k.shape[1]
    return S


def source_hotspot(k, q=Q, spot_mm=1.0, h=H_GRID):
    """같은 5W 를 다이 가운데 좁은 구역에만. 1D 로는 계산 불가능한 상황."""
    S = np.zeros(k.shape)
    nx = k.shape[1]
    w = max(1, int(round(spot_mm * 1e-3 / h)))
    lo = (nx - w) // 2
    S[0, lo:lo + w] = q / w
    return S


# ── 1D 가 예측하는 값 ──────────────────────────────────────────

def expected_1d(h=H_GRID):
    """균일 발열일 때 다이 맨 아래 칸이 가져야 할 온도.

    1D 전체 저항에서 '맨 아래 반 칸' 만큼을 뺀다. 발열이 칸 중심에 들어가므로
    그 아래 반 칸에는 열이 흐르지 않는다(바닥 단열). 그만큼 저항이 빠진다.
    """
    _, r_total, _ = m1.solve()
    a = W_CROSS * DEPTH
    r_half = (h / 2.0) / (K_SI * a)
    return T_AIR + Q * (r_total - r_half), r_total


# ── 자체 검증 ──────────────────────────────────────────────────

def _check(name, ok, detail=""):
    print("  [{}] {}{}".format("OK" if ok else "실패", name,
                               "  " + detail if detail else ""))
    return ok


def self_test():
    print("자체 검증\n")
    res = []

    k = build_grid()
    S = source_uniform(k)
    want, _ = expected_1d()

    # 1) 핵심: 균일 발열 + 옆면 단열 = 1D 와 같은 상황.
    #    초기값 자체가 1D 해라서 그냥 풀면 검산이 공짜가 된다. 일부러 흐트러뜨린
    #    출발점을 주고, 반복법이 1D 값으로 되돌아오는지 본다.
    rng = np.random.default_rng(42)
    T_start = guess_1d(k, Q) + rng.uniform(0.0, 10.0, k.shape)
    T, it, ch = solve(S, k, T0=T_start, tol=1e-10)
    got = float(T[0, 0])
    res.append(_check("흐트러진 출발점 → 1D 값 복귀", abs(got - want) < 1e-4,
                      "2D {:.4f}℃ vs 1D {:.4f}℃ ({}회)".format(got, want, it)))

    # 1-2) 표면 온도로 읽으면 '반 칸 보정' 없이 1D 칩 온도와 바로 같아야 한다
    _, _, t_chip = m1.solve()
    ts = float(surface_temperature(T, S, k)[0])
    res.append(_check("표면 온도 = 1D 칩 온도 (반 칸 보정 없이)", abs(ts - t_chip) < 1e-4,
                      "{:.4f}℃ vs {:.4f}℃".format(ts, t_chip)))

    # 2) 옆으로 균일해야 한다 (옆면이 단열이고 발열도 고르므로 좌우 차이가 없어야).
    #    출발점에 일부러 넣은 무작위 요동이 완전히 사라졌는지 보는 것이다.
    #    기준 0.001℃ 는 온도 상승 약 6℃ 의 0.02% 수준.
    spread = float(np.ptp(T[0]))
    res.append(_check("가로 방향 온도차 없음", spread < 1e-3,
                      "편차 {:.2e}℃".format(spread)))

    # 3) 에너지 보존: 윗면으로 나간 열 = 넣은 열
    res.append(_check("에너지 보존: 나간 열 = 5W", abs(heat_out(T, k, H_GRID) - Q) < 1e-4))

    # 4) 발열 0 이면 전체가 공기 온도
    T0, _, _ = solve(np.zeros(k.shape), k)
    res.append(_check("Q = 0 → 전부 공기 온도", abs(T0.max() - T_AIR) < 1e-9))

    # 5) Q 2배 → 온도 상승 2배 (선형)
    T2, _, _ = solve(source_uniform(k, q=2 * Q), k)
    res.append(_check("Q 2배 → dT 2배",
                      abs((T2[0, 0] - T_AIR) - 2 * (T[0, 0] - T_AIR)) < 1e-4))

    # 6) 폭과 발열을 같이 1/10 로 줄이면 같은 답 (면적당 조건이 같으므로)
    k_thin = build_grid(width=1e-3)
    T3, _, _ = solve(source_uniform(k_thin, q=Q / 10.0), k_thin)
    res.append(_check("폭·발열 같이 1/10 → 같은 답", abs(T3[0, 0] - got) < 1e-4,
                      "{:.4f}℃".format(float(T3[0, 0]))))

    # 7) 핫스팟: 넣은 열은 다 나가야 하고, 균일 발열보다는 뜨거워야 한다
    Th, it_h, _ = solve(source_hotspot(k), k)
    res.append(_check("핫스팟 에너지 보존", abs(heat_out(Th, k, H_GRID) - Q) < 1e-4))
    res.append(_check("핫스팟이 균일 발열보다 뜨겁다", Th.max() > T.max() + 1.0,
                      "{:.2f}℃ vs {:.2f}℃".format(float(Th.max()), float(T.max()))))

    print("\n{}/{} 통과".format(sum(res), len(res)))
    if not all(res):
        sys.exit("검증 실패")
    return k


def report(k):
    print("\n" + "=" * 60)
    print("격자 {}x{} 칸 (한 칸 {:.3f}mm), 총 발열 {:.0f}W, 공기 {:.0f}℃".format(
        k.shape[0], k.shape[1], H_GRID * 1e3, Q, T_AIR))
    print("=" * 60)

    _, r_total = expected_1d()
    print("\n1D 모델 예측 칩 온도 : {:.2f} ℃   (총 저항 {:.3f} K/W)".format(
        T_AIR + Q * r_total, r_total))

    rows = []
    for name, S in [("균일 발열 (1D 와 동일)", source_uniform(k)),
                    ("핫스팟 1mm 에 5W 집중", source_hotspot(k))]:
        T, it, ch = solve(S, k)
        rows.append((name, T, it))
        print("\n{}".format(name))
        print("  최고 온도      {:7.2f} ℃".format(T.max()))
        print("  다이 평균      {:7.2f} ℃".format(T[0].mean()))
        print("  윗면 평균      {:7.2f} ℃".format(T[-1].mean()))
        print("  반복 {}회 (잔차 {:.1e})".format(it, ch))

    t_uni, t_hot = rows[0][1], rows[1][1]
    print("\n" + "-" * 60)
    print("핫스팟이 1D 예측보다 {:.1f}℃ 뜨겁다.".format(t_hot.max() - (T_AIR + Q * r_total)))
    print("같은 5W 인데 최고 온도가 {:.1f}℃ 차이 난다 — 1D 로는 볼 수 없는 값.".format(
        t_hot.max() - t_uni.max()))


if __name__ == "__main__":
    k = self_test()
    report(k)
