"""1단계 외부 검증 — 2D 계산기 vs 발표된 해석해.

  python scripts/validate_spreading.py   →  figures/validation_spreading.png

정답지: Yovanovich, Muzychka, Culham (1999), J. Thermophysics and Heat Transfer 13(4):495-500
        "Spreading Resistance of Isoflux Rectangles and Strips on Compound Flux Channels"

비교하는 값: 확산 저항
    R_s = (열원 구간 평균온도 − 열원이 있는 면 전체 평균온도) / Q      (논문 Eq. 11~13)
  최고 온도가 아니라 '평균' 이다. 1D 로 계산되는 몫은 두 평균에 똑같이 들어 있어서 빼면 사라진다.
  남는 건 핫스팟 때문에 생기는 몫뿐이다.

순서
  0) 정답지를 잘못 베끼지 않았나: 논문 Table 2 숫자, 논문 속 극한식(Eq. 17, 18)과 비교
  1) 1층                 : 2D 계산기 vs Eq. 23
  2) 2층 실리콘 + 구리   : 2D 계산기 vs Eq. 22 + 15
  3) 2층 실리콘 + TIM    : 열전도율이 50배 차이 나는 경계. 계산기가 틀리기 가장 쉬운 경우

합격 기준 (결과를 보기 전에 정함)
  - 가장 촘촘한 격자에서 해석해와 1% 이내
  - 격자를 절반으로 줄일 때마다 오차가 줄어듦

기하 대응: 논문은 열원이 윗면, 냉각이 아랫면이다. 우리 모델은 뒤집혀 있을 뿐 같은 문제다.
  논문 c (채널 반폭) = 단면 가로 / 2,  a (열원 반폭) = 핫스팟 폭 / 2
"""

import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import thermal_2d as t2

N_TERMS = 20000         # 급수 항 수. 뒤쪽 항은 1/n³ 로 작아져서 이 정도면 충분하다

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures", "validation_spreading.png")

PASS_REL_ERR = 0.01     # 1%


# ── 정답지: 논문 공식 ──────────────────────────────────────────

def psi_semi_infinite(eps, n_terms=N_TERMS):
    """Eq. 24: 무한히 두꺼운 1층 위의 띠 열원. 무차원 k·R_s (깊이 1m당). Table 2 와 비교용."""
    n = np.arange(1, n_terms + 1)
    return np.sum(np.sin(n * np.pi * eps) ** 2 / n ** 3) / (np.pi ** 3 * eps ** 2)


def rs_isotropic(a, c, t, k, h, depth, n_terms=N_TERMS):
    """Eq. 23: 두께 t 인 1층, 반대쪽 면 대류 h. 반환 [K/W]."""
    n = np.arange(1, n_terms + 1)
    x = n * np.pi
    eps, tau, bi = a / c, t / c, h * c / k
    th = np.tanh(x * tau)
    phi = (x + bi * th) / (x * th + bi)
    k_rs = np.sum(np.sin(x * eps) ** 2 / n ** 3 * phi) / (np.pi ** 3 * eps ** 2)
    return k_rs / (k * depth)          # Eq. 23 은 깊이 1m 당 값 → 실제 깊이로 나눈다


def phi_compound(zeta, t1, t2, k1, k2, h):
    """Eq. 15 의 φ. 지수가 너무 커지지 않게 분자·분모를 e^{2ζ(2t1+t2)} 로 나눈 형태.

    원문 괄호가 추출 과정에서 모호했다. ϕ 가 두 항 모두에 곱해져야 t2→∞ 에서 Eq. 17,
    κ=1 에서 Eq. 18 로 줄어든다. 아래 검산 0-2, 0-3 이 그걸 확인한다.
    """
    kappa = k2 / k1
    alpha = (1.0 - kappa) / (1.0 + kappa)
    hk = h / k2
    vphi = (zeta + hk) / (zeta - hk)
    e_t2 = np.exp(-2.0 * zeta * t2)
    e_t12 = np.exp(-2.0 * zeta * (t1 + t2))
    e_t1 = np.exp(-2.0 * zeta * t1)
    num = alpha * e_t2 + e_t12 + vphi * (1.0 + alpha * e_t1)
    den = alpha * e_t2 - e_t12 + vphi * (1.0 - alpha * e_t1)
    return num / den


def rs_compound(a, c, t1, t2, k1, k2, h, depth, n_terms=N_TERMS):
    """Eq. 22 + 15: 2층 (열원 쪽이 1층). 반환 [K/W]."""
    m = np.arange(1, n_terms + 1)
    delta = m * np.pi / c
    per_depth = np.sum(np.sin(a * delta) ** 2 / delta ** 3
                       * phi_compound(delta, t1, t2, k1, k2, h)) / (a ** 2 * c * k1)
    return per_depth / depth


def phi_eq17(zeta, t1, k1, k2):
    """Eq. 17: 2층에서 아래층이 무한히 두꺼울 때. 대류 영향이 사라진다."""
    kappa = k2 / k1
    e1 = np.exp(-2.0 * zeta * t1)
    return ((1.0 - e1) * kappa + (1.0 + e1)) / ((1.0 + e1) * kappa + (1.0 - e1))


# ── 0) 정답지를 잘못 베끼지 않았나 ─────────────────────────────

TABLE2_ISOFLUX = {   # 논문 Table 2, μ = 0 (균일 열유속) 행
    0.02: 1.1377, 0.04: 0.9172, 0.06: 0.7883, 0.08: 0.6970, 0.10: 0.6263,
    0.20: 0.4083, 0.40: 0.1984, 0.60: 0.0882, 0.80: 0.0255,
}


def _check(name, ok, detail=""):
    print("  [{}] {}{}".format("OK" if ok else "실패", name, "  " + detail if detail else ""))
    return ok


def check_formulas():
    print("0) 공식 코드 검산 — 정답지를 제대로 베꼈나\n")
    res = []

    worst = max(abs(psi_semi_infinite(e) - v) for e, v in TABLE2_ISOFLUX.items())
    res.append(_check("논문 Table 2 숫자 9개와 일치", worst < 1e-4,
                      "최대 차이 {:.1e} (표는 소수점 4자리)".format(worst)))

    a, c, depth = 0.5e-3, 5e-3, 10e-3
    r_iso = rs_isotropic(a, c, 1.1e-3, 150.0, 1e4, depth)
    r_cmp = rs_compound(a, c, 0.1e-3, 1.0e-3, 150.0, 150.0, 1e4, depth)
    rel = abs(r_cmp - r_iso) / r_iso
    res.append(_check("2층 공식에 같은 재료를 넣으면 1층 공식과 같다 (Eq. 18)", rel < 1e-9,
                      "상대차 {:.1e}".format(rel)))

    zeta = np.arange(1, 200) * np.pi / c
    diff = np.max(np.abs(phi_compound(zeta, 0.1e-3, 1.0, 150.0, 400.0, 1e4)
                         - phi_eq17(zeta, 0.1e-3, 150.0, 400.0)))
    res.append(_check("아래층이 무한히 두꺼우면 Eq. 17 과 같다", diff < 1e-9,
                      "최대 차이 {:.1e}".format(diff)))

    k_rs_thick = rs_isotropic(a, c, 50 * c, 150.0, 1e4, depth) * 150.0 * depth
    rel = abs(k_rs_thick - psi_semi_infinite(a / c)) / psi_semi_infinite(a / c)
    res.append(_check("1층이 아주 두꺼우면 Eq. 24 와 같다", rel < 1e-9,
                      "상대차 {:.1e}".format(rel)))

    print("\n  {}/{} 통과\n".format(sum(res), len(res)))
    if not all(res):
        sys.exit("정답지 코드가 틀렸다 — 비교를 시작하지 않는다")


# ── 1~3) 2D 계산기와 비교 ──────────────────────────────────────

WIDTH = t2.W_CROSS
DEPTH = t2.DEPTH
H_SINK = t2.H_SINK
SPOT_MM = 1.0

CASES = [
    {
        "name": "1층 (실리콘 1mm)",
        "layers": [(1.0e-3, 150.0)],
        "grids_mm": [0.1, 0.05, 0.025, 0.0125],
    },
    {
        "name": "2층 (실리콘 0.1mm + 구리 1mm)",
        "layers": [(0.1e-3, 150.0), (1.0e-3, 400.0)],
        "grids_mm": [0.05, 0.025, 0.0125],
    },
    {
        "name": "2층 (실리콘 0.1mm + TIM 0.2mm)",
        "layers": [(0.1e-3, 150.0), (0.2e-3, 3.0)],
        "grids_mm": [0.05, 0.025, 0.0125],
    },
]


def rs_exact(layers):
    a, c = SPOT_MM * 1e-3 / 2.0, WIDTH / 2.0
    if len(layers) == 1:
        (t, k), = layers
        return rs_isotropic(a, c, t, k, H_SINK, DEPTH)
    (t1, k1), (t2_, k2) = layers
    return rs_compound(a, c, t1, t2_, k1, k2, H_SINK, DEPTH)


def rs_fdm(layers, h):
    """같은 계산 결과에서 확산 저항을 두 방식으로 읽는다.

    칸 중심 온도 (v1 방식) 와 표면 온도 (v2 방식, docs/validation-log.md 참고).
    판정은 표면 온도로 한다.
    """
    k = t2.build_grid(h=h, width=WIDTH, layers=layers)
    q = 1.0
    S = t2.source_hotspot(k, q=q, spot_mm=SPOT_MM, h=h)
    t0 = time.time()
    T, it, res = t2.solve(S, k, h=h, h_sink=H_SINK, tol=1e-10, max_iter=300000)
    sec = time.time() - t0
    in_spot = S[0] > 0
    center = T[0]
    surface = t2.surface_temperature(T, S, k, depth=DEPTH)
    rs_c = (center[in_spot].mean() - center.mean()) / q
    rs_s = (surface[in_spot].mean() - surface.mean()) / q
    return rs_c, rs_s, it, res, k.shape, sec


def run_cases():
    results = []
    for case in CASES:
        exact = rs_exact(case["layers"])
        print("{}  —  해석해 R_s = {:.5f} K/W".format(case["name"], exact))
        print("  {:>8s} {:>10s} {:>11s} {:>11s} {:>7s} {:>7s} {:>6s}".format(
            "칸 mm", "격자", "칸중심 오차", "표면 오차", "감소비", "반복", "초"))
        rows = []
        for h_mm in case["grids_mm"]:
            rs_c, rs_s, it, res, shape, sec = rs_fdm(case["layers"], h_mm * 1e-3)
            rel_c = (rs_c - exact) / exact
            rel_s = (rs_s - exact) / exact
            ratio = ""
            if rows and rel_s != 0:
                ratio = "{:.2f}".format(abs(rows[-1]["rel_err"]) / abs(rel_s))
            print("  {:8.4f} {:>10s} {:10.3f}% {:10.3f}% {:>7s} {:7d} {:6.1f}".format(
                h_mm, "{}x{}".format(*shape), rel_c * 100, rel_s * 100, ratio, it, sec),
                flush=True)
            if res >= 1e-10:
                print("    ※ 수렴 기준 미달 (잔차 {:.1e})".format(res))
            rows.append({"h_mm": h_mm, "rs": float(rs_s), "rel_err": float(rel_s),
                         "rs_center": float(rs_c), "rel_err_center": float(rel_c),
                         "iters": int(it), "residual": float(res), "seconds": sec})
        results.append({"name": case["name"], "exact": float(exact), "rows": rows})
        print()
    return results


def judge(results):
    print("판정 (기준: 가장 촘촘한 격자에서 1% 이내 + 격자를 줄일수록 오차 감소)\n")
    ok_all = True
    for r in results:
        errs = [abs(x["rel_err"]) for x in r["rows"]]
        finest = errs[-1]
        decreasing = all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))
        converged = all(x["residual"] < 1e-10 for x in r["rows"])
        ok = finest < PASS_REL_ERR and decreasing and converged
        ok_all &= ok
        print("  [{}] {}  최종 오차 {:.3f}%, 꾸준히 감소 {}, 전부 수렴 {}".format(
            "합격" if ok else "불합격", r["name"], finest * 100,
            "예" if decreasing else "아니오", "예" if converged else "아니오"))
    print()
    return ok_all


# ── 그림 ──────────────────────────────────────────────────────

def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    styles = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]   # 기본 팔레트 1~3번

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0, labelcolor=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.8, which="both")
    ax.set_axisbelow(True)

    for r, (color, marker) in zip(results, styles):
        hs = [x["h_mm"] for x in r["rows"]]
        errs = [abs(x["rel_err"]) * 100 for x in r["rows"]]
        ax.loglog(hs, errs, color=color, marker=marker, lw=2, ms=7, mec=SURFACE, mew=1.5,
                  label=r["name"], zorder=3)
        if "rel_err_center" in r["rows"][0]:
            errs_c = [abs(x["rel_err_center"]) * 100 for x in r["rows"]]
            ax.loglog(hs, errs_c, color=color, marker=marker, lw=1.2, ms=5, ls="--",
                      alpha=0.45, mec=SURFACE, mew=1.0, zorder=2)

    # 로그 축 기본 눈금은 수식 글꼴로 '10^-1' 을 그리는데, 맑은 고딕에 − 기호가 없어 깨진다.
    # 그냥 소수로 찍는다.
    from matplotlib.ticker import FuncFormatter, NullFormatter
    grids = sorted({x["h_mm"] for r in results for x in r["rows"]}, reverse=True)
    ax.set_xticks(grids)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.axhline(PASS_REL_ERR * 100, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(0.105, PASS_REL_ERR * 100 * 1.12, "합격 기준 1%", fontsize=9, color=INK2, ha="left")

    ax.invert_xaxis()
    ax.set_xlabel("격자 한 칸 크기 [mm]   (오른쪽으로 갈수록 촘촘함)", color=INK2, fontsize=9.5)
    ax.set_ylabel("해석해와의 오차 [%]", color=INK2, fontsize=9.5)
    ax.set_title("2D 계산기 vs 발표된 해석해 — 칸을 줄일수록 정답에 다가가는가",
                 loc="left", fontsize=11, color=INK, pad=20)
    ax.text(0, 1.02, "실선: 표면 온도로 읽음 (v2) · 흐린 점선: 칸 중심 온도로 읽음 (v1) · 해석해: Yovanovich 외 (1999)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print("저장:", OUT)


if __name__ == "__main__":
    import json
    data_path = os.path.join(os.path.dirname(HERE), "data", "validation_spreading.json")
    if "--plot-only" in sys.argv:          # 계산은 8분 걸린다. 그림만 다시 그릴 때
        with open(data_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        check_formulas()
        results = run_cases()
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    passed = judge(results)
    plot(results)
    if not passed:
        sys.exit("외부 검증 불합격")
