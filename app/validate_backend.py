"""웹 도구 백엔드 검증 — 화면에 올리기 전에 숫자가 맞는지 전부 확인한다.

  python app/validate_backend.py                 화면이 쓰는 모델(model.VERSION) 검증
  python app/validate_backend.py --version v1    논문 모델 검증
  python app/validate_backend.py --quick         해석기를 돌리는 검사(6~8)를 빼고 빠르게
  →  data/web_validation_<버전>.json, .log

미리 정해둔 합격 기준 (결과를 보기 전에 정함, docs/web-prd.md)
  1. 대리모델 재현     검증셋 평균 오차가 저장된 값과 같다 (v1: surrogate_void.json, v2: surrogate_v2.json)
  2. 1D 재현           800건 전부 CSV 의 t_1d 와 같다 (차이 1e-9 미만)
  3. 입력 처리         CSV 800건이 전부 범위 안이고 격자에 이미 맞춰져 있다. 범위 밖은 거절한다
  4. 95% 범위 적중률   검증 160건 중 90~98% 가 범위 안
  5. 속도              예측 0.5초, 위험 지도(2,000칸 이상) 1초 이내
  6. 해석기 재현       CSV 에서 무작위 5건을 다시 해석해 t_max 가 같다 (차이 1e-6℃ 미만)
  7. 위험 지도         기준 조건에서 무작위 30칸을 실제 해석과 비교, 95% 범위 적중 90% 이상
  8. 역산 안전성       무작위 20건, 답한 보이드 크기로 실제 해석한 온도가 모두 한계 이하

기록만 하는 것 (합격 기준 없음)
  - '자신 없음' 경고가 붙은 조건이 실제로 더 많이 틀리는지
  - 보이드가 핫스팟을 덮는 조건의 오차와 범위 적중률
  - 위험 지도에 나오는 음수(물리적으로 불가능) 의 크기
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

SEED = 42

_lines = []


def say(text=""):
    print(text, flush=True)
    _lines.append(text)


def check(results, name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    say("  [{}] {}{}".format("OK" if ok else "실패", name, "  — " + detail if detail else ""))


def random_cond(rng, with_void=True):
    """학습 데이터와 같은 분포로 조건 하나를 뽑는다 (sweep_void.sample 과 같은 규칙)."""
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    p = {
        "q_w": float(rng.uniform(*M.RANGES["q_w"])),
        "spot_mm": lu(*M.RANGES["spot_mm"]),
        "tim_mm": int(rng.integers(2, 11)) * M.H_MM,
        "tim_k": lu(*M.RANGES["tim_k"]),
        "r_sink": lu(*M.RANGES["r_sink"]),
        "void_mm": 0.0,
        "void_center_mm": 0.0,
    }
    if with_void:
        p["void_mm"] = lu(*M.RANGES["void_mm"])
        p["void_center_mm"] = float(rng.uniform(p["void_mm"] / 2, M.WIDTH_MM - p["void_mm"] / 2))
    return M.prepare(p)


def main(quick=False, version=M.VERSION):
    results, report = [], {"version": version}
    t_all = time.time()

    say("모델 {} 준비 (학습 640건, 교차검증으로 범위 보정)".format(version))
    t0 = time.time()
    s = M.Surrogate(version)
    report["model"] = {"ell": s.ell, "noise": s.noise, "k95": s.k95, "flag_half": s.flag_half,
                       "init_seconds": time.time() - t0, "n_train": len(s.train), "n_test": len(s.test)}
    say("  보정 배수 k95 = {:.3f}, 경고 기준 ±{:.2f}℃, 준비 {:.1f}초\n".format(
        s.k95, s.flag_half, report["model"]["init_seconds"]))

    test = s.test
    actual = np.array([r["t_max"] for r in test])
    conds = [M.snap(r) for r in test]
    mean, half, _ = s.predict_many(conds, [r["t_1d"] for r in test])
    err = mean - actual
    ae = np.abs(err)

    # 1. 대리모델 재현
    say("1. 대리모델 재현")
    if version == "v1":
        with open(M.SETTINGS, encoding="utf-8") as f:
            saved = json.load(f)["test"]["1D + AI"]["mae"]
    else:
        with open(M.SETTINGS_V2, encoding="utf-8") as f:
            saved = json.load(f)["test"][version]["mae"]
    check(results, "검증셋 평균 오차가 기존 결과와 같다", abs(ae.mean() - saved) < 1e-9,
          "{:.6f}℃ vs {:.6f}℃".format(ae.mean(), saved))

    # 2. 1D 재현
    say("\n2. 1D 공식 재현")
    rows = s.train + s.test
    worst = max(abs(M.t_1d(r) - r["t_1d"]) for r in rows)
    check(results, "800건 전부 CSV 의 1D 값과 같다", worst < 1e-9, "최대 차이 {:.1e}℃".format(worst))

    # 3. 입력 처리
    say("\n3. 입력 처리")
    worst, bad = 0.0, 0
    for r in rows:
        q = M.snap(r)
        worst = max(worst, max(abs(q[k] - r[k]) for k in M.KEYS))
        try:
            M.validate(q)
        except ValueError:
            bad += 1
    check(results, "CSV 800건이 격자에 이미 맞춰져 있다 (화면 슬라이더 = 학습 격자)", worst < 1e-9,
          "최대 차이 {:.1e}".format(worst))
    check(results, "CSV 800건이 전부 학습 범위 안", bad == 0, "범위 밖 {}건".format(bad))
    outside = [dict(M.DEFAULT, q_w=16.0), dict(M.DEFAULT, tim_mm=0.3), dict(M.DEFAULT, tim_k=0.5),
               dict(M.DEFAULT, r_sink=5.0), dict(M.DEFAULT, spot_mm=0.1),
               dict(M.DEFAULT, void_mm=5.0, void_center_mm=5.0)]
    rejected = 0
    for p in outside:
        try:
            M.prepare(p)
        except ValueError:
            rejected += 1
    check(results, "범위 밖 입력 6종을 모두 거절", rejected == len(outside),
          "{}/{} 거절".format(rejected, len(outside)))
    edge = M.snap(dict(M.DEFAULT, void_mm=2.0, void_center_mm=0.0))
    check(results, "가장자리 보이드는 다이 안으로 옮긴다", abs(edge["void_center_mm"] - 1.0) < 1e-9,
          "위치 0 → {:.3f}mm".format(edge["void_center_mm"]))

    # 4. 95% 범위 적중률
    say("\n4. 95% 범위 적중률 (검증 160건, 보정 배수는 학습 데이터로만 정함)")
    inside = ae <= half
    cover = float(inside.mean())
    raw_cover = float((ae <= half / s.k95 * M.Z95).mean())
    check(results, "95% 범위 적중률 90~98%", 0.90 <= cover <= 0.98,
          "{:.1%} ({}/{}건). 보정 전(정규분포 가정 1.96배)이었다면 {:.1%}".format(
              cover, int(inside.sum()), len(test), raw_cover))
    report["coverage"] = {"test": cover, "test_raw": raw_cover, "n": len(test),
                          "half_median": float(np.median(half)), "half_mean": float(half.mean())}
    say("    범위 반폭: 중앙값 ±{:.2f}℃, 평균 ±{:.2f}℃".format(np.median(half), half.mean()))

    # 기록: 경고와 덮음 그룹
    flagged = half > s.flag_half
    covers = np.array([M.gap_mm(c) == 0.0 for c in conds])
    groups = {}
    for label, mask in [("전체", np.ones(len(test), bool)), ("자신 없음 경고", flagged),
                        ("경고 없음", ~flagged), ("보이드가 핫스팟을 덮음", covers),
                        ("그 외", ~covers)]:
        if mask.sum() == 0:
            continue
        groups[label] = {"n": int(mask.sum()), "mae": float(ae[mask].mean()),
                         "max": float(ae[mask].max()), "coverage": float(inside[mask].mean())}
        say("    {:20s} {:3d}건  평균 오차 {:5.2f}℃  최대 {:6.2f}℃  범위 적중 {:5.1%}".format(
            label, groups[label]["n"], groups[label]["mae"], groups[label]["max"],
            groups[label]["coverage"]))
    report["groups"] = groups

    # 5. 속도
    say("\n5. 속도")
    times = []
    for _ in range(20):
        t0 = time.time()
        s.predict(M.DEFAULT)
        times.append(time.time() - t0)
    t_pred = float(np.median(times))
    check(results, "예측 1건 0.5초 이내", t_pred < 0.5, "{:.4f}초 (20회 중앙값)".format(t_pred))
    times = []
    for _ in range(5):
        t0 = time.time()
        rmap = s.risk_map(M.DEFAULT)
        times.append(time.time() - t0)
    t_map = float(np.median(times))
    check(results, "위험 지도 2,000칸 이상 1초 이내", rmap["n_cells"] >= 2000 and t_map < 1.0,
          "{}칸 {:.3f}초. 2D 해석으로는 약 {:.0f}분".format(
              rmap["n_cells"], t_map, rmap["n_cells"] * s.sim_seconds / 60))
    report["speed"] = {"predict_s": t_pred, "map_s": t_map, "map_cells": rmap["n_cells"],
                       "sim_seconds_mean": s.sim_seconds}
    neg = rmap["dT"][np.isfinite(rmap["dT"])]
    report["map_negative"] = {"min": float(neg.min()), "fraction": float((neg < 0).mean())}
    say("    (기록) 기준 조건 지도에서 음수 칸 {:.1%}, 가장 작은 값 {:.2f}℃ — 보이드는 온도를 못 낮추므로 예측 오차".format(
        (neg < 0).mean(), neg.min()))

    if not quick:
        rng = np.random.default_rng(SEED)

        # 6. 해석기 재현
        say("\n6. 해석기 재현 (CSV 무작위 5건을 다시 해석)")
        picks = rng.choice(len(rows), size=5, replace=False)
        worst = 0.0
        for i in picks:
            r = rows[i]
            out = M.run_2d(r)
            worst = max(worst, abs(out["t_max"] - r["t_max"]))
        check(results, "다시 해석해도 같은 답", worst < 1e-6, "최대 차이 {:.1e}℃".format(worst))

        # 7. 위험 지도 대 실제 해석
        say("\n7. 위험 지도 대 실제 해석 (기준 조건, 무작위 30칸)")
        base_2d = M.run_2d(dict(M.DEFAULT, void_mm=0.0))["t_max"]
        valid = np.argwhere(np.isfinite(rmap["dT"]))
        cells = valid[rng.choice(len(valid), size=30, replace=False)]
        map_err, map_in, map_rows = [], [], []
        for i, j in cells:
            p = M.prepare(dict(M.DEFAULT, void_mm=rmap["sizes"][i], void_center_mm=rmap["positions"][j]))
            pred = s.predict(p)
            true = M.run_2d(p)["t_max"]
            dt_true = true - base_2d
            dt_pred = rmap["dT"][i, j]
            map_err.append(abs(dt_pred - dt_true))
            map_in.append(abs(pred["t_max"] - true) <= pred["half95"])
            map_rows.append({"size_mm": p["void_mm"], "center_mm": p["void_center_mm"],
                             "dT_pred": float(dt_pred), "dT_true": float(dt_true),
                             "inside": bool(map_in[-1])})
        hit = float(np.mean(map_in))
        check(results, "지도 칸의 95% 범위 적중 90% 이상", hit >= 0.90,
              "{:.0%} ({}/30), 보이드 온도 상승 오차 평균 {:.2f}℃ 최대 {:.2f}℃".format(
                  hit, int(np.sum(map_in)), np.mean(map_err), np.max(map_err)))
        report["map_check"] = {"coverage": hit, "mae_dT": float(np.mean(map_err)),
                               "max_dT": float(np.max(map_err)), "cells": map_rows}

        # 8. 역산 안전성
        say("\n8. 허용 보이드 역산 안전성 (무작위 20건)")
        cases, n_safe, n_checked = [], 0, 0
        statuses = {}
        while len(cases) < 20:
            base = random_cond(rng, with_void=False)
            center = float(rng.uniform(0.0, M.WIDTH_MM))
            lo = s.predict(base)
            top = min(M.RANGES["void_mm"][1], 2 * min(center, M.WIDTH_MM - center))
            if top < M.RANGES["void_mm"][0]:
                continue
            hi = s.predict(dict(base, void_mm=top, void_center_mm=center))
            low_lim = lo["t_max"] + lo["half95"]
            limit = low_lim + rng.uniform(0.05, 1.0) * max(hi["t_max"] - low_lim, 0.5)
            ans = s.allowable_void(base, center, limit)
            statuses[ans["status"]] = statuses.get(ans["status"], 0) + 1
            case = {"status": ans["status"], "limit": limit, "center_mm": center,
                    "size_mm": ans["size_mm"]}
            if ans["status"] in ("size", "all"):
                true = M.run_2d(ans["cond"])["t_max"]
                n_checked += 1
                n_safe += true <= limit
                case.update({"t_true": true, "margin": limit - true})
            cases.append(case)
        margins = [c["margin"] for c in cases if "margin" in c]
        check(results, "답한 크기로 해석한 온도가 모두 한계 이하", n_checked > 0 and n_safe == n_checked,
              "{}/{}건 안전, 여유 평균 {:.2f}℃ (최소 {:.2f}℃). 크기로 답 {}건, 가장 작은 보이드부터 초과 {}건, 전부 허용 {}건".format(
                  n_safe, n_checked, np.mean(margins), np.min(margins), statuses.get("size", 0),
                  statuses.get("below_min", 0), statuses.get("all", 0)))
        report["inverse"] = {"n_checked": n_checked, "n_safe": n_safe, "statuses": statuses,
                             "margin_mean": float(np.mean(margins)),
                             "margin_min": float(np.min(margins)), "cases": cases}

    n_ok = sum(r["ok"] for r in results)
    say("\n{}/{} 통과  ({:.0f}초)".format(n_ok, len(results), time.time() - t_all))
    report.update({"checks": results, "passed": n_ok, "total": len(results), "quick": quick,
                   "date": time.strftime("%Y-%m-%d")})
    if not quick:
        out_json = M.validation_path(version)
        say("저장: " + out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(out_json[:-5] + ".log", "w", encoding="utf-8") as f:
            f.write("\n".join(_lines) + "\n")
    return n_ok == len(results)


if __name__ == "__main__":
    ver = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else M.VERSION
    ok = main(quick="--quick" in sys.argv, version=ver)
    sys.exit(0 if ok else 1)
