"""규격 판정 vs 실제 위험 — 엇갈린 칸을 실제 2D 해석으로 확인한다.

  python app/spec_check.py      (2D 해석 최대 21회, 약 1분)

  → data/spec_check.json, data/spec_check.log

사전 등록: docs/validation-log.md 의 "규격 판정 vs 실제 위험" 절 (기준 G1·G2).
기본 조건에서 '놓친 위험' 칸과 '괜한 불량' 칸을 각각 무작위 10칸 골라, 실제 ΔT 가 예측한 판정과 맞는지 본다.
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

SEED = 20261108
N_PICK = 10
OUT_JSON = os.path.join(M.ROOT, "data", "spec_check.json")
OUT_LOG = os.path.join(M.ROOT, "data", "spec_check.log")

_lines = []


def say(text=""):
    print(text, flush=True)
    _lines.append(text)


def main():
    t_all = time.time()
    s = M.Surrogate()
    base = M.prepare(dict(M.DEFAULT))
    sm = s.spec_map(base)
    say("규격 판정 vs 실제 위험 — 모델 {}, 위험 기준 +{:g}℃, 규격 단일 보이드 {:.0%} 초과 불합격".format(
        s.version, sm["limit"], M.SPEC_SINGLE))
    say("기본 조건: " + ", ".join("{}={:g}".format(k, base[k]) for k in M.KEYS if k not in ("void_mm", "void_center_mm")))
    say("\n칸 비율 ({}칸)".format(sm["n"]))
    for name in M.SPEC_CLASSES:
        say("  {:<8} {:>5}칸  {:5.1%}".format(name, sm["counts"][name], sm["counts"][name] / sm["n"]))

    no_void = M.run_2d(base)["t_max"]
    say("\n보이드 없음 2D 해석: {:.3f}℃ (AI {:.3f}℃)".format(no_void, sm["t_no_void"]))

    rng = np.random.default_rng(SEED)
    checks, picked = [], {}
    for k, name, want_danger in ((1, "놓친 위험", True), (2, "괜한 불량", False)):
        idx = np.argwhere(sm["cls"] == k)
        if len(idx) > N_PICK:
            idx = idx[np.sort(rng.choice(len(idx), N_PICK, replace=False))]
        rows = []
        say("\n{} 칸 확인 ({}칸)".format(name, len(idx)))
        for i, j in idx:
            size, pos = float(sm["sizes"][i]), float(sm["positions"][j])
            out = M.run_2d(dict(base, void_mm=size, void_center_mm=pos))
            true_dT = out["t_max"] - no_void
            agree = (true_dT > sm["limit"]) if want_danger else (true_dT <= sm["limit"])
            rows.append({"size_mm": size, "center_mm": pos, "pred_dT": float(sm["dT"][i, j]),
                         "half95": float(sm["half95"][i, j]), "true_dT": float(true_dT), "agree": bool(agree)})
            say("  폭 {:.3f}mm 위치 {:.3f}mm  예측 {:+.2f}℃ (±{:.2f})  실제 {:+.2f}℃  {}".format(
                size, pos, sm["dT"][i, j], sm["half95"][i, j], true_dT, "맞음" if agree else "틀림"))
        picked[name] = rows
        gid = "G1" if want_danger else "G2"
        if rows:
            rate = np.mean([r["agree"] for r in rows])
            checks.append({"name": "{} {} 칸이 실제로도 {}".format(gid, name, "위험" if want_danger else "안전"),
                           "ok": bool(rate >= 0.90), "detail": "{}/{}칸 ({:.0%})".format(
                               sum(r["agree"] for r in rows), len(rows), rate)})
        else:
            checks.append({"name": "{} {} 칸".format(gid, name), "ok": None, "detail": "해당 없음 (0칸)"})

    say("\n사전 등록 기준")
    for c in checks:
        mark = "해당 없음" if c["ok"] is None else ("합격" if c["ok"] else "불합격")
        say("  [{}] {}  ({})".format(mark, c["name"], c["detail"]))
    say("({:.0f}초)".format(time.time() - t_all))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "version": s.version, "limit": sm["limit"], "spec_single": M.SPEC_SINGLE,
                   "base": {k: base[k] for k in M.KEYS}, "n_cells": sm["n"], "counts": sm["counts"],
                   "no_void_2d": no_void, "no_void_ai": sm["t_no_void"], "picked": picked,
                   "checks": checks, "date": time.strftime("%Y-%m-%d")}, f, ensure_ascii=False, indent=2)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    say("저장: " + OUT_JSON)
    return all(c["ok"] is not False for c in checks)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
