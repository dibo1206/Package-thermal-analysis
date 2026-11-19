"""3단계 채점 — LLM 에이전트에게 질문 20개를 하고 정답과 비교한다

  python app/eval_agent.py   →  data/eval_agent.json

질문 4종류 (각 문제의 정답은 실행할 때 직접 계산한다)
  온도 8개     정답 = 2D 해석 최고 온도. 차이 1.0℃ 이내면 맞음
  비교 4개     정답 = 2D 해석 두 번의 차이. 차이 1.0℃ 이내면 맞음
  허용 보이드 4개  정답 = allowable_void 함수 결과. 0.03mm 이내면 맞음 (도구를 제대로 불렀는지)
  범위 밖 4개  정답 = '범위 밖' 이라고 답하고 계산하지 않기

도구를 한 번도 안 부르고 숫자를 답하면 '지어낸 답' 으로 따로 센다.
질문 목록은 답을 보기 전에 정했다. 답을 보고 질문을 바꾸지 않는다.
"""

import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import model as M
import agent as A

OUT = os.path.join(M.ROOT, "data", "eval_agent.json")
TOL_C = 1.0
TOL_MM = 0.03

D = M.DEFAULT
QUESTIONS = [
    # ── 온도 8개 ──
    ("온도", "기본 조건에서 칩 최고 온도는?", dict(D)),
    ("온도", "발열 10 W, 핫스팟 2 mm일 때 최고 온도 알려줘.", dict(D, q_w=10, spot_mm=2)),
    ("온도", "TIM 열전도율이 1.5 W/mK로 낮으면 최고 온도가 몇 도야?", dict(D, tim_k=1.5)),
    ("온도", "TIM 두께 0.15 mm, 방열판 저항 2 K/W, 발열 8 W이면?", dict(D, tim_mm=0.15, r_sink=2, q_w=8)),
    ("온도", "폭 2 mm 보이드가 핫스팟 바로 위에 있을 때 최고 온도는?", dict(D, void_mm=2, void_center_mm=5)),
    ("온도", "폭 1 mm 보이드가 핫스팟 중심에서 3 mm 떨어져 있으면 최고 온도는?", dict(D, void_mm=1, void_center_mm=8)),
    ("온도", "발열 12 W, 핫스팟 0.5 mm, 폭 1.5 mm 보이드가 핫스팟 위에 있으면 정확히 몇 도야?",
     dict(D, q_w=12, spot_mm=0.5, void_mm=1.5, void_center_mm=5)),
    ("온도", "핫스팟이 다이 전체(10 mm)로 균일하게 발열하고 발열 6 W면 최고 온도는?", dict(D, spot_mm=10, q_w=6)),
    # ── 비교 4개 (두 조건의 2D 온도 차이) ──
    ("비교", "폭 2 mm 보이드가 핫스팟 바로 위에 있을 때와 보이드가 없을 때 최고 온도 차이는?",
     (dict(D, void_mm=2, void_center_mm=5), dict(D))),
    ("비교", "폭 2 mm 보이드가 핫스팟 바로 위일 때와, 핫스팟 중심에서 2 mm 떨어졌을 때 온도 차이는?",
     (dict(D, void_mm=2, void_center_mm=5), dict(D, void_mm=2, void_center_mm=7))),
    ("비교", "TIM 열전도율을 3에서 6 W/mK로 바꾸면 최고 온도가 몇 도 내려가?",
     (dict(D), dict(D, tim_k=6))),
    ("비교", "발열 10 W에서 핫스팟 1 mm와 4 mm의 최고 온도 차이는?",
     (dict(D, q_w=10, spot_mm=1), dict(D, q_w=10, spot_mm=4))),
    # ── 허용 보이드 4개 (allowable_void 결과) ──
    ("허용", "핫스팟 바로 위(5 mm)에 보이드가 생긴다면 최고 온도 40℃를 넘지 않는 최대 보이드 폭은?",
     (dict(D), 5.0, 40.0)),
    ("허용", "발열 8 W일 때 핫스팟 위 보이드는 최고 온도 50℃ 이하가 되려면 몇 mm까지 괜찮아?",
     (dict(D, q_w=8), 5.0, 50.0)),
    ("허용", "보이드 중심이 6 mm 위치일 때 최고 온도 38℃ 한계라면 허용되는 보이드 폭은?",
     (dict(D), 6.0, 38.0)),
    ("허용", "TIM 두께 0.1 mm, 핫스팟 위 보이드, 최고 온도 한계 45℃에서 허용 보이드 폭은?",
     (dict(D, tim_mm=0.1), 5.0, 45.0)),
    # ── 범위 밖 4개 ──
    ("범위", "발열 30 W일 때 최고 온도는?", None),
    ("범위", "TIM 두께 0.5 mm이면 최고 온도 몇 도야?", None),
    ("범위", "TIM 열전도율 20 W/mK인 금속 TIM이면 최고 온도는?", None),
    ("범위", "방열판 저항 0.1 K/W일 때 최고 온도 알려줘.", None),
]


def truth(kind, spec, sur):
    if kind == "온도":
        return M.run_2d(spec)["t_max"]
    if kind == "비교":
        return abs(M.run_2d(spec[0])["t_max"] - M.run_2d(spec[1])["t_max"])
    if kind == "허용":
        return sur.allowable_void(spec[0], spec[1], spec[2])["size_mm"]
    return "범위 밖"


def parse(answer):
    """마지막 '답:' 줄을 읽는다."""
    lines = [l for l in answer.splitlines() if "답:" in l or "답 :" in l]
    if not lines:
        return None
    line = lines[-1]
    if "범위" in line:
        return "범위 밖"
    m = re.search(r"-?\d+(?:\.\d+)?", line.split(":", 1)[1])
    return float(m.group()) if m else None


def grade(kind, got, ans):
    if kind == "범위":
        return got == "범위 밖"
    if not isinstance(got, float):
        return False
    if kind == "비교":
        got = abs(got)
    return abs(got - ans) <= (TOL_MM if kind == "허용" else TOL_C)


if __name__ == "__main__":
    sur = A.surrogate()
    cli = A.client()
    results = []
    for i, (kind, q, spec) in enumerate(QUESTIONS, 1):
        ans = truth(kind, spec, sur)
        r = A.ask(q, cli=cli, verbose=False)
        got = parse(r["answer"])
        ok = grade(kind, got, ans)
        made_up = isinstance(got, float) and not r["calls"]
        results.append({"no": i, "model": r["model"], "kind": kind, "question": q, "truth": ans, "got": got, "correct": ok,
                        "made_up": made_up, "tools": [c["tool"] for c in r["calls"]],
                        "tokens": r["tokens"], "seconds": r["seconds"], "answer": r["answer"]})
        print("{:2d}. [{}] {} {} | 정답 {} | 답 {} | 도구 {}".format(
            i, kind, "O" if ok else "X", "(지어냄)" if made_up else "",
            "{:.2f}".format(ans) if isinstance(ans, float) else ans,
            "{:.2f}".format(got) if isinstance(got, float) else got,
            ",".join(c["tool"] for c in r["calls"]) or "없음"), flush=True)

    n_ok = sum(r["correct"] for r in results)
    by_kind = {}
    for r in results:
        k = by_kind.setdefault(r["kind"], [0, 0])
        k[0] += r["correct"]
        k[1] += 1
    tok_in = sum(r["tokens"]["in"] for r in results)
    tok_out = sum(r["tokens"]["out"] for r in results)
    print("\n정답 {}/{}  ({})".format(n_ok, len(results), ", ".join(
        "{} {}/{}".format(k, a, b) for k, (a, b) in by_kind.items())))
    print("지어낸 답 {}개, 토큰 입력 {} / 출력 {}".format(sum(r["made_up"] for r in results), tok_in, tok_out))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": results[-1]["model"],
                   "tol_c": TOL_C, "tol_mm": TOL_MM, "correct": n_ok, "total": len(results),
                   "by_kind": by_kind, "tokens": {"in": tok_in, "out": tok_out}, "results": results},
                  f, ensure_ascii=False, indent=2)
    print("저장:", OUT)
