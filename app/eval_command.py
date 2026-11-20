"""말로 조건 바꾸기 채점 — 명령 10개를 주고 바뀐 조건이 정답과 같은지 본다

  python app/eval_command.py   →  data/eval_command.json

시작 조건: 기본값 + 보이드 1 mm (중심 3 mm 위치)
정답
  조건을 바꾸는 명령 8개: 바뀐 조건 7개 값이 정답과 모두 같아야 맞음 (격자 맞춤 후 비교)
  바꾸면 안 되는 명령 2개 (범위 밖, 질문만): 조건을 바꾸지 않아야 맞음
합격 기준 (결과를 보기 전에 정함, 2026-11-20): 10개 중 9개 이상 → 화면 맨 위 입력창으로 넣는다
명령 목록은 결과를 보기 전에 정했다. 결과를 보고 바꾸지 않는다.
"""

import json
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import model as M
import agent as A

OUT = os.path.join(M.ROOT, "data", "eval_command.json")
START = M.prepare(dict(M.DEFAULT, void_mm=1.0, void_center_mm=3.0))

CASES = [
    ("발열을 두 배로 올려줘", dict(START, q_w=10.0)),
    ("TIM 열전도율 1 올려", dict(START, tim_k=4.0)),
    ("보이드 없애줘", dict(START, void_mm=0.0)),
    ("보이드를 핫스팟 바로 위로 옮겨", dict(START, void_center_mm=5.0)),
    ("보이드 폭을 2 mm로 키워", dict(START, void_mm=2.0)),
    ("TIM 두께 0.1 mm, 방열판 저항 0.5로 바꿔", dict(START, tim_mm=0.1, r_sink=0.5)),
    ("핫스팟을 다이 전체로 균일하게 해줘", dict(START, spot_mm=10.0)),
    ("보이드를 왼쪽으로 1 mm 옮겨", dict(START, void_center_mm=2.0)),
    ("발열 30 W로 해봐", None),
    ("지금 온도 왜 이렇게 높아?", None),
]


def same(a, b):
    a, b = M.prepare(a), M.prepare(b)
    return all(abs(a[k] - b[k]) < 1e-6 for k in M.KEYS)


if __name__ == "__main__":
    cli = A.client()
    results = []
    for i, (text, want) in enumerate(CASES, 1):
        r = A.command(text, START, cli=cli)
        got = r["new_cond"]
        ok = (got is None) if want is None else (got is not None and same(got, want))
        results.append({"no": i, "command": text, "want": want, "got": got, "correct": ok,
                        "tools": [c["tool"] for c in r["calls"]], "answer": r["answer"], "tokens": r["tokens"]})
        diff = "" if got is None else ", ".join("{} {:g}→{:g}".format(k, START[k], got[k])
                                                for k in M.KEYS if abs(got[k] - START[k]) > 1e-6)
        print("{:2d}. {} {} | 바뀜: {} | 도구 {}".format(i, "O" if ok else "X", text, diff or "없음",
                                                      ",".join(results[-1]["tools"]) or "없음"), flush=True)
    n_ok = sum(r["correct"] for r in results)
    print("\n정답 {}/{} → {}".format(n_ok, len(results), "넣는다" if n_ok >= 9 else "넣지 않는다"))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": A.os.environ.get("OPENROUTER_MODEL", A.DEFAULT_MODEL), "start": START,
                   "correct": n_ok, "total": len(results), "results": results}, f, ensure_ascii=False, indent=2)
    print("저장:", OUT)
