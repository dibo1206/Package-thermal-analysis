"""3단계 — LLM 에이전트: 말로 물으면 해석기·대리모델을 도구로 불러서 답한다

  python app/agent.py "보이드 2mm가 핫스팟 바로 위에 있으면 몇 도야?"

구조
  LLM 은 온도를 직접 계산하지 않는다 (숫자를 지어낼 수 있으므로).
  질문을 읽고 어떤 도구를 어떤 조건으로 부를지만 정하고, 도구가 돌려준 숫자로 답한다.

도구 3개
  predict_temperature  대리모델 (가우시안 과정 v2) — 0.01초, 95% 범위 포함
  run_simulation       2D 유한차분 해석 — 약 1초, 정답에 가장 가까움
  allowable_void       최고 온도 한계를 넘지 않는 가장 큰 보이드 폭 (대리모델 기반)

설정 (.env 파일 또는 환경변수, .env 는 .gitignore 에 들어 있어 GitHub 에 올라가지 않는다)
  OPENROUTER_API_KEY   필수
  OPENROUTER_MODEL     선택, 기본 anthropic/claude-haiku-5.5
"""

import json
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import model as M

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
MAX_TURNS = 8

SYSTEM = """너는 반도체 패키지 열해석 도우미다. 한국어로 짧게 답한다.

해석 대상: 다이(Si 0.1 mm) – TIM – 리드(Cu 1 mm) 단면, 다이 폭 10 mm, 주변 온도 {t_air:g}℃.
핫스팟은 항상 다이 가운데(5 mm 위치)에 있다. TIM 보이드는 공기로 채워진 빈틈이다.
보이드 위치(void_center_mm)는 다이 왼쪽 끝에서 보이드 중심까지 거리다. "핫스팟 바로 위" = 5 mm.
"핫스팟에서 2 mm 떨어진" = 중심 사이 거리 2 mm = 7 mm (또는 3 mm).

입력 범위 (이 밖은 계산할 수 없다):
- 발열량 q_w: {q[0]:g}~{q[1]:g} W
- 핫스팟 크기 spot_mm: {s[0]:g}~{s[1]:g} mm
- TIM 두께 tim_mm: {tm[0]:g}~{tm[1]:g} mm
- TIM 열전도율 tim_k: {tk[0]:g}~{tk[1]:g} W/m·K
- 방열판 저항 r_sink: {r[0]:g}~{r[1]:g} K/W
- 보이드 폭 void_mm: 0(없음) 또는 {v[0]:g}~{v[1]:g} mm
말하지 않은 조건은 기본값을 쓴다: {default}

규칙:
1. 온도를 절대 직접 계산하거나 추측하지 말고 반드시 도구를 불러라.
2. 보통은 predict_temperature 를 쓴다. 사용자가 정확한 값·검증을 원하거나, 예측의 uncertain 이 true 이면 run_simulation 도 불러라.
3. 도구가 범위 밖이라고 하면 계산하지 말고 범위 밖이라고 답하라. 범위를 임의로 바꿔서 계산하지 마라.
4. 두 경우를 비교하는 질문은 각각 도구를 불러서 차이를 계산하라.
5. 답의 마지막 줄은 반드시 다음 중 하나의 형식이다 (채점에 쓴다):
   답: <숫자> ℃        (온도 또는 온도 차이)
   답: <숫자> mm       (허용 보이드 폭)
   답: 범위 밖
""".format(t_air=M.T_AIR, q=M.RANGES["q_w"], s=M.RANGES["spot_mm"], tm=M.RANGES["tim_mm"],
           tk=M.RANGES["tim_k"], r=M.RANGES["r_sink"], v=M.RANGES["void_mm"],
           default=json.dumps(M.DEFAULT, ensure_ascii=False))

_COND = {
    "q_w": {"type": "number", "description": "발열량 [W]"},
    "spot_mm": {"type": "number", "description": "핫스팟 크기 [mm]"},
    "tim_mm": {"type": "number", "description": "TIM 두께 [mm]"},
    "tim_k": {"type": "number", "description": "TIM 열전도율 [W/m·K]"},
    "r_sink": {"type": "number", "description": "방열판 저항 [K/W]"},
    "void_mm": {"type": "number", "description": "보이드 폭 [mm], 0 이면 보이드 없음"},
    "void_center_mm": {"type": "number", "description": "다이 왼쪽 끝에서 보이드 중심까지 [mm]. 핫스팟 중심은 5"},
}

TOOLS = [
    {"type": "function", "function": {
        "name": "predict_temperature",
        "description": "대리모델로 칩 최고 온도를 빠르게 예측한다. 95% 범위와 1D 계산값도 준다.",
        "parameters": {"type": "object", "properties": _COND}}},
    {"type": "function", "function": {
        "name": "run_simulation",
        "description": "2D 유한차분 해석으로 칩 최고 온도를 계산한다. 약 1초. 가장 정확하다.",
        "parameters": {"type": "object", "properties": _COND}}},
    {"type": "function", "function": {
        "name": "allowable_void",
        "description": "주어진 위치에서 최고 온도(95% 범위 위쪽 끝)가 한계를 넘지 않는 가장 큰 보이드 폭을 찾는다.",
        "parameters": {"type": "object", "properties": dict(
            {k: v for k, v in _COND.items() if k not in ("void_mm", "void_center_mm")},
            center_mm={"type": "number", "description": "보이드 중심 위치 [mm]"},
            limit_c={"type": "number", "description": "최고 온도 한계 [℃]"}),
            "required": ["center_mm", "limit_c"]}}},
]

_surrogate = None


def surrogate():
    global _surrogate
    if _surrogate is None:
        _surrogate = M.Surrogate("v2")
    return _surrogate


def _cond(args):
    p = dict(M.DEFAULT)
    p.update({k: float(v) for k, v in args.items() if k in M.KEYS})
    if p["void_mm"] > 0 and "void_center_mm" not in args:
        p["void_center_mm"] = M.SPOT_CENTER_MM
    return p


def call_tool(name, args):
    """도구 실행. 결과는 LLM 에게 돌려줄 JSON 문자열."""
    try:
        if name == "predict_temperature":
            r = surrogate().predict(_cond(args))
            out = {"조건": r["cond"], "최고온도_C": round(r["t_max"], 2), "95%범위_C": round(r["half95"], 2),
                   "1D계산_C_핫스팟과_보이드를_무시한_균일발열_값": round(r["t_1d"], 2), "보이드가_핫스팟을_덮음": r["covers_hotspot"],
                   "uncertain": r["uncertain"]}
        elif name == "run_simulation":
            r = M.run_2d(_cond(args))
            out = {"조건": r["cond"], "최고온도_C": round(r["t_max"], 2), "계산시간_s": round(r["seconds"], 2)}
        elif name == "allowable_void":
            base = _cond({k: v for k, v in args.items() if k in M.KEYS})
            M.validate(M.snap(dict(base, void_mm=0.0)))
            r = surrogate().allowable_void(base, float(args["center_mm"]), float(args["limit_c"]))
            meaning = {"none": "보이드가 없어도 한계를 넘는다", "below_min": "가장 작은 보이드(0.25mm)부터 한계를 넘는다",
                       "size": "이 폭까지 허용", "all": "넣을 수 있는 가장 큰 보이드까지 한계 이내"}
            out = {"판정": meaning[r["status"]], "허용_보이드_폭_mm": r["size_mm"],
                   "보이드_없을때_최고온도_C": round(r["no_void"]["t_max"], 2)}
        else:
            out = {"오류": "없는 도구: " + name}
    except ValueError as e:
        out = {"범위_밖": str(e)}
    return json.dumps(out, ensure_ascii=False)


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(M.ROOT, ".env"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY 가 없습니다. thermal-analysis/.env 파일에 넣어 주세요.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def ask(question, cli=None, model=None, verbose=True):
    """질문 하나에 답한다. 돌려주는 값: 답 문장, 도구 호출 기록, 토큰 수, 걸린 시간."""
    cli = cli or client()
    model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls, tokens, t0 = [], {"in": 0, "out": 0}, time.time()
    for _ in range(MAX_TURNS):
        resp = cli.chat.completions.create(model=model, messages=messages, tools=TOOLS, temperature=0)
        if resp.usage:
            tokens["in"] += resp.usage.prompt_tokens
            tokens["out"] += resp.usage.completion_tokens
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "calls": calls, "tokens": tokens,
                    "seconds": time.time() - t0, "model": model}
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = call_tool(tc.function.name, args)
            calls.append({"tool": tc.function.name, "args": args, "result": json.loads(result)})
            if verbose:
                print("  [도구] {} {} → {}".format(tc.function.name, json.dumps(args, ensure_ascii=False), result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return {"answer": "(도구 호출이 {}번을 넘어 멈춤)".format(MAX_TURNS), "calls": calls, "tokens": tokens,
            "seconds": time.time() - t0, "model": model}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('사용법: python app/agent.py "질문"')
    r = ask(" ".join(sys.argv[1:]))
    print("\n" + r["answer"])
    print("\n(모델 {}, 도구 {}번, 토큰 입력 {} / 출력 {}, {:.1f}초)".format(
        r["model"], len(r["calls"]), r["tokens"]["in"], r["tokens"]["out"], r["seconds"]))
