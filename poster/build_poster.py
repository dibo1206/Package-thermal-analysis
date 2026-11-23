"""A2 포스터 만들기 — 숫자는 data/ 결과 파일에서 직접 읽는다 (손으로 옮겨 적지 않는다)

  python poster/build_poster.py    →  poster/poster.html
  python poster/render.py          →  poster/poster.pdf, poster/poster_preview.png

해석해 일치 0.11% 만 결과 파일에 없어 docs/abstract-example.md (2026-11-04 검산) 값을 그대로 쓴다.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))

import model as M  # noqa: E402

GITHUB = "https://github.com/dibo1206/Package-thermal-analysis"
GRID_ERR = "0.11%"            # docs/abstract-example.md: 0.025 mm 격자, 1·2층 구조, 발표된 해석해 대비


def load(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as f:
        return json.load(f)


NC = load("nn_compare.json")
SE = load("sensitivity.json")
IC = load("improve_check.json")
EC = load("eval_command.json")
EV = load("eval_agent.json")
DC = load("disagree_check.json")
FIG = load("abstract_fig1.json")
LC = load("learning_curve.json")
WV = load("web_validation_v2.json")

gp = NC["gp_v2"]["fresh_mae"]
nn = NC["nn_with_1d"]["fresh_mae_mean"]
nn_lo, nn_hi = NC["nn_with_1d"]["fresh_mae_min"], NC["nn_with_1d"]["fresh_mae_max"]
nd = NC["nn_direct"]["fresh_mae_mean"]
se = SE["A 가우시안 과정 + 1D"]["rise"]["ml_only"]
g_all, g_ov, g_rest = list(SE["groups"])
void_ov, void_rest = se["void_mm"][g_ov], se["void_mm"][g_rest]
sink = abs(se["r_sink"][g_all])
made_up = sum(r["made_up"] for r in EV["results"])
offs, rise = FIG["a"]["offsets"], FIG["a"]["rise"]
r_on, r_off = rise["2.0"][0], rise["2.0"][offs.index(2.0)]

# 학습 곡선: 평균 오차 2.0℃ 에 처음 도달하는 데이터 크기 (v1, 시험한 6단계 중)
def first_below(name, target=2.0):
    for n in LC["sizes"]:
        if LC["results"][name][str(n)]["mean"] <= target:
            return n
    return None


n_ai = first_below("AI 혼자")
n_hyb = first_below([k for k in LC["results"] if k != "AI 혼자"][0])

# 위험 지도: AI 시간 vs 2D 해석 시간
sur = M.Surrogate("v2")
cells = sur.risk_map(M.DEFAULT)["n_cells"]
sim_min = cells * sur.sim_seconds / 60


# ── 그림 (SVG, 인쇄해도 깨지지 않게 벡터) ─────────────────────

INK, INK2, MUTED, GRID, ACCENT = "#0b0b0b", "#52514e", "#6b6a66", "#e1e0d9", "#eb6834"


def void_chart():
    W, H, l, r, t, b = 640, 305, 70, 20, 12, 58
    xmax, ymax = 4.0, 16.0
    X = lambda x: l + (W - l - r) * x / xmax
    Y = lambda y: t + (H - t - b) * (1 - y / ymax)
    out = ['<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" font-family="Pretendard, Malgun Gothic">'.format(W, H)]
    out.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.10"/>'.format(
        X(0), t, X(0.5) - X(0), H - t - b, ACCENT))
    out.append('<text x="{}" y="{}" font-size="15" fill="{}" text-anchor="middle">핫스팟</text>'.format(X(0.25), Y(0.9), INK2))
    for y in range(0, 17, 4):
        out.append('<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{}" stroke-width="1"/>'.format(l, W - r, Y(y), Y(y), GRID))
        out.append('<text x="{}" y="{}" font-size="15" fill="{}" text-anchor="end">{}</text>'.format(l - 10, Y(y) + 5, MUTED, y))
    for x in range(0, 5):
        out.append('<text x="{}" y="{}" font-size="15" fill="{}" text-anchor="middle">{}</text>'.format(X(x), H - b + 24, MUTED, x))
    out.append('<text x="{}" y="{}" font-size="16" fill="{}" text-anchor="middle">가장 뜨거운 곳(핫스팟)에서 빈틈까지 거리 (mm)</text>'.format(
        (l + W - r) / 2, H - 10, INK2))
    out.append('<text x="18" y="{}" font-size="16" fill="{}" text-anchor="middle" transform="rotate(-90 18 {})">최고 온도가 오른 양 (℃)</text>'.format(
        (t + H - b) / 2, INK2, (t + H - b) / 2))
    shades = {"0.5": "#f8c9ad", "1.0": "#f19566", "2.0": "#b8471a"}
    for w in ("0.5", "1.0", "2.0"):
        pts = " ".join("{:.1f},{:.1f}".format(X(x), Y(y)) for x, y in zip(offs, rise[w]))
        out.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="4" stroke-linejoin="round"/>'.format(pts, shades[w]))
        k = ("2.0", "1.0", "0.5").index(w)
        out.append('<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{}" stroke-width="5"/>'.format(
            X(2.75), X(3.05), Y(14.5 - 2.2 * k), Y(14.5 - 2.2 * k), shades[w]))
        out.append('<text x="{}" y="{}" font-size="17" fill="{}">빈틈 폭 {:g} mm</text>'.format(
            X(3.12), Y(14.5 - 2.2 * k) + 6, INK2, float(w)))
    out.append('<circle cx="{}" cy="{}" r="7" fill="{}"/>'.format(X(0), Y(r_on), INK))
    out.append('<text x="{}" y="{}" font-size="22" font-weight="700" fill="{}">+{:.1f}℃</text>'.format(X(0) + 16, Y(r_on) - 4, INK, r_on))
    out.append('<circle cx="{}" cy="{}" r="7" fill="{}"/>'.format(X(2.0), Y(r_off), INK))
    out.append('<text x="{}" y="{}" font-size="22" font-weight="700" fill="{}">+{:.1f}℃</text>'.format(X(2.0) - 10, Y(r_off) - 16, INK, r_off))
    out.append("</svg>")
    return "".join(out)


def section_svg():
    """반도체 칩/TIM/뚜껑 단면 모식도 (두께는 보기 좋게 과장). 쉬운 이름을 앞에, 전문 용어를 뒤에."""
    return """<svg viewBox="0 0 720 230" xmlns="http://www.w3.org/2000/svg" font-family="Pretendard, Malgun Gothic">
      <text x="230" y="18" font-size="15" fill="#6b6a66" text-anchor="middle">↑ 열이 나가는 길: 방열판 → 바깥 공기 25℃</text>
      <rect x="20" y="30" width="420" height="72" fill="#dcb79c"/>
      <rect x="20" y="102" width="420" height="26" fill="#ece3c4"/>
      <rect x="20" y="128" width="420" height="46" fill="#d6d5cf"/>
      <rect x="188" y="102" width="84" height="26" fill="#fcfcfb" stroke="#0b0b0b" stroke-width="2" stroke-dasharray="5 4"/>
      <rect x="213" y="178" width="34" height="12" fill="#eb6834"/>
      <text x="452" y="72" font-size="17" fill="#52514e">뚜껑 · 구리 1 mm (리드)</text>
      <text x="452" y="121" font-size="17" fill="#52514e">열 전달 페이스트 (TIM)</text>
      <text x="452" y="157" font-size="17" fill="#52514e">반도체 칩 · 실리콘 0.1 mm (다이)</text>
      <text x="230" y="96" font-size="16" fill="#0b0b0b" text-anchor="middle">공기 빈틈 (보이드)</text>
      <text x="230" y="212" font-size="16" fill="#b8471a" text-anchor="middle">가장 뜨거운 곳 (핫스팟)</text>
    </svg>"""


def bar_chart():
    rows = [("가우시안 과정 + 1D", gp, None, ACCENT), ("신경망 + 1D", nn, (nn_lo, nn_hi), "#2a78d6"),
            ("신경망 단독", nd, (NC["nn_direct"]["fresh_mae_min"], NC["nn_direct"]["fresh_mae_max"]), "#a8a7a1")]
    W, l, bh, gap = 640, 190, 34, 22
    H = len(rows) * (bh + gap) + 40
    xmax = 1.2
    X = lambda v: l + (W - l - 70) * v / xmax
    out = ['<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" font-family="Pretendard, Malgun Gothic">'.format(W, H)]
    for i, (name, v, rng, c) in enumerate(rows):
        y = 10 + i * (bh + gap)
        out.append('<text x="{}" y="{}" font-size="18" fill="{}" text-anchor="end">{}</text>'.format(l - 12, y + bh / 2 + 6, INK, name))
        out.append('<rect x="{}" y="{}" width="{:.1f}" height="{}" rx="4" fill="{}"/>'.format(l, y, X(v) - l, bh, c))
        if rng:
            out.append('<line x1="{:.1f}" x2="{:.1f}" y1="{}" y2="{}" stroke="{}" stroke-width="2"/>'.format(
                X(rng[0]), X(rng[1]), y + bh / 2, y + bh / 2, INK2))
        out.append('<text x="{:.1f}" y="{}" font-size="19" font-weight="700" fill="{}">{:.2f}℃</text>'.format(
            X(max(v, rng[1] if rng else v)) + 10, y + bh / 2 + 7, INK, v))
    out.append('<text x="{}" y="{}" font-size="14" fill="{}">새 조건 160건 평균 오차 · 가는 선은 신경망 시드 5개 범위</text>'.format(l, H - 6, MUTED))
    out.append("</svg>")
    return "".join(out)


# ── 본문 ──────────────────────────────────────────────────────

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>TIM 보이드 열 예측 포스터</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
<style>
@page {{ size: 420mm 594mm; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{ --ink:#0b0b0b; --ink2:#52514e; --muted:#6b6a66; --grid:#e1e0d9; --panel:#f3f2ee; --surface:#fcfcfb;
        --accent:#eb6834; --accent-text:#b8471a; --blue:#2a78d6; --ok:#0b7a0b; }}
html, body {{ width: 420mm; height: 594mm; background: var(--surface); color: var(--ink);
  font-family: Pretendard, 'Malgun Gothic', sans-serif; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.page {{ width: 420mm; height: 594mm; padding: 14mm 17mm 10mm; display: flex; flex-direction: column; gap: 4.5mm; overflow: hidden; }}
.mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}

header {{ display: grid; grid-template-columns: 1fr 38mm; gap: 10mm; align-items: end;
  border-bottom: 1.2mm solid var(--ink); padding-bottom: 5mm; }}
.kicker {{ font-size: 17pt; font-weight: 600; color: var(--accent-text); letter-spacing: .02em; }}
h1 {{ font-size: 50pt; line-height: 1.12; font-weight: 800; letter-spacing: -0.035em; margin: 3mm 0 4mm; }}
h1 em {{ font-style: normal; color: var(--accent-text); }}
.sub {{ font-size: 23pt; color: var(--ink2); font-weight: 500; letter-spacing: -0.01em; }}
.byline {{ font-size: 16pt; color: var(--ink2); margin-top: 5mm; }}
.byline b {{ color: var(--ink); }}
.qr {{ text-align: center; font-size: 11pt; color: var(--muted); }}
.qr #qr {{ width: 34mm; height: 34mm; margin: 0 auto 2mm; }}
.qr #qr img, .qr #qr canvas {{ width: 34mm !important; height: 34mm !important; }}

.row {{ display: grid; gap: 10mm; }}
.r1 {{ grid-template-columns: 1fr 1.2fr; }}
.r2 {{ grid-template-columns: 1.55fr 1fr; }}
.r4 {{ grid-template-columns: 1.25fr 1fr 1fr; }}
h2 {{ font-size: 23pt; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 3.5mm; display: flex; gap: 4mm; align-items: baseline; }}
h2 .n {{ font-family: 'JetBrains Mono', monospace; font-size: 17pt; color: var(--accent-text); font-weight: 700; }}
p, li {{ font-size: 14.5pt; line-height: 1.5; color: var(--ink2); }}
p b, li b {{ color: var(--ink); }}
ul {{ padding-left: 6mm; }} li {{ margin-bottom: 1.6mm; }}
.cap {{ font-size: 11.5pt; color: var(--muted); line-height: 1.5; margin-top: 2mm; }}
.big {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--ink); letter-spacing: -0.03em; }}
.panel {{ background: var(--panel); border-radius: 3mm; padding: 6mm 7mm; }}
.q {{ font-size: 17pt; font-weight: 700; color: var(--ink); line-height: 1.45; border-left: 1.4mm solid var(--accent);
  padding-left: 4mm; margin-top: 4mm; }}
svg {{ width: 100%; height: auto; display: block; }}
.sec-svg svg {{ width: 96%; }}

.shot {{ border: 0.5mm solid var(--grid); border-radius: 2.5mm; overflow: hidden; }}
.shot img {{ width: 100%; display: block; }}
.callouts {{ display: flex; flex-direction: column; gap: 4.5mm; }}
.call {{ display: grid; grid-template-columns: 11mm 1fr; gap: 3.5mm; align-items: start; }}
.call .k {{ width: 11mm; height: 11mm; border-radius: 50%; background: var(--ink); color: #fff; font-weight: 800;
  font-size: 15pt; display: flex; align-items: center; justify-content: center; }}
.call h3 {{ font-size: 16.5pt; font-weight: 700; margin-bottom: .8mm; }}
.call h3 span {{ font-size: 12pt; color: var(--accent-text); font-weight: 700; margin-left: 2mm; }}
.call p {{ font-size: 13.5pt; line-height: 1.45; }}

.ai {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6mm; }}
.card {{ background: var(--panel); border-radius: 3mm; padding: 5mm 5.5mm 4.5mm; display: flex; flex-direction: column; gap: 2.5mm;
  border-top: 2mm solid var(--accent); }}
.card.no {{ border-top-color: #a8a7a1; background: #f7f6f2; }}
.card .t {{ font-size: 18pt; font-weight: 800; }}
.card .role {{ font-size: 12.5pt; color: var(--muted); }}
.card .big {{ font-size: 32pt; line-height: 1.05; margin-top: 1mm; }}
.card .big small {{ font-size: 15pt; color: var(--ink2); font-weight: 500; margin-left: 1.5mm; letter-spacing: 0; }}
.card .m {{ font-size: 12.5pt; color: var(--ink2); line-height: 1.5; }}
.card .m b {{ color: var(--ink); }}
.badge {{ margin-top: auto; align-self: flex-start; font-size: 12.5pt; font-weight: 800; padding: 1.5mm 3.5mm; border-radius: 10mm;
  background: var(--ink); color: #fff; }}
.card.no .badge {{ background: #fff; color: var(--ink2); border: .4mm solid #a8a7a1; }}
.ai-wrap {{ display: grid; grid-template-columns: 1fr; gap: 5mm; }}
.lesson {{ counter-reset: l; list-style: none; padding: 0; }}
.lesson li {{ counter-increment: l; position: relative; padding-left: 11mm; margin-bottom: 3mm; }}
.lesson li::before {{ content: counter(l); position: absolute; left: 0; top: .6mm; width: 8mm; height: 8mm; border-radius: 50%;
  background: var(--accent); color: #fff; font-weight: 800; font-size: 13pt; display: flex; align-items: center; justify-content: center; }}
footer {{ margin-top: auto; display: flex; justify-content: space-between; font-size: 11.5pt; color: var(--muted);
  border-top: .4mm solid var(--grid); padding-top: 3.5mm; }}
</style></head>
<body><div class="page">

<header>
  <div>
    <div class="kicker">천안형 AX 액션러닝 · 개인 프로젝트</div>
    <h1>반도체 칩 속 공기 빈틈은<br>크기보다 <em>위치</em>가 더 중요하다</h1>
    <div class="sub">빈틈이 반도체 칩 온도를 얼마나 올리는지 <b style="color:var(--ink)">바로</b> 알려주는 웹 도구 · 열 계산 + AI</div>
    <div class="byline"><b>이희석</b> · 상명대학교 전자공학과 2학년 · 코드는 AI 코딩 도구(Claude Code)를 활용해 구현·검증</div>
  </div>
  <div class="qr"><div id="qr"></div>코드·데이터·웹 도구</div>
</header>

<div class="row r1">
  <section>
    <h2><span class="n">01</span>왜 만들었나</h2>
    <div class="sec-svg">{section_svg}</div>
    <p><b>반도체 칩은 뜨거워지면 느려지고 빨리 고장 납니다.</b> 열은 고르게 나지 않고 <b>가장 뜨거운 곳(핫스팟)</b>에 몰려 위의 뚜껑으로 빠져나갑니다.
       반도체 칩과 뚜껑 사이를 메우는 <b>열 전달 페이스트(TIM)</b>(컴퓨터의 서멀 구리스와 같은 역할)에 <b>공기 빈틈(보이드)</b>이 생기면,
       공기는 열을 거의 못 전해 길이 막힙니다.</p>
    <p style="margin-top:2.5mm">그런데 ① 정밀 계산(2D 열해석)은 1회 약 {sim_s:.0f}초라 빈틈 위치·크기 {cells:,}가지를 다 보려면 약 <b>{sim_min:.0f}분</b>,
       ② 업계 검사 기준(예: MIL-STD-883의 반도체 칩 접합 기준)은 빈틈의 <b>면적</b>만 봅니다.</p>
    <div class="q">그래서: 빈틈이 어디에 있을 때 위험한지<br>빠르고, 믿을 수 있게 알려주는 도구를 만들자</div>
  </section>
  <section>
    <h2><span class="n">02</span>발견: 같은 빈틈도 위치에 따라 100배 넘게 차이</h2>
    {void_chart}
    <p>같은 5 W 반도체 칩에서 폭 2 mm 빈틈이 핫스팟 <b>바로 위</b>면 최고 온도 <b>+{r_on:.1f}℃</b>, 중심이 <b>2 mm만 떨어져도 +{r_off:.1f}℃</b>.
       열은 핫스팟 바로 위로 몰려 나가서, 그 길을 막으면 크게 오르고 옆을 막으면 돌아갑니다.
       <b>차가 몰리는 차선을 막느냐, 빈 차선을 막느냐</b>의 차이로, 면적만 보는 기준은 이를 못 가립니다.</p>
    <div class="cap">계산 검증: 단순한 경우 논문의 정답 공식과 {grid_err} 이내 일치 (2D 유한차분, 0.025 mm 격자)</div>
  </section>
</div>

<section>
  <h2><span class="n">03</span>만든 것: 조건을 넣으면 반도체 칩 온도를 바로 알려주는 웹 도구</h2>
  <div class="row r2">
    <div class="shot"><img src="web_main.png" alt="웹 도구 화면"></div>
    <div class="callouts">
      <div class="call"><div class="k">1</div><div><h3>말로 조건 바꾸기<span>생성형 AI</span></h3>
        <p>"보이드 2 mm를 핫스팟 위에 넣어 봐"라고 쓰면 챗GPT 같은 AI(LLM)가 조건을 바꿉니다. 온도는 AI가 지어내지 않고 계산 도구가 냅니다.</p></div></div>
      <div class="call"><div class="k">2</div><div><h3>예측과 95% 범위<span>머신러닝</span></h3>
        <p>간단한 공식(1D 열저항)으로 먼저 계산하고, 공식이 못 잡는 <b>핫스팟·빈틈 효과만 AI가</b> 더합니다. "95% 확률로 이 사이" 범위도 함께, 넓으면 "자신 없음" 경고.</p></div></div>
      <div class="call"><div class="k">3</div><div><h3>온도를 낮추려면<span>설명 가능한 AI</span></h3>
        <p>무엇을 바꾸면 온도가 가장 많이 내려가는지 순위로 보여줍니다. 위 화면에서는 "보이드 없애기"(빈틈 없애기)가 1등.</p></div></div>
      <div class="call"><div class="k">4</div><div><h3>정답 확인</h3>
        <p>버튼 하나로 같은 조건을 정밀 계산(2D 해석)해 AI 답과 비교. 빈틈 {cells:,}가지 위험 지도를 AI는 <b>0.3초</b>에 그립니다 (정밀 계산이면 약 {sim_min:.0f}분).</p></div></div>
    </div>
  </div>
</section>

<section class="ai-wrap">
  <h2><span class="n">04</span>AI 4종을 다 넣지 않고, 재 보고 골랐다</h2>
  <div class="ai">
    <div class="card"><div class="t">머신러닝</div><div class="role">가우시안 과정 + 간단한 공식</div>
      <div class="big">{gp:.2f}<small>℃</small></div>
      <div class="m">학습에 안 쓴 <b>처음 보는 조건 160개</b>에서 정밀 계산과의 평균 차이</div>
      <div class="badge">사용 · 예측과 95% 범위</div></div>
    <div class="card no"><div class="t">딥러닝</div><div class="role">신경망 (은닉층 3개)</div>
      <div class="big">{nn:.2f}<small>℃</small></div>
      <div class="m">정확도는 비슷했지만 95% 범위가 이미 주는 것 외에 <b>새 정보가 없었음</b>. 공식 없이 AI만 쓰면 {nd:.2f}℃ ({ratio:.1f}배 나쁨)</div>
      <div class="badge">쓰지 않음 · 근거만 공개</div></div>
    <div class="card"><div class="t">설명 가능한 AI</div><div class="role">순열 중요도 · 한 단계 바꿔 보기</div>
      <div class="big">{top1}/{icn}</div>
      <div class="m">"무엇을 바꾸면 가장 시원해지나" 1등 추천이 정밀 계산과 일치 (변화 오차 {imae:.2f}℃). 방열판 효과는 공식 몫 → AI 몫 <b>{sink:.2f}℃</b></div>
      <div class="badge">사용 · 온도 낮추는 방법</div></div>
    <div class="card"><div class="t">생성형 AI</div><div class="role">LLM 에이전트 · 계산 도구 호출</div>
      <div class="big">{ec}/{ect}</div>
      <div class="m">말로 내린 조건 변경 명령을 맞게 처리. 질문 답하기 {ev}/{evt}, <b>AI가 계산 없이 지어낸 숫자 {made_up}개</b>. 범위 밖 요청은 거절</div>
      <div class="badge">사용 · 말로 조건 바꾸기</div></div>
  </div>
</section>

<div class="row r4">
  <section>
    <h2><span class="n">05</span>배운 것</h2>
    <ul class="lesson">
      <li><b>공식으로 풀리는 부분은 공식에, 나머지만 AI에게.</b> 공식을 바탕으로 주자 AI(신경망) 오차 {nd:.2f}→{nn:.2f}℃, 같은 정확도에 필요한 계산 데이터 {n_ai}→{n_hyb}건(초기 모델 기준).</li>
      <li><b>"평균적으로 괜찮다"가 위험을 가린다.</b> 빈틈 크기의 영향은 핫스팟과 겹칠 때 +{void_ov:.1f}℃, 아닐 때 +{void_rest:.1f}℃ (약 {void_x:.0f}배).</li>
      <li><b>AI는 많이 넣는 것보다 재 보고 고르는 것.</b> 측정해 보니 새 정보가 없던 딥러닝은 메인 화면에서 뺐습니다.</li>
    </ul>
  </section>
  <section class="panel">
    <h2>한계</h2>
    <ul>
      <li>단면(2D) 계산이라 빈틈이 앞뒤로 긴 띠가 됨 → 실제보다 영향이 크게 나옴</li>
      <li>실제 반도체 칩 측정 비교 없음 · 재료 성질은 온도와 무관 가정</li>
      <li>학습 범위 밖 예측 안 함 · AI 채점 질문은 직접 만든 문장</li>
    </ul>
  </section>
  <section class="panel">
    <h2>검증 요약</h2>
    <ul>
      <li>열 계산: 논문 정답 공식 대비 {grid_err} · 웹 도구 검사 {wv}/{wvt}</li>
      <li>무작위 값 고정(시드 42) → 누가 돌려도 같은 결과, 파일 공개</li>
      <li>합격 기준은 결과 전에 정함 (딥러닝은 사후에 허점을 찾아 다시 비교)</li>
    </ul>
  </section>
</div>

<footer><span>{github}</span><span>2026. 11 · 이희석 · 상명대학교 전자공학과</span></footer>
</div>
<script>
new QRCode(document.getElementById("qr"), {{ text: "{github}", width: 512, height: 512, correctLevel: QRCode.CorrectLevel.M }});
</script>
</body></html>
"""

same = DC["posthoc_same_count"]["h_alone"]
out = HTML.format(
    section_svg=section_svg(), void_chart=void_chart(), bar_chart=bar_chart(),
    sim_s=sur.sim_seconds, cells=cells, sim_min=sim_min, r_on=r_on, r_off=r_off, grid_err=GRID_ERR,
    gp=gp, nn=nn, nd=nd, ratio=nd / nn, same=same, nbig=DC["n_big"],
    top1=IC["top1_same"], icn=IC["n"], imae=IC["mae_delta"], sink=sink,
    ec=EC["correct"], ect=EC["total"], ev=EV["correct"], evt=EV["total"], made_up=made_up,
    n_ai=n_ai, n_hyb=n_hyb, void_ov=void_ov, void_rest=void_rest, void_x=void_ov / void_rest,
    wv=WV["passed"], wvt=WV["total"], github=GITHUB)
with open(os.path.join(HERE, "poster.html"), "w", encoding="utf-8") as f:
    f.write(out)
print("저장: poster/poster.html")
print("확인: 학습곡선 2.0℃ 도달 AI혼자 {} / 1D+AI {}, 위험지도 {}칸 {:.0f}분".format(n_ai, n_hyb, cells, sim_min))
