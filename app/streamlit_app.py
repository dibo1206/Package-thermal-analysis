"""TIM 보이드 열 예측 — 웹 화면.

  streamlit run app/streamlit_app.py      (저장소 루트에서. .streamlit/config.toml 테마를 읽는다)

계산은 전부 model.py 가 한다. 이 파일은 입력을 받고 결과를 그리기만 한다.
화면의 성능 숫자는 data/web_validation_<버전>.json (validate_backend.py 가 측정한 값) 에서 읽는다.

디자인
  사이드바·기본 탭·metric 상자 같은 Streamlit 기본 모양을 쓰지 않는다.
  결과 판독(큰 숫자) 바로 아래에 조건 패널을 두어, 손잡이를 돌리면서 숫자를 같이 본다.
  강조색은 열을 뜻하는 주황 하나. 버튼은 검정 바탕 (흰 글자와 주황의 대비가 3.2:1 이라 글자 버튼에 못 쓴다).
"""

import html as _html
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import model as M

st.set_page_config(page_title="TIM 보이드 열 예측", page_icon=":material/device_thermostat:",
                   layout="wide", initial_sidebar_state="collapsed")

GITHUB = "https://github.com/dibo1206/Package-thermal-analysis"

# 색 (dataviz 기본 팔레트의 표면·잉크 + 주황 하나)
SURFACE, PANEL = "#fcfcfb", "#f3f2ee"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#6b6a66", "#e1e0d9"
ACCENT, ACCENT_TEXT = "#eb6834", "#b8471a"
BLUE = "#2a78d6"                     # 그래프에서 두 번째 계열이 필요할 때만 (AI 예측 선)
CRITICAL = "#d03b3b"
HEAT = [[0.0, "#fdf1ea"], [0.25, "#f8c9ad"], [0.5, "#f19566"], [0.75, "#eb6834"], [1.0, "#8c2f0b"]]
SANS = "Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, monospace"


# ── 스타일 ────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#6b6a66; --grid:#e1e0d9; --panel:#f3f2ee;
        --surface:#fcfcfb; --accent:#eb6834; --accent-text:#b8471a; --critical:#d03b3b;
        --mono:'JetBrains Mono', ui-monospace, monospace; --r:6px; }

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family: Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important; }

/* Streamlit 기본 장식 숨기기 */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stStatusWidget"], footer, #MainMenu { display:none !important; }
[data-testid="stAppViewContainer"] { background: var(--surface); }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1160px; padding: 0 28px 64px !important; }
[data-testid="stVerticalBlock"] { gap: 0.9rem; }

/* 위젯 라벨과 값 */
[data-testid="stWidgetLabel"] p { font-size: 13px !important; color: var(--ink2) !important; font-weight: 500; }
[data-testid="stSliderThumbValue"] { color: var(--ink) !important; font-family: var(--mono) !important;
  font-size: 12px !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
  font-family: var(--mono) !important; font-size: 11px !important; color: var(--muted) !important; }
[data-testid="stNumberInput"] input { font-family: var(--mono) !important; }

/* 버튼: 검정 바탕 */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  background: var(--ink) !important; color: #fcfcfb !important; border: 1px solid var(--ink) !important;
  border-radius: var(--r) !important; padding: 0.45rem 1.1rem !important; font-weight: 600 !important;
  transition: transform .12s ease, background .12s ease; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
  background: #2a2a28 !important; }
[data-testid="stBaseButton-primary"]:active, [data-testid="stBaseButton-secondary"]:active {
  transform: translateY(1px); }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p { color: #fcfcfb !important; }

/* 보기 전환: 라디오를 밑줄형 내비게이션으로 */
.st-key-view [role="radiogroup"] { gap: 0 !important; border-bottom: 1px solid var(--grid); width: 100%;
  flex-wrap: wrap; }
.st-key-view [data-testid="stRadioOption"] { margin: 0 22px -1px 0 !important; padding: 10px 0 11px !important;
  cursor: pointer; border-bottom: 2px solid transparent; }
.st-key-view [data-testid="stRadioOption"] > div > div > div:first-child { display: none !important; }
.st-key-view [data-testid="stRadioOption"] p { font-size: 15px !important; color: var(--muted) !important;
  font-weight: 500; }
.st-key-view [data-testid="stRadioOption"]:hover p { color: var(--ink) !important; }
.st-key-view [data-testid="stRadioOption"][data-selected="true"] { border-bottom-color: var(--ink); }
.st-key-view [data-testid="stRadioOption"][data-selected="true"] p { color: var(--ink) !important; font-weight: 700; }

/* 조건 패널 */
.st-key-conditions { background: var(--panel); border-radius: var(--r); padding: 18px 22px 6px; }

/* 직접 그리는 블록들 */
.topbar { display:flex; align-items:center; justify-content:space-between; height:60px;
  border-bottom:1px solid var(--grid); margin-bottom: 28px; }
.brand { font-weight:700; font-size:16px; color:var(--ink); letter-spacing:-0.01em; }
.brand span { color: var(--accent-text); }
.toplinks a { color: var(--ink2); text-decoration:none; font-size:14px; margin-left:22px; }
.toplinks a:hover { color: var(--ink); text-decoration: underline; }

.read-label { font-size:13px; color:var(--ink2); font-weight:500; }
.read-big { font-family:var(--mono); font-size:76px; font-weight:500; line-height:1.0; color:var(--ink);
  letter-spacing:-0.04em; margin:8px 0 6px; }
.read-big small { font-size:30px; color:var(--ink2); margin-left:6px; letter-spacing:0; }
.read-band { font-family:var(--mono); font-size:15px; color:var(--ink2); }
.read-band b { color: var(--accent-text); font-weight:600; }
.read-sub { display:flex; gap:28px; margin-top:20px; }
.read-sub div { font-size:12.5px; color:var(--muted); }
.read-sub div b { display:block; font-family:var(--mono); font-size:19px; color:var(--ink); font-weight:500;
  margin-top:2px; }

.note { border-left:3px solid var(--accent); background: #fdf1ea; padding:10px 14px;
  border-radius: 0 var(--r) var(--r) 0; font-size:13.5px; color:var(--ink); margin-top:12px; line-height:1.55; }
.note.info { border-left-color: var(--ink2); background: var(--panel); }
.note.bad { border-left-color: var(--critical); background:#fbeaea; }

.comp { margin-top: 6px; }
.comp-bar { display:flex; height:14px; gap:2px; }
.comp-bar div { height:100%; }
.comp-bar div:first-child { border-radius: 4px 0 0 4px; }
.comp-bar div:last-child { border-radius: 0 4px 4px 0; }
.comp-legend { display:flex; flex-wrap:wrap; gap:4px 16px; justify-content:space-between; margin-top:8px; font-size:12.5px; color:var(--ink2); }
.comp-legend b { font-family:var(--mono); font-weight:500; color:var(--ink); }

.group-h { font-size:14px; font-weight:700; color:var(--ink); margin: 0 0 6px; }

.trio { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 24px; margin: 6px 0 4px; }
.trio div { font-size:12.5px; color:var(--muted); }
.trio div b { display:block; font-family:var(--mono); font-size:30px; font-weight:500; color:var(--ink);
  margin-top:4px; letter-spacing:-0.02em; }
.trio div b.ok { color:#0b7a0b; } .trio div b.bad { color: var(--critical); }

.answer { font-size:22px; font-weight:600; color:var(--ink); line-height:1.45; margin: 4px 0; }
.answer .num { font-family:var(--mono); color: var(--accent-text); }
.answer-sub { font-size:13.5px; color:var(--ink2); line-height:1.6; }

.tbl { width:100%; border-collapse:collapse; font-size:13px; }
.tbl th { text-align:left; font-weight:600; color:var(--ink2); padding:8px 10px; border-bottom:1px solid var(--ink); }
.tbl td { padding:7px 10px; color:var(--ink); }
.tbl td.n { font-family:var(--mono); text-align:right; }
.tbl th.n { text-align:right; }
.tbl tr:nth-child(even) td { background: var(--panel); }

.checks { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }
.check-card { background: var(--panel); border-radius: var(--r); padding: 16px 18px; }
.check-card .ch { font-size:14px; margin:0 0 10px; color:var(--ink); font-weight:700; }
.check-row { display:grid; grid-template-columns: 42px 1fr; gap:10px; padding:5px 0; font-size:13px; }
.check-row .tag { font-size:11.5px; font-weight:700; color:#0b7a0b; }
.check-row .tag.bad { color: var(--critical); }
.check-row .d { color: var(--muted); font-family: var(--mono); font-size: 11.5px; display:block; margin-top:2px; }

.score { font-family:var(--mono); font-size:44px; font-weight:500; color:var(--ink); letter-spacing:-0.03em; }
.lead { font-size:15px; color:var(--ink2); line-height:1.7; max-width: 68ch; }
.lead b { color: var(--ink); }
.cap { font-size:12.5px; color:var(--muted); line-height:1.6; }
.foot { border-top:1px solid var(--grid); margin-top:48px; padding-top:16px; font-size:12.5px; color:var(--muted);
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
.foot a { color: var(--ink2); }

.sec { font-size:20px; font-weight:700; color:var(--ink); margin: 8px 0 2px; letter-spacing:-0.01em; }

@media (max-width: 760px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 0 16px 48px !important; }
  .read-big { font-size: 56px; }
  .trio { grid-template-columns: 1fr; gap: 12px; }
  .checks { grid-template-columns: 1fr; }
  .toplinks a { margin-left: 14px; }
}
</style>
"""


def html(s):
    """HTML 조각을 그대로 그린다. 줄바꿈을 지워 마크다운 해석(들여쓰기 = 코드 블록)을 피한다."""
    st.markdown(" ".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def esc(s):
    return _html.escape(str(s))


def style(fig, height, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=12, color=INK2),
        hoverlabel=dict(bgcolor=INK, font=dict(family=MONO, size=12, color=SURFACE), bordercolor=INK),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=12, color=INK2)),
    )
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", tickfont=dict(family=MONO, size=11))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_resource(show_spinner="대리모델 준비 중")
def get_model():
    return M.Surrogate()


@st.cache_data
def get_validation():
    return M.load_validation()


st.markdown(CSS, unsafe_allow_html=True)
s = get_model()
V = get_validation()


# ── 머리 ──────────────────────────────────────────────────────

html("""
<div class="topbar">
  <div class="brand">TIM 보이드 <span>열 예측</span></div>
  <div class="toplinks"><a href="?view=method" target="_self">방법과 검증</a><a href="{}" target="_blank" rel="noopener">GitHub</a></div>
</div>
""".format(GITHUB))


# ── 입력 (조건 패널은 판독 아래에 그리지만 값은 먼저 읽는다) ──

def _options(lo, hi, step, n=None):
    if n is None:
        vals = np.arange(round(lo / step), round(hi / step) + 1) * step
    else:
        vals = np.unique(np.round(np.geomspace(lo, hi, n) / step)) * step
    return [round(float(v), 4) for v in vals]


SPOT_OPTIONS = sorted(set(_options(0.25, 10.0, 0.05, n=40)) | {1.0})
TIM_OPTIONS = _options(*M.RANGES["tim_mm"], M.H_MM)

top = st.container()
panel = st.container(key="conditions")

with panel:
    html('<div class="group-h">조건</div>')
    r1 = st.columns(4, gap="medium")
    q_w = r1[0].slider("총 발열 (W)", *M.RANGES["q_w"], float(M.DEFAULT["q_w"]), 0.1)
    spot = r1[1].select_slider("핫스팟 폭 (mm, 다이 가운데)", SPOT_OPTIONS, value=1.0)
    tim_mm = r1[2].select_slider("TIM 두께 (mm)", TIM_OPTIONS, value=round(M.DEFAULT["tim_mm"], 4))
    tim_k = r1[3].slider("TIM 열전도율 (W/m·K)", *M.RANGES["tim_k"], float(M.DEFAULT["tim_k"]), 0.1)

    r2 = st.columns(4, gap="medium")
    r_sink = r2[0].slider("방열판 저항 (K/W)", *M.RANGES["r_sink"], float(M.DEFAULT["r_sink"]), 0.05)
    with r2[1]:
        st.write("")
        has_void = st.toggle("TIM 보이드 넣기", value=False)
    if has_void:
        void_mm = r2[2].slider("보이드 폭 (mm)", *M.RANGES["void_mm"], 1.0, M.H_MM)
        lo, hi = void_mm / 2.0, M.WIDTH_MM - void_mm / 2.0
        # 처음 값은 3mm. 폭이 바뀌어 위치가 범위를 벗어나면 다이 안으로 옮긴다
        current = st.session_state.get("void_center", 3.0)
        st.session_state.void_center = float(min(max(current, lo), hi))
        void_center = r2[3].slider("보이드 위치 (mm, 핫스팟 중심 = 5)", float(lo), float(hi),
                                   step=M.H_MM, key="void_center")
    else:
        void_mm, void_center = 0.0, 0.0
        with r2[2]:
            html('<div class="cap" style="padding-top:26px">보이드를 넣으면 폭과 위치를 고를 수 있습니다.</div>')
    html('<div class="cap" style="margin:-4px 0 10px">범위는 학습 데이터를 만든 범위와 같습니다. 이 밖은 예측하지 않습니다.</div>')

inputs = {"q_w": q_w, "spot_mm": spot, "tim_mm": tim_mm, "tim_k": tim_k, "r_sink": r_sink,
          "void_mm": void_mm, "void_center_mm": void_center}
try:
    pred = s.predict(inputs)
except ValueError as e:                                # 슬라이더로는 생길 수 없지만 막아둔다
    st.error(str(e))
    st.stop()
cond = pred["cond"]


# ── 판독 + 단면 ───────────────────────────────────────────────

def group_text(name):
    if not V or name not in V.get("groups", {}):
        return ""
    g = V["groups"][name]
    return "검증 {}건에서 평균 오차 {:.2f}℃, 최대 {:.1f}℃.".format(g["n"], g["mae"], g["max"])


def cross_section(c):
    """다이/TIM/리드 단면 모식도. 실제 두께 비율은 0.10 : 0.05 : 1.00mm 라 보기 좋게 과장했다."""
    fig = go.Figure()
    bands = [("다이 Si", 0.0, 1.0, "#d6d5cf"), ("TIM", 1.0, 1.55, "#ece3c4"), ("리드 Cu", 1.55, 3.1, "#dcb79c")]
    for name, y0, y1, color in bands:
        fig.add_shape(type="rect", x0=0, x1=M.WIDTH_MM, y0=y0, y1=y1, fillcolor=color,
                      line=dict(width=0), layer="below")
        fig.add_annotation(x=M.WIDTH_MM + 0.18, y=(y0 + y1) / 2, text=name, showarrow=False,
                           xanchor="left", font=dict(size=12, color=INK2))
    x0 = M.SPOT_CENTER_MM - c["spot_mm"] / 2
    fig.add_shape(type="rect", x0=x0, x1=x0 + c["spot_mm"], y0=-0.32, y1=-0.04,
                  fillcolor=ACCENT, line=dict(width=0))
    fig.add_annotation(x=M.SPOT_CENTER_MM, y=-0.42, text="발열 {:.2f}mm".format(c["spot_mm"]),
                       showarrow=False, yanchor="top", font=dict(size=12, color=ACCENT_TEXT))
    if c["void_mm"] > 0:
        v0 = c["void_center_mm"] - c["void_mm"] / 2
        fig.add_shape(type="rect", x0=v0, x1=v0 + c["void_mm"], y0=1.0, y1=1.55,
                      fillcolor=SURFACE, line=dict(width=1.2, color=INK, dash="dot"))
        fig.add_annotation(x=c["void_center_mm"], y=1.62, text="보이드 {:.2f}mm".format(c["void_mm"]),
                           showarrow=False, yanchor="bottom", font=dict(size=12, color=INK))
    fig.add_annotation(x=M.WIDTH_MM / 2, y=3.35, text="윗면: 방열판 {:.2f} K/W → 공기 25℃".format(c["r_sink"]),
                       showarrow=False, font=dict(size=11.5, color=MUTED))
    style(fig, 250)
    fig.update_xaxes(range=[-0.2, M.WIDTH_MM + 1.5], title=None, showgrid=False,
                     tickvals=[0, 2.5, 5, 7.5, 10], ticktext=["0", "2.5", "5 mm", "7.5", "10"])
    fig.update_yaxes(range=[-1.0, 3.6], visible=False)
    return fig


with top:
    left, right = st.columns([5, 7], gap="large")
    with left:
        lo95, hi95 = pred["t_max"] - pred["half95"], pred["t_max"] + pred["half95"]
        html("""
        <div class="read-label">칩 최고 온도, AI 예측</div>
        <div class="read-big">{:.1f}<small>℃</small></div>
        <div class="read-band">95% 범위 <b>{:.1f} ~ {:.1f} ℃</b></div>
        <div class="read-sub">
          <div>1D 공식<b>{:.1f} ℃</b></div>
          <div>핫스팟·보이드 몫<b>+{:.1f} ℃</b></div>
        </div>
        """.format(pred["t_max"], lo95, hi95, pred["t_1d"], pred["extra"]))
        if pred["covers_hotspot"]:
            html('<div class="note">보이드가 핫스팟을 덮고 있습니다. 오차가 가장 큰 영역입니다. {} '
                 '아래 <b>정답 확인</b>으로 실제 해석과 비교해 보세요.</div>'.format(
                     esc(group_text("보이드가 핫스팟을 덮음"))))
        if pred["uncertain"]:
            html('<div class="note">AI가 자신 없어하는 조건입니다. 95% 범위가 ±{:.1f}℃보다 넓습니다. {}</div>'.format(
                s.flag_half, esc(group_text("자신 없음 경고"))))
    with right:
        chart(cross_section(cond))
        rise_1d = pred["t_1d"] - M.T_AIR
        extra = max(pred["extra"], 0.0)
        total = rise_1d + extra + pred["half95"]
        html("""
        <div class="comp">
          <div class="comp-bar">
            <div style="width:{:.2f}%; background:#d6d5cf" title="1D 공식이 계산하는 상승"></div>
            <div style="width:{:.2f}%; background:{}" title="AI가 학습한 핫스팟·보이드 몫"></div>
            <div style="width:{:.2f}%; background:repeating-linear-gradient(135deg,{} 0 2px,transparent 2px 5px)"
                 title="95% 범위"></div>
          </div>
          <div class="comp-legend">
            <span>공기 25℃에서 <b>+{:.1f}</b> 1D 열저항</span>
            <span><b>+{:.1f}</b> 핫스팟·보이드 (AI)</span>
            <span><b>±{:.1f}</b> 범위</span>
          </div>
        </div>
        """.format(100 * rise_1d / total, 100 * extra / total, ACCENT, 100 * pred["half95"] / total, ACCENT,
                   rise_1d, pred["extra"], pred["half95"]))


# ── 보기 전환 ─────────────────────────────────────────────────

st.write("")
VIEWS = ["정답 확인", "보이드 위험 지도", "허용 보이드 크기", "검증과 방법"]
if st.query_params.get("view") == "method":           # 머리의 '방법과 검증' 링크
    st.session_state.view = VIEWS[3]
    del st.query_params["view"]
view = st.radio("보기", VIEWS, horizontal=True, label_visibility="collapsed", key="view")


# 보기 1: 정답 확인

if view == "정답 확인":
    html('<p class="lead">같은 조건으로 <b>실제 2D 해석</b>(격자 46 × 400칸)을 돌려 AI 예측과 비교합니다. 1~4초 걸립니다.</p>')
    if st.button("2D 해석 실행", type="primary"):
        with st.spinner("2D 해석 중"):
            out = M.run_2d(cond)
        err = round(pred["t_max"] - out["t_max"], 2) + 0.0
        row = {
            "발열 W": cond["q_w"], "핫스팟 mm": cond["spot_mm"], "TIM mm": cond["tim_mm"],
            "TIM k": round(cond["tim_k"], 2), "방열판 K/W": round(cond["r_sink"], 2),
            "보이드 mm": cond["void_mm"], "위치 mm": round(cond["void_center_mm"], 3),
            "AI 예측 ℃": round(pred["t_max"], 2), "± ℃": round(pred["half95"], 2),
            "실제 ℃": round(out["t_max"], 2), "오차 ℃": err,
            "범위 안": abs(pred["t_max"] - out["t_max"]) <= pred["half95"],
            "해석 초": round(out["seconds"], 2),
        }
        st.session_state.setdefault("history", []).append(row)
        st.session_state.last_run = {"cond": cond, "out": out, "row": row}

    last = st.session_state.get("last_run")
    if last and last["cond"] == cond:
        out, row = last["out"], last["row"]
        inside = row["범위 안"]
        html("""
        <div class="trio">
          <div>실제 최고 온도, 2D 해석<b>{:.2f} ℃</b></div>
          <div>AI 오차 (예측 − 실제)<b>{:+.2f} ℃</b></div>
          <div>95% 범위 안에 들었나<b class="{}">{}</b></div>
        </div>
        <div class="cap">해석 {:.2f}초, 반복 {}회. 범위는 ±{:.2f}℃.</div>
        """.format(out["t_max"], row["오차 ℃"], "ok" if inside else "bad", "예" if inside else "아니오",
                   out["seconds"], out["iters"], row["± ℃"]))

        T = out["T"]
        x = (np.arange(T.shape[1]) + 0.5) * M.H_MM
        y = (np.arange(T.shape[0]) + 0.5) * M.H_MM
        a, b = st.columns([7, 5], gap="large")
        with a:
            html('<div class="sec">단면 온도</div><div class="cap">점선은 다이/TIM, TIM/리드 경계. 세로는 실제 두께 그대로라 얇게 보입니다.</div>')
            heat = go.Figure(go.Heatmap(z=T, x=x, y=y, colorscale=HEAT,
                                        colorbar=dict(title=dict(text="℃", font=dict(size=11)), thickness=10,
                                                      outlinewidth=0, tickfont=dict(family=MONO, size=10)),
                                        hovertemplate="%{x:.2f}mm, 높이 %{y:.3f}mm<br>%{z:.2f}℃<extra></extra>"))
            for yy in (0.10, 0.10 + cond["tim_mm"]):
                heat.add_hline(y=yy, line=dict(color=INK, width=1, dash="dot"))
            style(heat, 260)
            heat.update_xaxes(title=dict(text="가로 위치 (mm)", font=dict(size=12)))
            heat.update_yaxes(title=dict(text="높이 (mm)", font=dict(size=12)))
            chart(heat)
        with b:
            html('<div class="sec">다이 바닥 온도</div><div class="cap">발열이 나는 줄. 최고 온도가 여기서 나옵니다.</div>')
            prof = go.Figure()
            prof.add_trace(go.Scatter(x=x, y=out["surface"], mode="lines", line=dict(color=ACCENT, width=2),
                                      name="실제 해석", hovertemplate="%{x:.2f}mm: %{y:.2f}℃<extra></extra>"))
            prof.add_hline(y=pred["t_max"], line=dict(color=BLUE, width=1.5, dash="dash"))
            prof.add_annotation(x=0.2, y=pred["t_max"], text="AI 예측 최고 {:.2f}℃".format(pred["t_max"]),
                                showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11.5, color=INK2))
            style(prof, 260)
            prof.update_xaxes(title=dict(text="가로 위치 (mm)", font=dict(size=12)))
            prof.update_yaxes(title=dict(text="℃", font=dict(size=12)))
            chart(prof)
    elif last:
        html('<div class="note info">조건이 바뀌었습니다. 다시 실행하면 새 조건으로 해석합니다.</div>')

    hist = st.session_state.get("history", [])
    if hist:
        cols = ["발열 W", "핫스팟 mm", "TIM mm", "TIM k", "보이드 mm", "위치 mm", "AI 예측 ℃", "실제 ℃", "오차 ℃"]
        head = "".join('<th class="n">{}</th>'.format(c) for c in cols) + "<th>범위 안</th>"
        body = "".join(
            "<tr>" + "".join('<td class="n">{:g}</td>'.format(h[c]) for c in cols)
            + "<td>{}</td></tr>".format("예" if h["범위 안"] else "아니오")
            for h in reversed(hist))
        errs = np.abs([h["오차 ℃"] for h in hist])
        html('<div class="sec">이번 접속에서 확인한 기록</div>'
             '<div class="cap">{}건, 평균 오차 {:.2f}℃, 범위 안 {}건. 새로고침하면 사라집니다.</div>'.format(
                 len(hist), errs.mean(), sum(h["범위 안"] for h in hist)))
        html('<div style="overflow-x:auto"><table class="tbl"><tr>{}</tr>{}</table></div>'.format(head, body))


# 보기 2: 위험 지도

elif view == "보이드 위험 지도":
    html('<p class="lead">보이드가 <b>어디에, 얼마나 크게</b> 생기면 온도가 오르는지. '
         '보이드 외의 조건은 위 패널 값을 따릅니다.</p>')
    t0 = time.time()
    rm = s.risk_map(cond)
    took = time.time() - t0

    dT, hw = rm["dT"], rm["half95"]
    clear = np.where(dT > hw, dT, np.nan)                       # AI 범위보다 확실히 큰 영향
    small = np.where(np.isfinite(dT) & ~(dT > hw), 0.0, np.nan)  # 영향이 범위 이하
    pos, sizes = rm["positions"], rm["sizes"]
    custom = np.dstack([np.nan_to_num(dT), np.nan_to_num(hw)])

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=small, x=pos, y=sizes, colorscale=[[0, "#ecebe6"], [1, "#ecebe6"]],
                             showscale=False, customdata=custom, name="영향 작음",
                             hovertemplate="위치 %{x:.2f}mm, 폭 %{y:.2f}mm<br>"
                                           "%{customdata[0]:+.2f}℃, 범위 ±%{customdata[1]:.2f}℃ 이하"
                                           "<extra>영향 작음</extra>"))
    fig.add_trace(go.Heatmap(z=clear, x=pos, y=sizes, colorscale=HEAT, zmin=0,
                             colorbar=dict(title=dict(text="상승 ℃", font=dict(size=11)), thickness=10,
                                           outlinewidth=0, tickfont=dict(family=MONO, size=10)),
                             customdata=custom,
                             hovertemplate="위치 %{x:.2f}mm, 폭 %{y:.2f}mm<br>"
                                           "+%{z:.2f}℃ (±%{customdata[1]:.2f})<extra></extra>"))
    unc = np.argwhere(rm["uncertain"])
    if len(unc):
        fig.add_trace(go.Scatter(x=pos[unc[:, 1]], y=sizes[unc[:, 0]], mode="markers",
                                 marker=dict(symbol="x-thin", size=6, line=dict(width=1, color=INK2)),
                                 name="AI 자신 없음", hoverinfo="skip"))
    fig.add_vrect(x0=M.SPOT_CENTER_MM - cond["spot_mm"] / 2, x1=M.SPOT_CENTER_MM + cond["spot_mm"] / 2,
                  line=dict(color=INK, width=1, dash="dot"), fillcolor="rgba(0,0,0,0)",
                  annotation_text="핫스팟", annotation_position="top",
                  annotation_font=dict(size=11.5, color=INK))
    if cond["void_mm"] > 0:
        fig.add_trace(go.Scatter(x=[cond["void_center_mm"]], y=[cond["void_mm"]], mode="markers",
                                 marker=dict(symbol="diamond", size=12, color=INK, line=dict(width=2, color=SURFACE)),
                                 name="지금 넣은 보이드", hovertemplate="지금 넣은 보이드<extra></extra>"))
    style(fig, 460, legend=True)
    fig.update_xaxes(title=dict(text="보이드 위치 (mm, 다이 왼쪽 끝 기준)", font=dict(size=12)))
    fig.update_yaxes(title=dict(text="보이드 폭 (mm)", font=dict(size=12)))
    chart(fig)

    a, b = st.columns(2, gap="large")
    with a:
        html("""<div class="trio" style="grid-template-columns:1fr 1fr">
          <div>AI로 계산<b>{:.2f} 초</b></div>
          <div>2D 해석이라면<b>약 {:.0f} 분</b></div></div>
          <div class="cap">{:,}칸. 해석 1회 평균 {:.1f}초로 계산한 값입니다.</div>""".format(
            took, rm["n_cells"] * s.sim_seconds / 60, rm["n_cells"], s.sim_seconds))
    with b:
        txt = ("회색 칸은 예측된 상승이 AI의 95% 범위보다 작아 영향을 구분할 수 없는 곳입니다. "
               "보이드는 온도를 낮출 수 없는데도 이런 칸에서는 음수 예측이 나올 수 있어 0으로 칠했습니다.")
        if V and "map_check" in V:
            mc = V["map_check"]
            txt += (" 기준 조건에서 무작위 30칸을 실제 해석과 비교했을 때 95% 범위 적중 {:.0%}, "
                    "온도 상승 오차 평균 {:.2f}℃였습니다.".format(mc["coverage"], mc["mae_dT"]))
        html('<div class="cap">{}</div>'.format(esc(txt)))


# 보기 3: 허용 보이드

elif view == "허용 보이드 크기":
    html('<p class="lead">최고 온도를 한계 아래로 지키려면 <b>이 위치의 보이드는 몇 mm까지 허용되나.</b> '
         '평균 예측이 아니라 95% 범위의 위쪽 끝으로 판단합니다. 나머지 조건은 위 패널 값을 따릅니다.</p>')
    i1, i2 = st.columns([1, 2], gap="large")
    limit = i1.number_input("온도 한계 (℃)", min_value=26.0, max_value=300.0, value=45.0, step=1.0)
    center = i2.slider("보이드 위치 (mm)", 0.0, M.WIDTH_MM, M.SPOT_CENTER_MM, 0.05, key="inv_center")

    ans = s.allowable_void(cond, center, limit)
    nv = ans["no_void"]
    if ans["status"] == "none":
        html('<div class="answer">보이드가 없어도 한계를 넘을 수 있습니다.</div>'
             '<div class="answer-sub">보이드 없음 예측 {:.1f}℃, 위쪽 끝 {:.1f}℃. 한계를 올리거나 다른 조건을 바꿔야 합니다.</div>'.format(
                 nv["t_max"], nv["upper"]))
    elif ans["status"] == "below_min":
        html('<div class="answer">학습 범위의 가장 작은 보이드(<span class="num">{:.2f} mm</span>)부터 한계를 넘습니다.</div>'
             '<div class="answer-sub">위쪽 끝 {:.1f}℃. 그보다 작은 보이드는 학습하지 않아 답할 수 없습니다.</div>'.format(
                 ans["first_over"]["size_mm"], ans["first_over"]["upper"]))
    elif ans["status"] == "all":
        if ans["max_checked_mm"] < M.RANGES["void_mm"][0]:
            html('<div class="answer">다이 가장자리라 학습 범위의 보이드가 들어갈 수 없는 위치입니다.</div>')
        else:
            html('<div class="answer">넣을 수 있는 가장 큰 보이드(<span class="num">{:.2f} mm</span>)까지 한계 안입니다.</div>'
                 '<div class="answer-sub">위쪽 끝 {:.1f}℃. 학습 범위(4mm)를 넘는 크기는 추정하지 않습니다.</div>'.format(
                     ans["size_mm"], ans["upper"]))
    else:
        html('<div class="answer">허용 보이드 폭 <span class="num">{:.3f} mm</span> 까지</div>'
             '<div class="answer-sub">이때 예측 {:.1f}℃, 위쪽 끝 {:.2f}℃로 한계 {:g}℃ 이하. '
             '{:.3f}mm부터 위쪽 끝이 {:.2f}℃로 한계를 넘습니다.</div>'.format(
                 ans["size_mm"], ans["t_max"], ans["upper"], limit,
                 ans["first_over"]["size_mm"], ans["first_over"]["upper"]))

    if "curve" in ans:
        cv = ans["curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cv["sizes"], y=cv["t_max"], mode="lines", name="AI 예측",
                                 line=dict(color=BLUE, width=2),
                                 hovertemplate="폭 %{x:.3f}mm: %{y:.2f}℃<extra>AI 예측</extra>"))
        fig.add_trace(go.Scatter(x=cv["sizes"], y=cv["upper"], mode="lines", name="95% 범위 위쪽 끝",
                                 line=dict(color=ACCENT, width=2),
                                 hovertemplate="폭 %{x:.3f}mm: %{y:.2f}℃<extra>위쪽 끝</extra>"))
        fig.add_hline(y=limit, line=dict(color=INK, width=1, dash="dash"))
        fig.add_annotation(x=cv["sizes"][0], y=limit, text="한계 {:.0f}℃".format(limit), showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11.5, color=INK))
        if ans["status"] in ("size", "all") and ans["size_mm"] > 0:
            fig.add_vline(x=ans["size_mm"], line=dict(color=MUTED, width=1, dash="dot"))
        style(fig, 330, legend=True)
        fig.update_xaxes(title=dict(text="보이드 폭 (mm)", font=dict(size=12)))
        fig.update_yaxes(title=dict(text="최고 온도 (℃)", font=dict(size=12)))
        chart(fig)

    if ans["status"] in ("size", "all") and ans["size_mm"] > 0:
        if st.button("이 답을 실제 2D 해석으로 확인", type="primary"):
            with st.spinner("2D 해석 중"):
                out = M.run_2d(ans["cond"])
            ok = out["t_max"] <= limit
            html('<div class="note {}">보이드 {:.3f}mm의 실제 최고 온도 {:.2f}℃. 한계 {:.0f}℃ {} (여유 {:+.2f}℃)</div>'.format(
                "info" if ok else "bad", ans["size_mm"], out["t_max"], limit,
                "이하입니다." if ok else "를 넘었습니다.", limit - out["t_max"]))
    if V and "inverse" in V:
        iv = V["inverse"]
        html('<div class="cap" style="margin-top:10px">검증: 무작위 {}건에서 답한 크기로 실제 해석한 결과 {}건이 한계 이하, '
             '여유 평균 {:.2f}℃.</div>'.format(iv["n_checked"], iv["n_safe"], iv["margin_mean"]))


# 보기 4: 검증과 방법

else:
    a, b = st.columns([5, 7], gap="large")
    with a:
        if V:
            html('<div class="read-label">웹 도구 검증</div><div class="score">{} / {}</div>'
                 '<div class="cap">{} 측정. 합격 기준은 결과를 보기 전에 정했습니다.</div>'.format(
                     V["passed"], V["total"], V["date"]))
        else:
            html('<div class="note info">검증 결과 파일이 없습니다. python app/validate_backend.py 를 실행하세요.</div>')
    with b:
        html("""<p class="lead">다이(Si 0.1mm), TIM, 리드(Cu 1mm) 단면을 0.025mm 격자로 나눈 <b>2D 열해석</b>이 정답 역할을 합니다.
        해석기는 발표된 확산 저항 해석해와 0.03% 이내로 일치합니다. 조건 7개를 바꾼 해석 800건으로
        <b>최고 온도 = 1D 공식 + 발열 × AI가 배운 1W당 추가 온도</b> 모델을 학습했습니다.</p>""")

    if V:
        groups = [("같은 답을 내나", [0, 1, 9]), ("입력을 지키나", [2, 3, 4, 5]),
                  ("얼마나 맞나", [6, 10]), ("빠르고 안전한가", [7, 8, 11])]
        cards = []
        for title, idx in groups:
            rows = "".join(
                '<div class="check-row"><span class="tag{}">{}</span><span>{}<span class="d">{}</span></span></div>'.format(
                    "" if V["checks"][i]["ok"] else " bad", "통과" if V["checks"][i]["ok"] else "실패",
                    esc(V["checks"][i]["name"]), esc(V["checks"][i]["detail"]))
                for i in idx if i < len(V["checks"]))
            cards.append('<div class="check-card"><div class="ch">{}</div>{}</div>'.format(title, rows))
        html('<div class="checks">{}</div>'.format("".join(cards)))

        html('<div class="sec">조건별 오차</div><div class="cap">검증 160건. 이 화면이 쓰는 모델 기준.</div>')
        head = '<th>그룹</th><th class="n">건수</th><th class="n">평균 오차 ℃</th><th class="n">최대 ℃</th><th class="n">범위 적중</th>'
        body = "".join('<tr><td>{}</td><td class="n">{}</td><td class="n">{:.2f}</td><td class="n">{:.2f}</td>'
                       '<td class="n">{:.0%}</td></tr>'.format(esc(k), g["n"], g["mae"], g["max"], g["coverage"])
                       for k, g in V["groups"].items())
        html('<table class="tbl"><tr>{}</tr>{}</table>'.format(head, body))

    if M.os.path.exists(M.SETTINGS_V2):
        with open(M.SETTINGS_V2, encoding="utf-8") as f:
            v2 = M.json.load(f)
        t1, t2_ = v2["test"]["v1 (논문 설정)"], v2["test"]["v2"]
        html('<div class="sec">모델 v1과 v2</div>'
             '<p class="lead">v1(논문 모델)은 보이드의 폭과 핫스팟 중심과의 거리만 알았습니다. '
             '보이드가 핫스팟에 닿는 순간 온도가 꺾여 오르는데, v1은 이 꺾임을 매끄럽게 뭉갰습니다. '
             'v2는 <b>보이드가 핫스팟을 얼마나 덮는지</b>를 조건 정보로 더했습니다.</p>')
        head = ('<th>모델</th><th class="n">교차검증 평균 ℃</th><th class="n">검증 평균 ℃</th>'
                '<th class="n">검증 최대 ℃</th><th class="n">핫스팟 덮음 평균 ℃</th>')
        body = "".join('<tr><td>{}</td><td class="n">{:.2f}</td><td class="n">{:.2f}</td><td class="n">{:.2f}</td>'
                       '<td class="n">{:.2f}</td></tr>'.format(name, cvv["mae"], t["mae"], t["max"], t["covers_mae"])
                       for name, cvv, t in [("v1 (논문)", v2["cv"]["v1"], t1), ("v2", v2["cv"]["v2"], t2_)])
        html('<table class="tbl"><tr>{}</tr>{}</table>'.format(head, body))
        html('<div class="cap">설정은 학습 640건 교차검증으로만 골랐고 검증 160건은 마지막에 한 번 채점했습니다. '
             '다만 v2의 착안은 검증셋 오차 분석을 본 뒤에 나왔습니다.</div>')

    html('<div class="sec">연구 그림</div>')
    figs = [("validation_spreading.png", "외부 검증. 발표된 해석해와 비교"),
            ("learning_curve.png", "학습 곡선. 1D를 결합하면 필요한 데이터가 절반 (v1 기준)"),
            ("error_analysis.png", "오차 분석. 보이드가 핫스팟을 덮을 때 오차가 몰린다 (v1 기준)")]
    gcols = st.columns(3, gap="medium")
    for col, (name, cap) in zip(gcols, figs):
        col.image(M.os.path.join(M.ROOT, "figures", name), width="stretch")
        with col:
            html('<div class="cap">{}</div>'.format(esc(cap)))

    html('<div class="sec">한계</div><p class="lead">2차원 단면 해석이라 보이드가 깊이 방향으로 이어진 띠로 계산되어, '
         '실제 원형 보이드보다 영향이 크게 나옵니다. 실측과 비교하지 않았고, 재료 물성은 온도에 따라 변하지 않는다고 '
         '가정했습니다. 학습 범위 밖 조건은 예측하지 않습니다.</p>')


# ── 바닥 ──────────────────────────────────────────────────────

html("""
<div class="foot">
  <span>2차원 단면 해석 기준이며 실측과 비교하지 않았습니다.</span>
  <span>AI 코딩 도구(Claude Code)의 도움을 받아 작성. <a href="{}" target="_blank" rel="noopener">코드와 데이터</a></span>
</div>
""".format(GITHUB))
