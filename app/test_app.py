"""화면 동작 검사 — 브라우저 없이 Streamlit 앱을 실행해 입력을 바꾸고 버튼을 눌러본다.

  python app/test_app.py

확인하는 것
  - 기본 조건으로 열면 오류 없이 그려지고, 사이드바가 없고, 보기 네 개를 모두 열 수 있다
  - 보이드를 켜고 핫스팟 위로 옮기면 경고가 뜬다
  - 정답 확인 버튼이 해석을 돌리고 기록을 남긴다
  - 보이드 폭을 키워 위치가 범위를 벗어나도 오류 없이 안으로 옮겨진다
  - 허용 보이드 탭의 확인 버튼이 결과를 낸다
  - 화면의 숫자가 model.py 를 직접 부른 값과 같다
"""

import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from streamlit.testing.v1 import AppTest

import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "streamlit_app.py")
TIMEOUT = 120

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("  [{}] {}{}".format("OK" if ok else "실패", name, "  — " + detail if detail else ""), flush=True)


def no_exception(at, step):
    errs = [e.value for e in at.exception]
    check("{}: 예외 없음".format(step), not errs, "; ".join(map(str, errs))[:300])


def texts(at):
    out = [m.value for m in at.markdown] + [c.value for c in at.caption]
    out += [w.value for w in at.warning] + [w.value for w in at.success] + [w.value for w in at.error]
    out += [str(m.value) for m in at.metric]
    return "\n".join(out)


def readout(at, value):
    """큰 숫자 판독이 model.py 값과 같은가."""
    return '<div class="read-big">{:.1f}<small>'.format(value) in texts(at)


def main():
    print("화면 동작 검사\n")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    no_exception(at, "처음 열기")
    view = at.radio(key="view")
    check("보기 4개", len(view.options) == 4, str(view.options))
    check("사이드바를 쓰지 않는다", len(at.sidebar.children) == 0)

    s = M.Surrogate()
    ref = s.predict(M.DEFAULT)
    check("기본 조건 예측값이 model.py 와 같다", readout(at, ref["t_max"]), "{:.2f}".format(ref["t_max"]))

    # 보이드 켜기 → 핫스팟 위로
    at.toggle[0].set_value(True).run()
    no_exception(at, "보이드 켜기")
    void_slider = next(sl for sl in at.slider if sl.label.startswith("보이드 폭"))
    pos_slider = next(sl for sl in at.slider if sl.key == "void_center")
    check("보이드 슬라이더가 나타난다", void_slider is not None and pos_slider is not None)
    pos_slider.set_value(5.0).run()
    no_exception(at, "보이드를 핫스팟 위로")
    check("핫스팟 덮음 경고", "핫스팟을 덮고" in texts(at))

    ref = s.predict(dict(M.DEFAULT, void_mm=1.0, void_center_mm=5.0))
    check("보이드 조건 예측값이 model.py 와 같다", readout(at, ref["t_max"]), "{:.2f}".format(ref["t_max"]))

    # 정답 확인
    run_btn = next(b for b in at.button if b.label == "2D 해석 실행")
    run_btn.click().run()
    no_exception(at, "정답 확인 실행")
    hist = at.session_state["history"] if "history" in at.session_state else []
    check("기록 1건이 쌓인다", len(hist) == 1, str(hist[-1] if hist else ""))
    if hist:
        true = M.run_2d(dict(M.DEFAULT, void_mm=1.0, void_center_mm=5.0))["t_max"]
        check("기록의 실제 온도가 해석기와 같다", abs(hist[-1]["실제 ℃"] - round(true, 2)) < 1e-9,
              "{} vs {:.2f}".format(hist[-1]["실제 ℃"], true))

    # 위치를 가장자리로 → 폭을 키우면 안으로 옮겨져야 한다
    pos_slider = next(sl for sl in at.slider if sl.key == "void_center")
    pos_slider.set_value(pos_slider.min).run()
    void_slider = next(sl for sl in at.slider if sl.label.startswith("보이드 폭"))
    void_slider.set_value(4.0).run()
    no_exception(at, "가장자리에서 보이드 폭 4mm 로 키우기")
    pos_slider = next(sl for sl in at.slider if sl.key == "void_center")
    check("위치가 다이 안으로 옮겨진다", abs(pos_slider.value - 2.0) < 1e-9, "위치 {}".format(pos_slider.value))

    # 위험 지도, 검증 보기
    for name in ["보이드 위험 지도", "검증과 방법"]:
        at.radio(key="view").set_value(name).run()
        no_exception(at, name + " 보기")
    check("검증 보기에 점수가 나온다", 'class="score"' in texts(at))

    # 허용 보이드 보기
    at.radio(key="view").set_value("허용 보이드 크기").run()
    no_exception(at, "허용 보이드 보기")
    at.slider(key="inv_center").set_value(5.0).run()
    no_exception(at, "허용 보이드 위치 지정")
    at.number_input[0].set_value(45.0).run()
    no_exception(at, "온도 한계 45℃")
    btns = [b for b in at.button if b.label == "이 답을 실제 2D 해석으로 확인"]
    if btns:
        btns[0].click().run()
        no_exception(at, "역산 답 확인")
        check("역산 확인 결과가 나온다", "실제 최고 온도" in texts(at))
    else:
        check("역산 답이 크기로 나온다 (확인 버튼 있음)", False, "버튼 없음: " + texts(at)[-300:])

    # 학습 범위 끝 값들
    for label, val in [("총 발열", 15.0), ("TIM 열전도율", 1.0), ("방열판 저항", 3.0)]:
        sl = next(x for x in at.slider if x.label.startswith(label))
        sl.set_value(val).run()
    no_exception(at, "범위 끝 값")

    print("\n{}/{} 통과".format(sum(results), len(results)))
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
