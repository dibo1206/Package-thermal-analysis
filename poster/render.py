"""poster.html → poster.pdf (A2) + poster_preview.png. Windows Edge 를 화면 없이 띄워 인쇄한다.

  python poster/render.py
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websocket

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9334
W_PX, H_PX = 1587, 2245          # 420 x 594 mm @ 96 dpi

proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                         "--remote-debugging-port={}".format(PORT),
                         "--user-data-dir=" + os.path.join(tempfile.gettempdir(), "poster-edge"),
                         "--allow-file-access-from-files", "about:blank"])
try:
    for _ in range(50):
        try:
            page = next(t for t in json.load(urllib.request.urlopen("http://127.0.0.1:{}/json".format(PORT)))
                        if t["type"] == "page")
            break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120)
    n = [0]

    def call(method, **params):
        n[0] += 1
        ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == n[0]:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    call("Page.enable")
    call("Emulation.setDeviceMetricsOverride", width=W_PX, height=H_PX, deviceScaleFactor=1, mobile=False)
    call("Page.navigate", url="file:///" + os.path.join(HERE, "poster.html").replace("\\", "/"))
    time.sleep(6)                                   # 글꼴·QR 코드 로딩
    pdf = call("Page.printToPDF", paperWidth=420 / 25.4, paperHeight=594 / 25.4, marginTop=0, marginBottom=0,
               marginLeft=0, marginRight=0, printBackground=True, preferCSSPageSize=True)
    with open(os.path.join(HERE, "poster.pdf"), "wb") as f:
        f.write(base64.b64decode(pdf["data"]))
    shot = call("Page.captureScreenshot", format="png")
    with open(os.path.join(HERE, "poster_preview.png"), "wb") as f:
        f.write(base64.b64decode(shot["data"]))
    over = call("Runtime.evaluate", expression="(() => { const p = document.querySelector('.page'); "
                "return [p.scrollHeight, p.clientHeight]; })()", returnByValue=True)["result"]["value"]
    print("저장: poster.pdf, poster_preview.png / 내용 높이 {} vs 페이지 {} px{}".format(
        over[0], over[1], "  ← 넘침!" if over[0] > over[1] + 1 else ""))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
