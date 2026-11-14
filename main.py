import io
import os
import re
import time
import textwrap
import threading
from collections import deque

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, HTMLResponse, JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageOps

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 깃허브에 올린 글꼴 파일 이름 (아래 이름 중 있는 것을 씀)
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_NAMES = ["Pretendard-Regular.ttf", "NanumGothicBold.ttf"]
FONT_PATH = next((os.path.join(HERE, f) for f in FONT_NAMES if os.path.exists(os.path.join(HERE, f))),
                 os.path.join(HERE, FONT_NAMES[0]))

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

RULES = (
    "규칙:\n"
    "1. 어려운 한자어와 전문 용어는 쉬운 일상어로 풀어써.\n"
    "2. 반드시 '한 줄에 한 문장씩' 줄바꿈해서 작성해.\n"
    "3. 한 문장은 18자 이내로 짧게 끊고, 핵심만 간략하게 최대한 {n}문장 안에 정리해.\n"
    "4. 번호(1. 2.), 별표(**), 이모지 같은 특수기호는 절대 쓰지 마.\n"
    "5. 숫자, 횟수, 시간, 양(예: 하루 3번, 30분, 1알)은 빼지 말고 꼭 넣어."
    "6. 사진을 묘사하지 말고, 사진 속 글자를 인식해서 풀어써. 글자가 없다면 글자를 잘 못 읽었어요., 다시 한 번 찍어주세요.라고 출력하도록해"
    "7. 영어가 있다면 쉬운 한글말로 해줘"
)
PHOTO_PROMPT = (
    "너는 특수학교 선생님이야. 사진 속 글을 읽고 지적장애 및 경계선지능 학생(초등 1~2학년 수준)이 "
    "이해하기 쉽게 바꿔줘.\n" + RULES.format(n=5)
)
TEXT_PROMPT = (
    "너는 특수학교 선생님이야. 아래 글을 지적장애 및 경계선지능 학생(초등 1~2학년 수준)이 "
    "이해하기 쉽게 바꿔줘.\n" + RULES.format(n=8) + "\n\n글:\n"
)

# ---------------- 기록 (서버 메모리에 최근 30개) ----------------
# Render 무료 서버는 다시 시작하면 기록이 지워져요. 웹 화면이 휴대폰에도 따로 저장해 둬요.
HISTORY = deque(maxlen=30)
PHOTOS = {}
LOCK = threading.Lock()


def new_id(kind):
    return f"{kind}-{int(time.time() * 1000)}"


def save_history(kind, sentences, photo_bytes=None, source=""):
    item_id = new_id(kind)
    item = {"id": item_id, "kind": kind, "time": int(time.time() * 1000),
            "sentences": sentences, "source": source[:300], "photo": bool(photo_bytes)}
    with LOCK:
        if len(HISTORY) == HISTORY.maxlen:
            PHOTOS.pop(HISTORY[0]["id"], None)
        HISTORY.append(item)
        if photo_bytes:
            PHOTOS[item_id] = photo_bytes
    return item


def make_thumb(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((640, 640))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=70)
        return out.getvalue()
    except Exception as e:
        print(f"사진 줄이기 오류: {e}")
        return None


def to_sentences(text, limit):
    cleaned = re.sub(r'[*#_`~]', '', text or "").strip()
    lines = [re.sub(r'^\s*(\d+[.)]|-)\s*', '', s).strip() for s in cleaned.split("\n")]
    return [s for s in lines if s][:limit]


# ---------------- 본체 화면용 그림 ----------------
def load_fonts():
    try:
        return ImageFont.truetype(FONT_PATH, 28), ImageFont.truetype(FONT_PATH, 18)
    except Exception as e:
        print(f"글꼴 오류: {e}  (찾은 위치: {FONT_PATH})")
        f = ImageFont.load_default()
        return f, f


FONT_MAIN, FONT_PAGE = load_fonts()


def render_sentence_to_bitmap(sentence: str, index: int, total: int) -> bytes:
    """한 문장을 320x240 해상도의 1비트 흑백 이미지(9,600 bytes)로 변환"""
    W, H = 320, 240
    img = Image.new("1", (W, H), 1)  # 1 = 흰 배경
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, W - 5, H - 5], outline=0, width=2)
    draw.text((245, 14), f"{index} / {total}", font=FONT_PAGE, fill=0)
    draw.line([(12, 42), (W - 13, 42)], fill=0, width=1)
    wrapped_lines = textwrap.wrap(sentence, width=10)
    y = 65 if len(wrapped_lines) <= 3 else 50
    for line in wrapped_lines[:4]:
        draw.text((20, y), line, font=FONT_MAIN, fill=0)
        y += 42
    return img.tobytes()


# ---------------- 본체(ESP32)가 부르는 곳: 바뀐 것 없음 ----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "쉬운말 서버 작동 중"}


@app.post("/simplify")
def simplify_image(image: UploadFile = File(...)):
    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise ValueError("이미지 없음")
        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=image.content_type or "image/jpeg"),
                      PHOTO_PROMPT],
        )
        sentences = to_sentences(response.text, 5)
        if not sentences:
            sentences = ["글자를 잘 못 읽었어요.", "다시 한 번 찍어주세요."]
        print(f"[변환 결과 ({len(sentences)}문장)]: {sentences}  사진 {len(image_bytes) // 1024}KB")
        save_history("photo", sentences, photo_bytes=make_thumb(image_bytes))

        payload = bytearray([len(sentences)])
        for idx, sentence in enumerate(sentences, start=1):
            payload.extend(render_sentence_to_bitmap(sentence, idx, len(sentences)))
        return Response(content=bytes(payload), media_type="application/octet-stream")

    except Exception as e:
        print(f"오류: {e}")
        err_bmp = render_sentence_to_bitmap("다시 한 번 찍어주세요.", 1, 1)
        return Response(content=bytes([1]) + err_bmp, media_type="application/octet-stream")


# ---------------- 웹 화면이 부르는 곳 ----------------
class TextIn(BaseModel):
    text: str


@app.post("/simplify-text")
def simplify_text(body: TextIn):
    text = (body.text or "").strip()[:3000]
    if not text:
        return JSONResponse({"error": "글을 넣어 주세요."}, status_code=400)
    try:
        response = client.models.generate_content(model=MODEL, contents=[TEXT_PROMPT + text])
        sentences = to_sentences(response.text, 8)
        if not sentences:
            return JSONResponse({"error": "글을 바꾸지 못했어요. 다시 해 주세요."}, status_code=500)
        print(f"[글 바꾸기 ({len(sentences)}문장)]: {sentences}")
        return save_history("text", sentences, source=text)
    except Exception as e:
        print(f"글 바꾸기 오류: {e}")
        return JSONResponse({"error": "지금은 바꿀 수 없어요. 잠시 뒤 다시 해 주세요."}, status_code=500)


@app.get("/history")
def history():
    with LOCK:
        return list(reversed(HISTORY))


@app.delete("/history")
def clear_history():
    with LOCK:
        HISTORY.clear()
        PHOTOS.clear()
    return {"ok": True}


@app.get("/photo/{item_id}")
def photo(item_id: str):
    data = PHOTOS.get(item_id)
    if not data:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def web_app():
    return WEB_PAGE


WEB_PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>쉬운 말 도우미</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; background: #fff; color: #000;
    font-family: "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif; }
  body { font-size: 22px; line-height: 1.5; }
  .wrap { max-width: 640px; margin: 0 auto; padding: 20px 18px 80px; }
  h1 { font-size: 32px; margin: 4px 0 6px; }
  .sub { margin: 0 0 24px; font-size: 20px; }
  button { font: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; }
  button:focus-visible, textarea:focus-visible { outline: 4px solid #000; outline-offset: 4px; }
  .big { display: block; width: 100%; min-height: 112px; margin: 0 0 16px; padding: 18px 22px;
    border: 3px solid #000; border-radius: 10px; background: #fff; color: #000; text-align: left; }
  .big b { display: block; font-size: 30px; }
  .big span { display: block; font-size: 20px; margin-top: 2px; }
  .big:active, .btn:active, .item:active, .back:active { background: #000; color: #fff; }
  .top { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
  .back { min-height: 60px; padding: 0 20px; border: 3px solid #000; border-radius: 10px; background: #fff;
    color: #000; font-size: 22px; font-weight: 700; }
  .title { font-size: 28px; font-weight: 800; }
  .list { display: grid; gap: 12px; }
  .item { display: flex; gap: 14px; align-items: center; width: 100%; min-height: 104px; padding: 12px;
    border: 3px solid #000; border-radius: 10px; background: #fff; color: #000; text-align: left; }
  .thumb { width: 84px; height: 84px; flex: none; object-fit: cover; border: 2px solid #000; border-radius: 6px;
    display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; background: #fff; }
  .item b { display: block; font-size: 23px; }
  .item span { font-size: 18px; }
  .tag { display: inline-block; margin-right: 6px; padding: 0 8px; border-radius: 4px; background: #000; color: #fff;
    font-size: 16px; font-weight: 700; }
  .empty { border: 3px dashed #000; border-radius: 10px; padding: 26px; text-align: center; }
  textarea { width: 100%; min-height: 220px; padding: 16px; font: inherit; font-size: 22px; color: #000;
    border: 3px solid #000; border-radius: 10px; background: #fff; resize: vertical; }
  .row { display: flex; gap: 12px; margin-top: 14px; }
  .btn { flex: 1; min-height: 80px; border: 3px solid #000; border-radius: 10px; background: #fff; color: #000;
    font-size: 26px; font-weight: 800; }
  .btn.main { background: #000; color: #fff; }
  .btn.main:active { background: #fff; color: #000; }
  .btn:disabled { border-color: #bbb; color: #bbb; background: #fff; }
  .card { border: 3px solid #000; border-radius: 10px; padding: 24px 20px; min-height: 220px;
    display: flex; flex-direction: column; justify-content: center; }
  .count { font-size: 22px; font-weight: 800; margin-bottom: 10px; }
  .sentence { font-size: 36px; font-weight: 800; line-height: 1.45; word-break: keep-all; }
  .photo { width: 100%; max-height: 150px; object-fit: contain; border: 2px solid #000; border-radius: 6px;
    margin-bottom: 14px; background: #fff; }
  .wide { width: 100%; margin-top: 14px; }
  .msg { margin-top: 14px; padding: 14px 16px; border: 3px solid #000; border-radius: 10px; font-weight: 700; }
  .overlay { position: fixed; inset: 0; background: #fff; display: none; z-index: 10;
    flex-direction: column; align-items: center; justify-content: center; gap: 24px; font-size: 30px; font-weight: 800; }
  .bar { width: 220px; height: 14px; border: 3px solid #000; border-radius: 7px; overflow: hidden; }
  .bar i { display: block; width: 40%; height: 100%; background: #000; animation: move 1.2s ease-in-out infinite; }
  @keyframes move { 0% { transform: translateX(-100%); } 100% { transform: translateX(250%); } }
  .small-link { display: block; margin: 28px auto 0; border: none; background: none; color: #000; font-size: 18px;
    text-decoration: underline; min-height: 48px; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<div class="wrap">

  <!-- 처음 화면 -->
  <section id="home">
    <h1>쉬운 말 도우미</h1>
    <p class="sub">어려운 글을 쉬운 말로 바꿔 줘요.</p>
    <button class="big" onclick="openHistory()"><b>찍은 글 보기</b><span>기계로 찍은 글을 다시 봐요</span></button>
    <button class="big" onclick="go('paste')"><b>글 바꾸기</b><span>휴대폰 속 글을 쉽게 바꿔요</span></button>
  </section>

  <!-- 찍은 글 목록 -->
  <section id="history" hidden>
    <div class="top"><button class="back" onclick="go('home')">← 처음</button><span class="title">찍은 글</span></div>
    <div id="list" class="list"></div>
    <button class="small-link" onclick="clearAll()">기록 모두 지우기</button>
  </section>

  <!-- 글 붙여넣기 -->
  <section id="paste" hidden>
    <div class="top"><button class="back" onclick="go('home')">← 처음</button><span class="title">글 바꾸기</span></div>
    <textarea id="input" placeholder="여기에 어려운 글을 넣어 주세요."></textarea>
    <div class="row">
      <button class="btn" onclick="pasteClip()">붙여넣기</button>
      <button class="btn" onclick="clearInput()">지우기</button>
    </div>
    <div class="row"><button id="go" class="btn main" onclick="simplify()">쉽게 바꾸기</button></div>
    <div id="pasteMsg" class="msg" hidden></div>
  </section>

  <!-- 한 문장씩 보기 -->
  <section id="viewer" hidden>
    <div class="top"><button class="back" onclick="backFromViewer()">← 뒤로</button><span class="title" id="vTitle">쉬운 말</span></div>
    <img id="vPhoto" class="photo" alt="찍은 사진" hidden>
    <div class="card">
      <div class="count" id="vCount">1 / 1</div>
      <div class="sentence" id="vText" aria-live="polite"></div>
    </div>
    <div class="row">
      <button id="prev" class="btn" onclick="step(-1)">← 이전</button>
      <button id="next" class="btn main" onclick="step(1)">다음 →</button>
    </div>
    <button class="btn wide" onclick="speak()">소리로 듣기</button>
  </section>
</div>

<div class="overlay" id="wait"><div>바꾸고 있어요</div><div class="bar"><i></i></div></div>

<script>
  // ---------- 휴대폰에 저장 (서버가 다시 켜져도 남아요) ----------
  const KEY = 'easyread-history-v1';
  function load() { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; } }
  function store(items) { try { localStorage.setItem(KEY, JSON.stringify(items.slice(0, 40))); } catch (e) {} }
  let items = load();
  let current = null, page = 0, viewerFrom = 'home', seenIds = new Set(items.map(i => i.id)), poll = null;

  function go(id) {
    for (const s of ['home', 'history', 'paste', 'viewer']) document.getElementById(s).hidden = (s !== id);
    window.scrollTo(0, 0);
    if (poll) { clearInterval(poll); poll = null; }
    if (id === 'history') { refresh(); poll = setInterval(refresh, 4000); }
    speechSynthesis.cancel();
  }

  function when(ms) {
    const d = new Date(ms), now = new Date();
    const t = d.toLocaleTimeString('ko-KR', { hour: 'numeric', minute: '2-digit' });
    return d.toDateString() === now.toDateString() ? '오늘 ' + t : (d.getMonth() + 1) + '월 ' + d.getDate() + '일 ' + t;
  }

  // ---------- 찍은 글 ----------
  function openHistory() { go('history'); }

  async function refresh() {
    try {
      const r = await fetch('/history', { cache: 'no-store' });
      const server = await r.json();
      let changed = false;
      for (const it of server) {
        if (!items.some(x => x.id === it.id)) { items.push(it); changed = true; }
      }
      if (changed) {
        items.sort((a, b) => b.time - a.time);
        store(items);
        for (const it of items) if (it.photo && !it.img) savePhoto(it);
      }
    } catch (e) {}
    drawList();
  }

  // 사진도 휴대폰에 저장해 둬요
  async function savePhoto(it) {
    try {
      const r = await fetch('/photo/' + it.id);
      if (!r.ok) return;
      const b = await r.blob();
      const fr = new FileReader();
      fr.onload = () => { it.img = fr.result; store(items); drawList(); };
      fr.readAsDataURL(b);
    } catch (e) {}
  }

  function drawList() {
    const list = document.getElementById('list');
    list.innerHTML = '';
    const photos = items.filter(i => i.kind === 'photo' || i.kind === 'text');
    if (!photos.length) {
      list.innerHTML = '<div class="empty">아직 기록이 없어요.<br>기계의 <b>1번</b>을 눌러<br>글을 찍어 보세요.</div>';
      return;
    }
    for (const it of photos) {
      const b = document.createElement('button');
      b.className = 'item';
      const th = it.img ? `<img class="thumb" src="${it.img}" alt="">`
               : it.kind === 'photo' ? `<img class="thumb" src="/photo/${it.id}" alt="" onerror="this.outerHTML='<div class=thumb>사진</div>'">`
               : '<div class="thumb">글</div>';
      b.innerHTML = th + '<div><b></b><span></span></div>';
      b.querySelector('b').textContent = it.sentences[0] || '';
      b.querySelector('span').innerHTML = (seenIds.has(it.id) ? '' : '<span class="tag">새 글</span>');
      b.querySelector('span').append((it.kind === 'photo' ? '찍은 글 · ' : '바꾼 글 · ') + when(it.time));
      b.onclick = () => { seenIds.add(it.id); openViewer(it, 'history'); };
      list.appendChild(b);
    }
  }

  async function clearAll() {
    if (!confirm('기록을 모두 지울까요?')) return;
    items = []; store(items); seenIds = new Set();
    try { await fetch('/history', { method: 'DELETE' }); } catch (e) {}
    drawList();
  }

  // ---------- 글 붙여넣기 ----------
  async function pasteClip() {
    try {
      const t = await navigator.clipboard.readText();
      if (t) document.getElementById('input').value = t;
    } catch (e) {
      showMsg('글 칸을 길게 눌러서 "붙여넣기"를 골라 주세요.');
    }
  }
  function clearInput() { document.getElementById('input').value = ''; document.getElementById('pasteMsg').hidden = true; }
  function showMsg(t) { const m = document.getElementById('pasteMsg'); m.textContent = t; m.hidden = false; }

  async function simplify() {
    const text = document.getElementById('input').value.trim();
    if (!text) { showMsg('먼저 글을 넣어 주세요.'); return; }
    document.getElementById('pasteMsg').hidden = true;
    document.getElementById('wait').style.display = 'flex';
    try {
      const r = await fetch('/simplify-text', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({ text }) });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || '오류');
      items.unshift(data); store(items); seenIds.add(data.id);
      openViewer(data, 'paste');
    } catch (e) {
      showMsg(e.message && e.message !== 'Failed to fetch' ? e.message : '서버가 깨어나는 중이에요. 1분 뒤 다시 눌러 주세요.');
    } finally {
      document.getElementById('wait').style.display = 'none';
    }
  }

  // ---------- 한 문장씩 보기 ----------
  function openViewer(it, from) {
    current = it; page = 0; viewerFrom = from;
    const img = document.getElementById('vPhoto');
    if (it.kind === 'photo') {
      img.src = it.img || ('/photo/' + it.id);
      img.hidden = false;
      img.onerror = () => { img.hidden = true; };
    } else img.hidden = true;
    document.getElementById('vTitle').textContent = it.kind === 'photo' ? '찍은 글' : '바꾼 글';
    go('viewer');
    show();
  }
  function show() {
    const n = current.sentences.length;
    document.getElementById('vText').textContent = current.sentences[page];
    document.getElementById('vCount').textContent = (page + 1) + ' / ' + n;
    document.getElementById('prev').disabled = (page === 0);
    document.getElementById('next').textContent = (page === n - 1) ? '끝' : '다음 →';
  }
  function step(d) {
    speechSynthesis.cancel();
    const n = current.sentences.length;
    if (d > 0 && page === n - 1) { backFromViewer(); return; }
    page = Math.max(0, Math.min(n - 1, page + d));
    show();
  }
  function backFromViewer() { go(viewerFrom === 'paste' ? 'paste' : viewerFrom); }

  function speak() {
    try {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(current.sentences[page]);
      u.lang = 'ko-KR'; u.rate = 0.85;
      speechSynthesis.speak(u);
    } catch (e) {}
  }
</script>
</body>
</html>
"""
