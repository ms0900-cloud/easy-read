import os
import re
import textwrap
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# [고침 1] 깃허브에 올린 글꼴 파일 이름과 똑같이 (대소문자까지)
#          아래 이름 중 깃허브에 있는 것을 순서대로 찾아서 씀
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_NAMES = ["Pretendard-Regular.ttf", "NanumGothicBold.ttf"]
FONT_PATH = next((os.path.join(HERE, f) for f in FONT_NAMES if os.path.exists(os.path.join(HERE, f))),
                 os.path.join(HERE, FONT_NAMES[0]))

# [고침 2] gemini-2.5-flash 대신 지금 쓸 수 있는 모델
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

PROMPT = (
    "너는 특수학교 선생님이야. 사진 속 글을 읽고 지적장애 및 경계선지능 학생(초등 1~2학년 수준)이 "
    "이해하기 쉽게 바꿔줘.\n"
    "규칙:\n"
    "1. 어려운 한자어와 전문 용어는 쉬운 일상어로 풀어써.\n"
    "2. 반드시 '한 줄에 한 문장씩' 줄바꿈해서 작성해.\n"
    "3. 한 문장은 18자 이내로 짧게 끊고, 핵심만 간략하게 최대한 5문장 안에 정리해.\n"
    "4. 한 사진에 많은 양의 텍스트가 있는 경우, 적절하게 구역을 나누어서 여러 차례에 걸쳐 하도록해\n"
    "5. 번호(1. 2.), 별표(**), 이모지 같은 특수기호는 절대 쓰지 마."
    "6. 숫자, 횟수, 시간, 양 (예. 하루 3번, 30분, 1알)은 빼지 말고 꼭 넣어."
)


def load_fonts():
    try:
        return ImageFont.truetype(FONT_PATH, 28), ImageFont.truetype(FONT_PATH, 18)
    except Exception as e:
        # 글꼴을 못 찾으면 한글이 네모로 나와요. Render 로그에서 이 줄을 확인하세요.
        print(f"글꼴 오류: {e}  (찾은 위치: {FONT_PATH})")
        f = ImageFont.load_default()
        return f, f


FONT_MAIN, FONT_PAGE = load_fonts()


def render_sentence_to_bitmap(sentence: str, index: int, total: int) -> bytes:
    """한 문장을 320x240 해상도의 1비트 흑백 이미지(9,600 bytes)로 변환"""
    W, H = 320, 240
    img = Image.new("1", (W, H), 1)  # 1 = 흰 배경
    draw = ImageDraw.Draw(img)

    # 테두리 및 상단 진행 표시 (예: 1 / 4)
    draw.rectangle([4, 4, W - 5, H - 5], outline=0, width=2)
    draw.text((245, 14), f"{index} / {total}", font=FONT_PAGE, fill=0)
    draw.line([(12, 42), (W - 13, 42)], fill=0, width=1)

    # 한 줄에 10자씩 자동 줄바꿈
    wrapped_lines = textwrap.wrap(sentence, width=10)

    # 문장 길이에 맞춰 세로 위치 조절 후 그리기
    y = 65 if len(wrapped_lines) <= 3 else 50
    for line in wrapped_lines[:4]:
        draw.text((20, y), line, font=FONT_MAIN, fill=0)
        y += 42

    return img.tobytes()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "쉬운말 서버 작동 중"}


@app.post("/simplify")
async def simplify_image(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="이미지 없음")

        # [고침 3] 새 모델에서는 temperature 설정 줄을 뺐어요
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                PROMPT
            ],
        )

        # 특수기호 제거 후 한 줄(문장) 단위로 분리
        cleaned = re.sub(r'[*#_`~-]', '', response.text or "").strip()
        sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]
        sentences = sentences[:5]  # 최대 5문장 제한

        if not sentences:
            sentences = ["글자를 잘 못 읽었어요.", "다시 한 번 찍어주세요."]

        total = len(sentences)
        print(f"[변환 결과 ({total}문장)]: {sentences}  사진 {len(image_bytes) // 1024}KB")

        # 전송 패킷: [문장개수 1바이트] + [문장1 이미지 9600B] + [문장2 이미지 9600B] ...
        payload = bytearray([total])
        for idx, sentence in enumerate(sentences, start=1):
            payload.extend(render_sentence_to_bitmap(sentence, idx, total))

        return Response(content=bytes(payload), media_type="application/octet-stream")

    except Exception as e:
        print(f"오류: {e}")
        err_bmp = render_sentence_to_bitmap("다시 한 번 찍어주세요.", 1, 1)
        return Response(content=bytes([1]) + err_bmp, media_type="application/octet-stream")
