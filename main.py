import os, json, random, asyncio
import numpy as np
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (AudioFileClip, VideoFileClip, ImageClip,
                            CompositeVideoClip, concatenate_videoclips, vfx)
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

W, H = 1080, 1920
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
STOCK_KEY = os.environ.get("PIXABAY_API_KEY") or os.environ["PEXELS_API_KEY"]
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"] if m]
VOICE = os.environ.get("VOICE", "en-US-GuyNeural")
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history.txt"
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
TOPICS = [
    "space and planets", "animals", "the human body", "ancient history",
    "oceans", "technology and inventions", "food and nutrition",
    "psychology", "geography and countries", "science experiments",
]


def read_history():
    if not os.path.exists(HISTORY):
        return []
    with open(HISTORY, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-40:]


def make_script():
    prompt = f"""You are Tom, a friendly narrator of a YouTube Shorts channel about amazing facts.
Topic area: {random.choice(TOPICS)}.
Write ONE surprising and TRUE fact video. Do NOT repeat these titles: {read_history()}.
Return ONLY JSON with this shape:
{{"title": "curiosity-driven title, max 70 chars",
"script": "95-120 words, start with a strong hook, end with a question for comments, no emojis",
"script_fr": "natural French translation of the script, same meaning, no emojis",
"keywords": ["4 simple English phrases to search stock footage"],
"tags": ["5 to 8 tags"]}}"""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0},
    }
    last_err = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json=body, timeout=120)
        if r.status_code == 200:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            print("Model used:", model)
            return json.loads(text)
        last_err = f"{model}: {r.status_code} {r.text[:200]}"
        print("Model failed ->", last_err)
    raise RuntimeError("All Gemini models failed. Last: " + str(last_err))


async def _tts(text, path):
    await edge_tts.Communicate(text, VOICE, rate="+5%").save(path)


def get_clip(query, i):
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": STOCK_KEY, "q": query, "per_page": 15, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    hits = r.json().get("hits", [])
    random.shuffle(hits)
    for h in hits:
        url = None
        for size in ("large", "medium", "small"):
            d = h["videos"].get(size) or {}
            if d.get("url") and (d.get("width") or 0) <= 1920:
                url = d["url"]
                break
        if not url:
            continue
        path = f"clip{i}.mp4"
        with requests.get(url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(path, "wb") as out:
                for chunk in dl.iter_content(1 << 20):
                    out.write(chunk)
        return path
    return None


def fit(clip, dur):
    clip = clip.without_audio()
    s = max(W / clip.w, H / clip.h) * 1.01
    clip = clip.resize(s)
    clip = clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=W, height=H)
    if clip.duration < dur:
        clip = clip.fx(vfx.loop, duration=dur)
    return clip.subclip(0, dur)


def _font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise RuntimeError("No font found")


def _caption_image(text):
    font = _font(64)
    img = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines, line = [], ""
    for w in text.split():
        test = (line + " " + w).strip()
        if line and d.textlength(test, font=font) > W - 370:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    lh = 80
    y = (420 - lh * len(lines)) // 2
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=5, stroke_fill=(0, 0, 0, 255))
        y += lh
    return np.array(img)


def add_french_subs(video, text, dur):
    try:
        words = text.split()
        parts = [" ".join(words[i:i + 6]) for i in range(0, len(words), 6)]
        total = sum(len(p) for p in parts)
        t = 0.0
        subs = []
        for p in parts:
            d = dur * len(p) / total
            c = (ImageClip(_caption_image(p))
                 .set_start(t).set_duration(d)
                 .set_position(("center", 1180)))
            subs.append(c)
            t += d
        out = CompositeVideoClip([video] + subs).set_duration(video.duration)
        return out.set_audio(video.audio)
    except Exception as e:
        print("Subtitles skipped:", e)
        return video


def build_video(meta):
    asyncio.run(_tts(meta["script"], "voice.mp3"))
    audio = AudioFileClip("voice.mp3")
    total = audio.duration + 0.5
    paths = [p for p in (get_clip(k, i) for i, k in enumerate(meta["keywords"])) if p]
    if not paths:
        raise RuntimeError("No stock clips found")
    seg = total / len(paths)
    clips = [fit(VideoFileClip(p), seg) for p in paths]
    video = concatenate_videoclips(clips).set_audio(audio)
    if meta.get("script_fr"):
        video = add_french_subs(video, meta["script_fr"], audio.duration)
    video.write_videofile("out.mp4", fps=24, codec="libx264", audio_codec="aac",
                          preset="ultrafast", threads=2, logger=None)
    return "out.mp4"


def get_creds():
    cid = os.environ["YT_CLIENT_ID"].strip()
    sec = os.environ["YT_CLIENT_SECRET"].strip()
    rt = os.environ["YT_REFRESH_TOKEN"].strip()
    print("Check -> client_id format ok:", cid.endswith(".apps.googleusercontent.com"),
          "| secret starts with GOCSPX-:", sec.startswith("GOCSPX-"),
          "| refresh token starts with 1//:", rt.startswith("1//"))
    creds = Credentials(
        None,
        refresh_token=rt,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=sec,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())  # fail fast, before rendering the video
    print("YouTube authorization OK")
    return creds


def upload(path, meta, creds):
    yt = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": meta["title"][:95],
            "description": meta["script"] + "\n\n#Shorts #Facts #DidYouKnow",
            "tags": meta["tags"],
            "categoryId": "27",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    req = yt.videos().insert(
        part="snippet,status", body=body,
        media_body=MediaFileUpload(path, chunksize=-1, resumable=True),
    )
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    print("Uploaded:", resp["id"])


if __name__ == "__main__":
    creds = get_creds()
    meta = make_script()
    print("Title:", meta["title"])
    video_path = build_video(meta)
    upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(meta["title"] + "\n")
