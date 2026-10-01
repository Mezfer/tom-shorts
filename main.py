import os, json, random, asyncio
import requests
import edge_tts
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips, vfx
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

W, H = 1080, 1920
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
PEXELS_KEY = os.environ["PEXELS_API_KEY"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.environ.get("VOICE", "en-US-GuyNeural")
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history.txt"

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
 "keywords": ["4 simple English phrases to search stock footage"],
 "tags": ["5 to 8 tags"]}}"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"
    r = requests.post(url, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0},
    }, timeout=90)
    r.raise_for_status()
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])


async def _tts(text, path):
    await edge_tts.Communicate(text, VOICE, rate="+5%").save(path)


def get_clip(query, i):
    r = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": PEXELS_KEY},
        params={"query": query, "orientation": "portrait", "per_page": 10},
        timeout=30,
    )
    vids = r.json().get("videos", [])
    if not vids:
        return None
    v = random.choice(vids[:6])
    files = [f for f in v["video_files"]
             if f.get("file_type") == "video/mp4" and f.get("width") and f["width"] <= 1080]
    files = files or v["video_files"]
    f = max(files, key=lambda x: x.get("width") or 0)
    path = f"clip{i}.mp4"
    with requests.get(f["link"], stream=True, timeout=120) as d:
        d.raise_for_status()
        with open(path, "wb") as out:
            for chunk in d.iter_content(1 << 20):
                out.write(chunk)
    return path


def fit(clip, dur):
    clip = clip.without_audio()
    s = max(W / clip.w, H / clip.h) * 1.01
    clip = clip.resize(s)
    clip = clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=W, height=H)
    if clip.duration < dur:
        clip = clip.fx(vfx.loop, duration=dur)
    return clip.subclip(0, dur)


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
    video.write_videofile("out.mp4", fps=24, codec="libx264", audio_codec="aac",
                          preset="ultrafast", threads=2, logger=None)
    return "out.mp4"


def upload(path, meta):
    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
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
    meta = make_script()
    print("Title:", meta["title"])
    video_path = build_video(meta)
    upload(video_path, meta)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(meta["title"] + "\n")
