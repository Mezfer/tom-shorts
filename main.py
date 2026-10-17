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
import wave
from moviepy.editor import CompositeAudioClip
import moviepy.audio.fx.all as afx


def make_music(dur, path="bg.wav"):
    sr = 44100
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    chords = [[220.00, 261.63, 329.63], [174.61, 220.00, 261.63],
              [130.81, 164.81, 196.00], [196.00, 246.94, 293.66]]
    out = np.zeros_like(t)
    seg = 6.0
    for k in range(int(dur // seg) + 1):
        s, e = int(k * seg * sr), min(int((k + 1) * seg * sr), len(t))
        if s >= e:
            break
        tt = t[s:e]
        env = np.clip(np.minimum((tt - tt[0]) / 1.5, (tt[-1] - tt) / 1.5), 0, 1)
        for f in chords[k % 4]:
            out[s:e] += env * (np.sin(2*np.pi*f*tt) + 0.3*np.sin(2*np.pi*2*f*tt))
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.5 * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(st.tobytes())
    return path


def add_music(voice, dur):
    m = AudioFileClip(make_music(dur + 1)).subclip(0, dur)
    m = m.fx(afx.volumex, 0.25).fx(afx.audio_fadein, 1).fx(afx.audio_fadeout, 2)
    return CompositeAudioClip([voice, m])


W, H = 1080, 1920
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
# Pixabay key (the secret may still be named PEXELS_API_KEY in your repo; the code calls Pixabay)
STOCK_KEY = os.environ.get("PIXABAY_API_KEY") or os.environ["PEXELS_API_KEY"]
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"] if m]
VOICE = os.environ.get("VOICE", "en-US-GuyNeural")
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history.txt"
USED_CLIPS = "used_clips.txt"
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# ONE niche only: ocean life. Every topic is visual, which is what performed best.
TOPICS = [
    "weird deep sea animals", "invisible or transparent ocean animals",
    "ocean animals with unusual abilities", "shipwrecks and sunken history",
    "bioluminescent sea creatures", "dangerous but fascinating sea animals",
    "ocean mysteries scientists are still studying", "the biggest and smallest ocean animals",
    "coral reefs and their secrets", "sharks, rays and whales facts",
    "octopus, squid and jellyfish facts", "how animals survive in the deepest ocean",
]


def read_lines(path, n):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-n:]


def gemini_json(prompt, temperature=1.0):
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
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


def fact_check(meta):
    prompt = f"""You are a strict science fact-checker. Check this short video script.
Title: {meta['title']}
Script: {meta['script']}
Mark accurate=false if ANY claim is false, exaggerated, disputed, oversimplified into a myth,
or cannot be confirmed by well-established sources.
Return ONLY JSON: {{"accurate": true or false, "confidence": 1-10, "issue": "short reason"}}"""
    return gemini_json(prompt, temperature=0.0)


def make_script():
    topic = random.choice(TOPICS)
    history = read_lines(HISTORY, 150)
    prompt = f"""You are Tom, a friendly narrator of a YouTube Shorts channel about amazing OCEAN and sea-animal facts.
Topic area: {topic}.
Write ONE surprising and TRUE fact video. Do NOT repeat these titles: {history}.
Only use well-established, verifiable science. No myths, no exaggeration, no "scientists say maybe".
Return ONLY JSON with this shape:
{{"title": "curiosity-driven title, max 70 chars, naming the animal or subject, include one keyword like ocean, deep sea, or the animal name",
"script": "70-85 words. The FIRST sentence is a punchy hook of max 10 words that shows the strangest part immediately (never start with 'Did you know'). Short sentences. End with a question that makes people comment. No emojis.",
"keywords": ["4 simple English phrases to search stock footage: concrete visual subjects like jellyfish, deep sea, coral reef, shark, octopus, underwater. No abstract words"],
"tags": ["6 to 8 tags: always include ocean facts, sea creatures, deep sea, did you know, plus the specific animal or subject"]}}"""
    for attempt in range(4):
        meta = gemini_json(prompt)
        check = fact_check(meta)
        print("Fact-check:", check)
        if check.get("accurate") is True and check.get("confidence", 0) >= 8:
            return meta
        print("Rejected, regenerating:", meta.get("title"))
    raise RuntimeError("No script passed the fact-check")


async def _tts(text, path):
    try:
        await edge_tts.Communicate(text, VOICE, rate="+5%").save(path)
    except Exception as e:
        print("edge-tts failed, using gTTS:", e)
        from gtts import gTTS
        gTTS(text, lang="en").save(path)


def get_clip(query, i, used):
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": STOCK_KEY, "q": query, "per_page": 20, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    hits = [h for h in r.json().get("hits", []) if str(h.get("id")) not in used]
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
        used.add(str(h["id"]))
        with open(USED_CLIPS, "a", encoding="utf-8") as f:
            f.write(str(h["id"]) + "\n")
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
    font = _font(72)
    img = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines, line = [], ""
    for w in text.split():
        test = (line + " " + w).strip()
        if line and d.textlength(test, font=font) > W - 200:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    lh = 90
    y = (420 - lh * len(lines)) // 2
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=6, stroke_fill=(0, 0, 0, 255))
        y += lh
    return np.array(img)


def add_subs(video, text, dur):
    """English captions (same language as the voice), 3-4 words at a time."""
    try:
        words = text.split()
        parts = [" ".join(words[i:i + 4]) for i in range(0, len(words), 4)]
        total = sum(len(p) for p in parts)
        t = 0.0
        subs = []
        for p in parts:
            d = dur * len(p) / total
            c = (ImageClip(_caption_image(p))
                 .set_start(t).set_duration(d)
                 .set_position(("center", 1100)))
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
    used = set(read_lines(USED_CLIPS, 2000))
    paths = [p for p in (get_clip(k, i, used) for i, k in enumerate(meta["keywords"])) if p]
    if not paths:
        raise RuntimeError("No stock clips found")
    seg = total / len(paths)
    clips = [fit(VideoFileClip(p), seg) for p in paths]
    video = concatenate_videoclips(clips).set_audio(add_music(audio, total))
    video = add_subs(video, meta["script"], audio.duration)
    video.write_videofile("out.mp4", fps=24, codec="libx264", audio_codec="aac",
                          preset="ultrafast", threads=2, logger=None)
    return "out.mp4"


def get_creds():
    cid = os.environ["YT_CLIENT_ID"].strip()
    sec = os.environ["YT_CLIENT_SECRET"].strip()
    rt = os.environ["YT_REFRESH_TOKEN"].strip()
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
            "description": (meta["script"]
                            + "\n\nNew ocean & sea creature facts every day. Subscribe for more!"
                            + "\n\n#Shorts #OceanFacts #DeepSea #SeaCreatures #DidYouKnow"),
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
