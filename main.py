import os, json, random, asyncio, re, csv, difflib, datetime
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
IDEAS_LOG = "ideas_log.jsonl"        # one full "video card" per published video
VIDEO_LOG = "video_log.csv"          # performance table (views, likes, ...)
REVIEW_QUEUE = "review_queue.jsonl"  # ideas rejected by the checks, kept for manual review
STATS_MANUAL = "stats_manual.csv"    # optional: numbers copied by hand from YouTube Studio
REPORT = "report.md"
CARD = "latest_card.md"


def _env_num(name, default, cast):
    try:
        return cast(os.environ.get(name) or default)
    except ValueError:
        return default


# Every new feature can be switched off from the workflow (env) if it ever causes trouble.
SMART_ENGINE = (os.environ.get("SMART_ENGINE") or "1") == "1"
ANIMAL_COOLDOWN = _env_num("ANIMAL_COOLDOWN", 8, int)      # videos before an animal can return
SIM_LIMIT = _env_num("SIM_LIMIT", 0.55, float)             # similarity above this = too similar
MIN_EXACT_RATIO = _env_num("MIN_EXACT_RATIO", 0.5, float)  # share of scenes that must show the real animal
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


# Famous animals people already search for, mixed with angles that make people stop scrolling.
ANIMALS = [
    "blue whale", "octopus", "great white shark", "orca", "dolphin", "sea turtle",
    "manta ray", "seahorse", "jellyfish", "giant squid", "narwhal", "clownfish",
    "sea otter", "pufferfish", "humpback whale", "whale shark", "anglerfish",
    "electric eel", "sperm whale", "hammerhead shark", "stingray", "starfish",
    "crab", "lobster", "walrus", "seal", "penguin", "coral reef", "sea cucumber",
    "mantis shrimp", "pistol shrimp", "leatherback turtle", "swordfish", "barracuda",
]
ANGLES = [
    "a size or weight comparison with a famous everyday thing (an elephant, a bus, a car, a human)",
    "an extreme ability (fastest, strongest, loudest, deepest)",
    "a strange body part and what it is used for",
    "how it survives or hunts in a surprising way",
    "a weird sleeping, eating or parenting habit",
    "a record this animal holds",
]
HOOKS = [
    "start with a shocking comparison",
    "start with a surprising number",
    "start with 'Imagine' and a vivid scene",
    "start with a question the viewer cannot answer",
    "start by stating the strangest part directly",
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
Mark accurate=false ONLY if the MAIN claim is clearly false. Ignore minor wording, rounding, or tiny details.
Accept well-known facts. Return confidence 10 only when you are sure the main claim is wrong.
Return ONLY JSON: {{"accurate": true or false, "confidence": 1-10, "issue": "short reason"}}"""
    return gemini_json(prompt, temperature=0.0)


def make_script_legacy():
    animal = random.choice(ANIMALS)
    angle = random.choice(ANGLES)
    hook = random.choice(HOOKS)
    topic = f"{animal} - {angle}"
    history = read_lines(HISTORY, 150)
    prompt = f"""You are Tom, a friendly narrator of a YouTube Shorts channel about amazing OCEAN and sea-animal facts.
Subject: {topic}. Hook style: {hook}.
Sound like a real person talking to a friend: simple words, contractions, a little wonder or humor. Never sound like a textbook or a robot.
Write ONE surprising and TRUE fact video. Do NOT repeat these titles: {history}.
Choose a famous, simple, well-documented fact about a sea ANIMAL or natural place. Avoid shipwrecks, history, war ships, exact numbers and anything disputed.
Return ONLY JSON with this shape:
{{"title": "curiosity-driven title, max 70 chars, MUST name the animal and contain a surprising comparison or number, like 'The Blue Whale Tongue Weighs As Much As An Elephant'",
"script": "70-85 words. The FIRST sentence is a punchy hook of max 10 words that shows the strangest part immediately (never start with 'Did you know'). Short sentences. End with a question that makes people comment. No emojis.",
"keywords": ["4 English phrases to search stock footage. The FIRST two MUST be the exact animal named in the script (for example: blue whale, sea otter). The other two are generic like underwater, ocean, coral reef. No abstract words"],
"tags": ["6 to 8 tags: always include ocean facts, sea creatures, deep sea, did you know, plus the specific animal name"]}}"""
    for attempt in range(6):
        meta = gemini_json(prompt)
        check = fact_check(meta)
        print("Fact-check:", check)
        if check.get("accurate") is True and check.get("confidence", 0) >= 8:
            return meta
        print("Rejected, regenerating:", meta.get("title"))
    raise RuntimeError("No script passed the fact-check")


# =====================================================================
#  Smart engine: idea ranking, hooks, verified script, matched footage,
#  similarity guard, performance log. Everything below is additive: the
#  original generator is kept as make_script_legacy() and is used as a
#  fallback if the smart engine hits a technical error.
# =====================================================================
FAILED_ANIMALS = set()   # animals whose footage failed earlier in this run


class VerificationFailed(Exception):
    """No idea passed the checks (script quality / fact verification)."""


class MatchFailure(Exception):
    """Not enough footage that really shows the animal."""


SLOTS = {
    "clear_fact": "a strong, easy-to-understand fact that hits in the first second",
    "behavior_story": "a surprising behavior or survival story (a problem and a strange solution)",
    "comparison_puzzle": "a size/ability comparison, a science puzzle, or a gradual reveal of a hidden secret",
}
HOOK_TYPES = [
    "comparison with something familiar",
    "true fact that is hard to believe",
    "unexpected ability",
    "survival problem and a strange solution",
    "behavior that looks impossible",
    "curiosity-raising science question",
    "gradual reveal of an anatomical secret",
    "comparison between two animals",
    "common mistake and its correction",
    "surprising result first, then the reason",
]
SERIES = [
    "Deep Ocean Secrets", "Biggest Animals In The Sea", "Strangest Hunting Tricks",
    "Sea Abilities That Sound Fake", "Scary-Looking But Surprising",
    "Ocean Survival Battles",
]
CTA_LINES = [
    "More ocean facts every day on Tom Facts.",
    "Follow Tom Facts for the next sea creature secret.",
    "New sea animal facts daily. See you in the next one.",
    "Want more weird ocean science? Tom Facts has you covered.",
]
GENERIC_SUBJECTS = {"", "ocean", "sea", "underwater", "deep sea", "scenery", "generic"}
_STOP = {"the", "a", "an", "of", "to", "in", "on", "is", "are", "as", "and", "or", "its", "it", "for",
         "with", "that", "this", "has", "have", "can", "by", "at", "from", "why", "how", "what", "not",
         "actually", "more", "than", "you", "your", "their", "they", "them", "be", "was", "were", "but",
         "so", "just", "even", "still", "very", "about", "when", "while", "which", "who", "do", "does"}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)


def current_slot():
    s = (os.environ.get("SLOT") or "").strip()
    if s in SLOTS:
        return s
    h = now_utc().hour
    return "clear_fact" if h < 15 else ("behavior_story" if h < 19 else "comparison_puzzle")


def jsonl_read(path, n=300):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out[-n:]


def jsonl_append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _tokens(text):
    """Content words only, with a crude plural fix, so shared title templates do not count."""
    words = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in _STOP]


def similarity(a, b):
    """0..1 similarity of two texts by their content words (overlap or word-order match)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    jac = len(set(ta) & set(tb)) / len(set(ta) | set(tb))
    seq = difflib.SequenceMatcher(None, ta, tb).ratio()
    return max(jac, seq)


def animals_in_text(text):
    t = (text or "").lower()
    return [a for a in ANIMALS if a in t]


def clean_text(text):
    """Printable, single-spaced text (safe for titles and captions)."""
    text = re.sub(r"\s+", " ", str(text or ""))
    text = "".join(ch for ch in text if ch.isprintable())
    return " ".join(text.split())


def recent_context():
    titles = read_lines(HISTORY, 150)
    logs = jsonl_read(IDEAS_LOG, 200)
    recent_animals = set()
    for t in titles[-ANIMAL_COOLDOWN:]:
        recent_animals.update(animals_in_text(t))
    for r in logs[-ANIMAL_COOLDOWN:]:
        if r.get("animal"):
            recent_animals.add(str(r["animal"]).lower())
    past_texts = list(titles) + [f"{r.get('animal', '')} {r.get('key_fact', '')}" for r in logs]
    recent_animals |= FAILED_ANIMALS
    recent_hooks = [r.get("hook_type") for r in logs[-2:]]
    return recent_animals, titles, past_texts, recent_hooks, logs


def _num(x, lo, hi):
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = 0.0
    return max(lo, min(hi, v))


def model_score(sc):
    sc = sc if isinstance(sc, dict) else {}
    return (_num(sc.get("curiosity"), 0, 25) + _num(sc.get("clarity"), 0, 20)
            + _num(sc.get("visual"), 0, 20) + _num(sc.get("story"), 0, 15)
            + _num(sc.get("science"), 0, 10))


def _extract_json(text):
    text = (text or "").strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in reply")
    s = min(starts)
    e = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[s:e + 1])


def generate_ideas(slot, allowed, avoid_titles, n=8):
    hooks = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(HOOK_TYPES))
    shape = {
        "animal": "one name from the list",
        "key_fact": "the single scientific fact, one sentence",
        "angle": "short angle label",
        "hook_type": "number 1-10 from the hook list",
        "hooks": ["3 different opening lines, max 10 words each"],
        "best_hook": "the best of the 3",
        "surprise": "the core surprise",
        "promise": "what the viewer will learn",
        "source_hint": "institution that documents it, e.g. NOAA",
        "precision": "exact | approximate | estimate",
        "footage_needed": "what real footage of the animal should show",
        "slot_fit": "clear_fact | behavior_story | comparison_puzzle",
        "why": "why this idea can hold attention",
        "scores": {"curiosity": "0-25", "clarity": "0-20", "visual": "0-20",
                   "story": "0-15", "science": "0-10"},
    }
    prompt = f"""You are the idea editor of "Tom Facts", an English YouTube Shorts channel about ocean animals and the sea (audience: USA, UK, Canada, Australia).
Create {n} DIFFERENT video ideas. Every idea must use a DIFFERENT animal chosen only from this list: {", ".join(allowed)}.
Today's slot: {SLOTS[slot]}.
Rules:
- Only well-documented facts that NOAA, Smithsonian Ocean, natural-history museums or peer-reviewed research state. No myths, no exaggeration, no "scientists say maybe".
- If a number is an estimate, say so in "precision". Avoid exact numbers that differ between sources. Be careful with wording (for example a swordfish has a bill, not a nose).
- Prefer a fact that is easy to picture with real footage of that animal.
- Do NOT repeat these recent videos: {avoid_titles[-40:]}
Hook types (choose the one that fits the fact best, do not choose randomly):
{hooks}
Return ONLY a JSON array of {n} objects with exactly this shape:
{json.dumps(shape)}"""
    data = gemini_json(prompt, temperature=1.0)
    if isinstance(data, dict):
        data = data.get("ideas") or next((v for v in data.values() if isinstance(v, list)), [])
    return [d for d in data if isinstance(d, dict)]


def rank_ideas(ideas, allowed, past_texts, recent_hooks, slot):
    """Internal ranking out of 100: model rubric (90) + novelty computed in code (10)."""
    allowed_set = {a.lower() for a in allowed}
    ranked = []
    for idea in ideas:
        animal = clean_text(idea.get("animal")).lower()
        fact = clean_text(idea.get("key_fact"))
        if animal not in allowed_set or not fact:
            continue
        try:
            ht = HOOK_TYPES[int(str(idea.get("hook_type")).strip().split(".")[0]) - 1]
        except (ValueError, IndexError):
            ht = HOOK_TYPES[1]
        hooks = [clean_text(h) for h in (idea.get("hooks") or []) if clean_text(h)][:3]
        best = clean_text(idea.get("best_hook")) or (hooks[0] if hooks else "")
        if not best:
            continue
        text = f"{animal} {fact}"
        max_sim = max([similarity(text, p) for p in past_texts] or [0.0])
        if max_sim >= SIM_LIMIT:
            continue  # same fact as something already published
        novelty = round(10 * (1 - max_sim), 1)
        base = model_score(idea.get("scores"))
        bonus = 4 if idea.get("slot_fit") == slot else 0
        penalty = 3 if ht in recent_hooks else 0
        idea.update({
            "animal": animal, "key_fact": fact, "hook_type": ht, "hooks": hooks, "best_hook": best,
            "angle": clean_text(idea.get("angle")),
            "score_breakdown": {"rubric_model": round(base, 1), "novelty_code": novelty,
                                "slot_bonus": bonus, "hook_repeat_penalty": penalty,
                                "total": round(base + novelty + bonus - penalty, 1)},
        })
        ranked.append(idea)
    ranked.sort(key=lambda i: -i["score_breakdown"]["total"])
    return ranked


_footage_cache = {}


def match_level(hit, animal):
    """exact = every word of the animal name is in the clip tags; family = only the last word."""
    words = [w for w in animal.lower().split() if len(w) > 2]
    tags = (hit.get("tags") or "").lower()
    if words and all(w in tags for w in words):
        return "exact"
    if words and words[-1] in tags:
        return "family"
    return "none"


def footage_count(animal, used):
    if animal in _footage_cache:
        return _footage_cache[animal]
    r = requests.get("https://pixabay.com/api/videos/",
                     params={"key": STOCK_KEY, "q": animal, "per_page": 50, "safesearch": "true"},
                     timeout=30)
    r.raise_for_status()
    levels = [match_level(h, animal) for h in r.json().get("hits", [])
              if str(h.get("id")) not in used]
    _footage_cache[animal] = (levels.count("exact"), levels.count("family"))
    return _footage_cache[animal]


def write_script(idea, slot):
    brief = {k: idea.get(k) for k in ("animal", "key_fact", "angle", "best_hook", "surprise",
                                      "promise", "precision", "footage_needed")}
    shape = {
        "title_candidates": ["3 accurate curiosity-driven titles, max 70 chars, naming the animal"],
        "title": "the best one; every word of it must be true",
        "scenes": [{
            "line": "the spoken sentence(s) of this scene",
            "animal": "the animal on screen, or 'ocean' if the shot is generic scenery",
            "visual": "what the viewer sees",
            "shot": "close-up | wide | macro | slow-motion",
            "query": "2-3 words to search stock footage, containing the animal name unless the scene is generic scenery",
        }],
        "interactive_question": "a specific question for the comments about THIS fact",
        "pinned_comment": "a short friendly pinned comment, max 200 characters",
        "series": "one of: " + " | ".join(SERIES),
        "tags": ["6 to 8 tags: ocean facts, sea creatures, plus the animal name and topic words"],
    }
    prompt = f"""You are Tom, narrator of the English YouTube Shorts channel "Tom Facts" (ocean animals).
Write ONE vertical video script about this idea: {json.dumps(brief)}
Hook type to use: {idea['hook_type']}. Opening line (you may polish it but keep its meaning): "{idea['best_hook']}".
Style: a real person talking to a friend. Short sentences, simple words, a little wonder or humor. No greeting, no "welcome back", no "did you know" opener, no emojis.
Order: 1) the hook, the strongest part first 2) the fact or the problem 3) escalation or the reason 4) the strongest reveal 5) a last line: a specific question about THIS fact, or a light invite to follow only if it fits. Never end with a generic "which animal next" question.
Length: 50 to 85 words in total (about 20 to 35 seconds spoken). Never invent numbers. If the fact is an estimate say "about" or "roughly".
Split the script into 4 to 6 scenes of one or two sentences each; every scene needs real footage that fits what is said.
Return ONLY JSON with exactly this shape:
{json.dumps(shape)}"""
    return gemini_json(prompt, temperature=0.9)


def validate_script(meta, idea, past_titles, past_scripts):
    scenes = []
    for s in (meta.get("scenes") or []):
        if isinstance(s, dict) and clean_text(s.get("line")):
            animal = clean_text(s.get("animal") or idea["animal"]).lower()
            scenes.append({
                "line": clean_text(s["line"]), "animal": animal,
                "visual": clean_text(s.get("visual")), "shot": clean_text(s.get("shot")),
                "query": clean_text(s.get("query")) or animal,
            })
    if not 3 <= len(scenes) <= 7:
        raise ValueError(f"{len(scenes)} scenes")
    script = " ".join(s["line"] for s in scenes)
    words = len(script.split())
    if not 40 <= words <= 110:
        raise ValueError(f"script has {words} words")
    title = clean_text(meta.get("title"))
    if not title or len(title) > 80:
        raise ValueError("bad title")
    if max([similarity(title, t) for t in past_titles] or [0.0]) >= SIM_LIMIT:
        raise ValueError("title too similar to an earlier video")
    if max([similarity(script, s) for s in past_scripts if s] or [0.0]) >= SIM_LIMIT:
        raise ValueError("script too similar to an earlier video")
    tags = [clean_text(t) for t in (meta.get("tags") or []) if clean_text(t)][:10]
    series = clean_text(meta.get("series"))
    meta.update({
        "title": title, "scenes": scenes, "script": script, "tags": tags or ["ocean facts", "sea creatures"],
        "series": series if series in SERIES else "",
        "interactive_question": clean_text(meta.get("interactive_question")),
        "pinned_comment": clean_text(meta.get("pinned_comment"))[:300],
        "title_candidates": [clean_text(t) for t in (meta.get("title_candidates") or [])][:3],
        "keywords": [s["query"] for s in scenes],
    })
    return meta


def gemini_grounded(prompt):
    """Gemini call WITH Google Search. Returns {"data","sources","model"} or None if unavailable."""
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.0}}
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json=body, timeout=120)
        except requests.RequestException as e:
            print("Grounded call error ->", model, repr(e))
            continue
        if r.status_code == 429:
            print("Google Search grounding: quota reached ->", model, "(skipping search for this run)")
            return None
        if r.status_code != 200:
            print("Grounded call failed ->", model, r.status_code, r.text[:150])
            continue
        try:
            cand = r.json()["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
            chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
            sources = [{"title": (c["web"].get("title") or ""), "uri": (c["web"].get("uri") or "")}
                       for c in chunks if c.get("web")]
            return {"data": _extract_json(text), "sources": sources, "model": model}
        except (KeyError, IndexError, ValueError) as e:
            print("Grounded reply unusable ->", model, repr(e))
    return None


def verify_script(meta, idea):
    """Claim-by-claim verification. Passes only if no claim is false or disputed and the main claim
    is established (or an estimate that the script words as approximate)."""
    prompt = f"""You are a meticulous marine-science fact-checker. Check this short video script against reliable sources (NOAA, Smithsonian, natural-history museums, universities, peer-reviewed research).
Title: {meta['title']}
Script: {meta['script']}
Check every claim, number, unit and comparison, and also the WORDING (for example, a swordfish has a bill, not a nose; do not accept a rough number worded as exact).
Status per claim: "established", "estimate" (true only approximately; the script must say so), "disputed", or "false".
Return ONLY JSON: {{"claims": [{{"claim": "...", "status": "...", "note": "short"}}], "main_claim_status": "established|estimate|disputed|false", "problems": "short reason or empty"}}"""
    mode, sources, data = "grounded", [], None
    res = gemini_grounded(prompt)
    if res and res["sources"]:
        data, sources = res["data"], res["sources"]
    else:
        mode = "model_knowledge_only"   # no live search available: NOT an external verification
        data = gemini_json(prompt, temperature=0.0)
    claims = data.get("claims") or []
    statuses = [str(c.get("status", "")).lower() for c in claims if isinstance(c, dict)]
    main = str(data.get("main_claim_status", "")).lower()
    problems = clean_text(data.get("problems"))
    base_ok = (main in ("established", "estimate") and bool(claims)
               and not any(st in ("false", "disputed") for st in statuses))
    # Without live search the check is only the model's memory, so ANY caveat it raises
    # (for example "depends on the species") must be fixed before publishing.
    ok = base_ok and (mode == "grounded" or not problems)
    return {"pass": ok, "base_ok": base_ok, "fixable": base_ok or "false" not in statuses,
            "mode": mode, "verified_externally": mode == "grounded",
            "claims": claims, "main_claim_status": main, "problems": problems,
            "sources": sources[:8], "checked_utc": now_utc().isoformat() + "Z"}


def revise_script(meta, idea, problems, past_titles, past_scripts):
    """One rewrite that fixes the over-generalisation the checker found (keeps scenes and footage fields)."""
    keep = [{k: s[k] for k in ("line", "animal", "visual", "shot", "query")} for s in meta["scenes"]]
    prompt = f"""You are Tom, narrator of the English YouTube Shorts channel "Tom Facts".
A fact-checker found this problem in the script: {problems}
Rewrite the title and the scene lines so every claim is accurate and properly hedged (use words like "many", "some species", "about" where needed). Keep the strong hook, the same number of scenes ({len(keep)}), and the same "animal", "visual", "shot" and "query" fields. Keep it 50 to 85 words in total and keep the final specific question. The title must also be fully true.
Title: {meta['title']}
Scenes: {json.dumps(keep)}
Return ONLY JSON: {{"title": "...", "scenes": [{{"line": "...", "animal": "...", "visual": "...", "shot": "...", "query": "..."}}]}}"""
    new = gemini_json(prompt, temperature=0.4)
    fixed = dict(meta)
    fixed["title"], fixed["scenes"] = new.get("title"), new.get("scenes")
    return validate_script(fixed, idea, past_titles, past_scripts)


def make_script_smart():
    slot = current_slot()
    recent_animals, past_titles, past_texts, recent_hooks, logs = recent_context()
    allowed = [a for a in ANIMALS if a not in recent_animals] or list(ANIMALS)
    ideas = generate_ideas(slot, allowed, past_titles)
    ranked = rank_ideas(ideas, allowed, past_texts, recent_hooks, slot)
    print(f"Slot: {slot} | ideas: {len(ideas)} | usable: {len(ranked)}")
    if not ranked:
        raise RuntimeError("no usable ideas")
    used = set(read_lines(USED_CLIPS, 2000))
    chosen = []
    for idea in ranked:
        exact, family = footage_count(idea["animal"], used)
        idea["footage"] = {"exact": exact, "family": family}
        print(f"  idea {idea['score_breakdown']['total']:>5} {idea['animal']}: footage exact={exact}")
        if exact >= 2:
            chosen.append(idea)
        if len(chosen) >= 3:
            break
    if not chosen:
        raise RuntimeError("no idea has footage that really shows its animal")
    past_scripts = [r.get("script", "") for r in logs]
    for idea in chosen:
        meta = write_script(idea, slot)
        try:
            meta = validate_script(meta, idea, past_titles, past_scripts)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            print("Idea rejected (script check):", idea["animal"], "-", e)
            jsonl_append(REVIEW_QUEUE, {"when": now_utc().isoformat() + "Z", "reason": f"script check: {e}",
                                        "idea": idea})
            continue
        ver = verify_script(meta, idea)
        print("Verification:", ver["mode"], "pass" if ver["pass"] else "FAIL", ver["problems"])
        if not ver["pass"] and ver["fixable"]:
            try:   # one rewrite, then check again (a second check only needs no false/disputed claim)
                meta = revise_script(meta, idea, ver["problems"] or "the main claim is too general", past_titles, past_scripts)
                ver = verify_script(meta, idea)
                ver["pass"], ver["revised"] = ver["base_ok"], True
                print("After revision:", "pass" if ver["pass"] else "FAIL", ver["problems"])
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                print("Revision failed:", e)
        if not ver["pass"]:
            jsonl_append(REVIEW_QUEUE, {"when": now_utc().isoformat() + "Z", "reason": "fact verification",
                                        "title": meta["title"], "script": meta["script"], "verification": ver})
            continue
        meta.update({"engine": "smart", "slot": slot, "idea": idea, "animal": idea["animal"],
                     "key_fact": idea["key_fact"], "angle": idea["angle"],
                     "hook_type": idea["hook_type"], "hook": idea["best_hook"],
                     "verification": ver})
        return meta
    raise VerificationFailed("no idea passed the script checks and the fact verification")


def make_script():
    if not SMART_ENGINE:
        return make_script_legacy()
    try:
        return make_script_smart()
    except VerificationFailed:
        raise  # never publish something unverified just to have a video
    except Exception as e:
        print("Smart engine failed, using the legacy generator:", repr(e))
        return make_script_legacy()


async def _tts(text, path):
    try:
        await edge_tts.Communicate(text, VOICE, rate="+5%").save(path)
    except Exception as e:
        print("edge-tts failed, using gTTS:", e)
        from gtts import gTTS
        gTTS(text, lang="en").save(path)


def get_clip(query, i, used, strict=False):
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": STOCK_KEY, "q": query, "per_page": 30, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    hits = [h for h in r.json().get("hits", []) if str(h.get("id")) not in used]
    random.shuffle(hits)
    words = [w for w in query.lower().split() if len(w) > 2]

    def score(h):
        tags = (h.get("tags") or "").lower()
        return sum(1 for w in words if w in tags)

    hits.sort(key=lambda h: -score(h))  # clips whose tags match the animal come first
    if strict:
        hits = [h for h in hits if score(h) > 0]  # never use an unrelated clip for the animal itself
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


def _download(url, path):
    with requests.get(url, stream=True, timeout=120) as dl:
        dl.raise_for_status()
        with open(path, "wb") as out:
            for chunk in dl.iter_content(1 << 20):
                out.write(chunk)


def find_clip(query, i, used, animal=None):
    """Download a clip for scene i. With `animal`, ONLY clips whose tags name that animal are used.
    Returns (path, info) or (None, None). info keeps the source page and licence of the clip."""
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": STOCK_KEY, "q": query, "per_page": 40, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    hits = [h for h in r.json().get("hits", []) if str(h.get("id")) not in used]
    random.shuffle(hits)
    for h in hits:
        level = match_level(h, animal) if animal else "generic"
        if animal and level != "exact":
            continue
        url = None
        for size in ("large", "medium", "small"):
            d = h["videos"].get(size) or {}
            if d.get("url") and (d.get("width") or 0) <= 1920:
                url = d["url"]
                break
        if not url:
            continue
        path = f"clip{i}.mp4"
        _download(url, path)
        used.add(str(h["id"]))
        with open(USED_CLIPS, "a", encoding="utf-8") as f:
            f.write(str(h["id"]) + "\n")
        return path, {"clip_id": str(h["id"]), "page_url": h.get("pageURL", ""),
                      "license": "Pixabay Content License", "query_used": query}
    return None, None


def build_scene_clips(meta, audio_dur, total, used):
    """One clip per scene, timed to the spoken words. Returns (clips, timed_captions)."""
    scenes = meta["scenes"]
    words = [max(1, len(s["line"].split())) for s in scenes]
    core = [audio_dur * w / sum(words) for w in words]
    clips, timed, log, t = [], [], [], 0.0
    animal_scenes = exact_scenes = 0
    for i, (s, d) in enumerate(zip(scenes, core)):
        animal = s["animal"]
        generic = animal in GENERIC_SUBJECTS
        path = info = None
        queries = list(dict.fromkeys([s["query"], animal] if not generic else [s["query"], "deep ocean"]))
        for q in queries:
            path, info = find_clip(q, i, used, animal=None if generic else animal)
            if path:
                break
        if generic:
            match = "generic_by_design"
        else:
            animal_scenes += 1
            if path:
                match = "exact"
                exact_scenes += 1
            else:
                match = "fallback_generic"   # honest label: the animal is NOT on screen here
        if not path:
            path, info = find_clip("underwater ocean", i, used)
        if not path:
            raise MatchFailure(f"no footage at all for scene {i + 1}")
        seg = d + (total - audio_dur if i == len(scenes) - 1 else 0)
        clips.append(fit(VideoFileClip(path), seg))
        timed.append((t, d, s["line"]))
        log.append({"line": s["line"], "animal": animal, "visual": s["visual"], "shot": s["shot"],
                    "duration_s": round(seg, 1), "match": match, **(info or {})})
        t += seg
    meta["scene_log"] = log
    if animal_scenes and exact_scenes / animal_scenes < MIN_EXACT_RATIO:
        raise MatchFailure(f"only {exact_scenes}/{animal_scenes} scenes show the real animal")
    return clips, timed


def add_subs_timed(video, timed):
    """Captions synchronised with the scene timing (3-4 words at a time)."""
    try:
        subs = []
        for start, dur, text in timed:
            words = clean_text(text).split()
            parts = [" ".join(words[i:i + 4]) for i in range(0, len(words), 4)]
            if len(parts) > 1 and len(parts[-1].split()) == 1:
                parts[-2] += " " + parts.pop()
            total = sum(len(p) for p in parts) or 1
            t = start
            for p in parts:
                d = dur * len(p) / total
                subs.append(ImageClip(_caption_image(p)).set_start(t).set_duration(d)
                            .set_position(("center", 1100)))
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
    meta["audio_duration"] = round(audio.duration, 1)
    used = set(read_lines(USED_CLIPS, 2000))
    if meta.get("scenes"):
        # smart engine: one clip per scene, timed to the words, captions synced to the scenes
        clips, timed = build_scene_clips(meta, audio.duration, total, used)
        video = concatenate_videoclips(clips).set_audio(add_music(audio, total))
        video = add_subs_timed(video, timed)
    else:
        paths = []
        for i, k in enumerate(meta["keywords"]):
            p = get_clip(k, i, used, strict=(i < 2)) or get_clip("underwater ocean", i, used)
            if p:
                paths.append(p)
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
            "description": build_description(meta),
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
    return resp["id"]


def build_description(meta):
    parts = [meta["script"]]
    if meta.get("interactive_question"):
        parts.append(meta["interactive_question"])
    if meta.get("series"):
        parts.append("Series: " + meta["series"])
    parts.append(random.choice(CTA_LINES))
    tag = re.sub(r"[^A-Za-z0-9]", "", str(meta.get("animal", "")).title())
    parts.append("#Shorts #OceanFacts #SeaCreatures #DidYouKnow" + (f" #{tag}" if tag else ""))
    return "\n\n".join(parts)[:4900]


VIDEO_COLUMNS = ["video_id", "url", "published_utc", "title", "animal", "angle", "hook_type", "series",
                 "slot", "engine", "duration_s", "score", "exact_match_ratio", "views", "likes", "comments",
                 "avg_view_duration_s", "avg_view_pct", "stayed_to_watch_pct", "subs_gained", "stats_updated_utc"]


def _read_rows():
    if not os.path.exists(VIDEO_LOG):
        return []
    with open(VIDEO_LOG, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(rows):
    with open(VIDEO_LOG, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def log_video(meta, video_id):
    """Save the full video card (jsonl) + one performance row (csv) + a copy-ready card (markdown)."""
    when = now_utc().isoformat() + "Z"
    scenes = meta.get("scene_log") or []
    animal_scenes = [s for s in scenes if s["match"] in ("exact", "fallback_generic")]
    ratio = (round(sum(1 for s in animal_scenes if s["match"] == "exact") / len(animal_scenes), 2)
             if animal_scenes else "")
    ver = meta.get("verification") or {}
    card = {
        "video_id": video_id, "url": f"https://youtube.com/shorts/{video_id}", "published_utc": when,
        "status": "published", "engine": meta.get("engine", "legacy"), "slot": meta.get("slot", ""),
        "animal": meta.get("animal", ""), "key_fact": meta.get("key_fact", ""), "angle": meta.get("angle", ""),
        "hook_type": meta.get("hook_type", ""), "hook": meta.get("hook", ""), "series": meta.get("series", ""),
        "title_candidates": meta.get("title_candidates", []), "title": meta["title"], "script": meta["script"],
        "scenes": scenes, "tags": meta.get("tags", []), "description": build_description(meta),
        "pinned_comment": meta.get("pinned_comment", ""), "interactive_question": meta.get("interactive_question", ""),
        "idea_score": (meta.get("idea") or {}).get("score_breakdown", {}),
        "why_chosen": (meta.get("idea") or {}).get("why", ""),
        "verification": {k: ver.get(k) for k in ("mode", "verified_externally", "main_claim_status",
                                                  "problems", "sources", "checked_utc", "claims")},
        "duration_s": meta.get("audio_duration", ""),
    }
    jsonl_append(IDEAS_LOG, card)
    rows = _read_rows()
    rows.append({"video_id": video_id, "url": card["url"], "published_utc": when, "title": meta["title"],
                 "animal": card["animal"], "angle": card["angle"], "hook_type": card["hook_type"],
                 "series": card["series"], "slot": card["slot"], "engine": card["engine"],
                 "duration_s": card["duration_s"], "score": card["idea_score"].get("total", ""),
                 "exact_match_ratio": ratio})
    _write_rows(rows)
    lines = [f"# {meta['title']}", "", f"{card['url']}  |  slot: {card['slot']}  |  series: {card['series']}",
             "", f"**Idea:** {card['key_fact']}", f"**Hook ({card['hook_type']}):** {card['hook']}",
             f"**Why chosen:** {card['why_chosen']}", "", "## Script", meta["script"], "",
             "## Pinned comment (paste it yourself, YouTube has no API to pin)", card["pinned_comment"], "",
             "## Question", card["interactive_question"], "",
             f"## Verification: {ver.get('mode', 'legacy check')} "
             f"(external: {ver.get('verified_externally', False)})"]
    lines += [f"- {s.get('title', '')} {s.get('uri', '')}" for s in ver.get("sources", [])]
    lines += ["", "## Scenes"] + [f"{i + 1}. [{s['match']}] {s['line']}  ->  {s.get('page_url', '')}"
                                   for i, s in enumerate(scenes)]
    with open(CARD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fetch_public_stats(ids):
    """Public numbers (views/likes/comments) with a free YouTube Data API key (secret YT_API_KEY)."""
    key = (os.environ.get("YT_API_KEY") or "").strip()
    if not key:
        return {}, "no YT_API_KEY secret: views were not refreshed automatically"
    yt = build("youtube", "v3", developerKey=key)
    out = {}
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(part="statistics", id=",".join(ids[i:i + 50])).execute()
        for it in resp.get("items", []):
            s = it.get("statistics", {})
            out[it["id"]] = {"views": s.get("viewCount", ""), "likes": s.get("likeCount", ""),
                             "comments": s.get("commentCount", "")}
    return out, "YouTube Data API (public statistics)"


def _age_days(row):
    try:
        return (now_utc() - datetime.datetime.fromisoformat(row["published_utc"].rstrip("Z"))).days
    except (ValueError, KeyError, AttributeError):
        return 0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_report(rows, source_note):
    measured = [r for r in rows if _f(r.get("views")) is not None and _age_days(r) >= 2]
    out = ["# Tom Facts performance report", "", f"Updated: {now_utc().isoformat()}Z",
           f"Numbers source: {source_note}",
           f"Videos logged: {len(rows)} | measured (at least 2 days old): {len(measured)}", ""]
    if len(measured) < 10:
        out += ["> Too little data. Read the tables below as observations, never as rules.", ""]

    def bucket(r):
        d = _f(r.get("duration_s"))
        return "unknown" if d is None else ("under 25s" if d < 25 else ("25-35s" if d <= 35 else "over 35s"))

    for field, label in [("hook_type", "Hook type"), ("animal", "Animal"), ("series", "Series"),
                         ("slot", "Slot"), ("length", "Length")]:
        groups = {}
        for r in measured:
            groups.setdefault(bucket(r) if field == "length" else (r.get(field) or "n/a"), []).append(r)
        if not groups:
            continue
        out += [f"## By {label}", "", f"| {label} | videos | avg views | avg % watched | subs per 1000 views |",
                "|---|---|---|---|---|"]
        for name, g in sorted(groups.items(), key=lambda kv: -sum(_f(r["views"]) for r in kv[1]) / len(kv[1])):
            views = [_f(r["views"]) for r in g]
            pct = [_f(r.get("avg_view_pct")) for r in g if _f(r.get("avg_view_pct")) is not None]
            subs = [_f(r.get("subs_gained")) for r in g if _f(r.get("subs_gained")) is not None]
            tv = sum(views)
            sub_txt = f"{1000 * sum(subs) / tv:.1f}" if subs and tv else "n/a"
            flag = " (too few to conclude)" if len(g) < 3 else ""
            out.append(f"| {name}{flag} | {len(g)} | {tv / len(g):.0f} | "
                       f"{(sum(pct) / len(pct)):.0f}% | {sub_txt} |" if pct else
                       f"| {name}{flag} | {len(g)} | {tv / len(g):.0f} | n/a | {sub_txt} |")
        out.append("")
    if measured:
        ranked = sorted(measured, key=lambda r: -_f(r["views"]))
        out += ["## Best videos", ""] + [f"- {r['views']} views: {r['title']} ({r.get('hook_type', '')})"
                                         for r in ranked[:5]]
        out += ["", "## Weakest videos (reasons are hypotheses to check, not facts)", ""]
        for r in ranked[-3:]:
            hints = []
            if _f(r.get("exact_match_ratio")) is not None and _f(r["exact_match_ratio"]) < 0.8:
                hints.append("footage did not always show the animal")
            if _f(r.get("avg_view_pct")) is not None and _f(r["avg_view_pct"]) < 60:
                hints.append("viewers left early (weak hook or slow middle)")
            out.append(f"- {r['views']} views: {r['title']} | " + ("; ".join(hints) or "no clear signal yet"))
    out += ["", "Retention, swipe-away and subscriber numbers are not available through the free API used here. "
            "Copy them from YouTube Studio into stats_manual.csv (columns: video_id, avg_view_duration_s, "
            "avg_view_pct, stayed_to_watch_pct, subs_gained) and they are merged on the next run."]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def update_analytics():
    rows = _read_rows()
    if not rows:
        return
    stats, note = {}, ""
    try:
        stats, note = fetch_public_stats([r["video_id"] for r in rows])
    except Exception as e:   # analytics must never break publishing
        note = f"automatic refresh failed: {e!r}"
    stamp = now_utc().isoformat() + "Z"
    for r in rows:
        if r["video_id"] in stats:
            r.update(stats[r["video_id"]])
            r["stats_updated_utc"] = stamp
    if os.path.exists(STATS_MANUAL):
        with open(STATS_MANUAL, encoding="utf-8", newline="") as f:
            manual = {m["video_id"]: m for m in csv.DictReader(f) if m.get("video_id")}
        for r in rows:
            for k, v in (manual.get(r["video_id"]) or {}).items():
                if k != "video_id" and str(v).strip() != "":
                    r[k] = v
        note += " + manual stats_manual.csv"
    _write_rows(rows)
    build_report(rows, note)


if __name__ == "__main__":
    creds = get_creds()
    video_path = None
    for attempt in range(2):
        meta = make_script()
        print("Title:", meta["title"])
        try:
            video_path = build_video(meta)
            break
        except MatchFailure as e:
            print("Footage check failed:", e)
            FAILED_ANIMALS.add(str(meta.get("animal", "")).lower())
            jsonl_append(REVIEW_QUEUE, {"when": now_utc().isoformat() + "Z", "reason": f"footage: {e}",
                                        "title": meta["title"], "animal": meta.get("animal", "")})
    if not video_path:
        raise RuntimeError("No video passed the footage check (see review_queue.jsonl)")
    video_id = upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(meta["title"] + "\n")
    try:
        log_video(meta, video_id)
    except Exception as e:
        print("Video log skipped:", repr(e))
    try:
        update_analytics()
    except Exception as e:
        print("Analytics skipped:", repr(e))
