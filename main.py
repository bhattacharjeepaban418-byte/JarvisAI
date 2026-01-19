## jarvis_v3.py
"""
Jarvis Ultimate v2 (safe learning + Chrome automation + multilingual + teach mode + adds embedding memory + daily transcripts)

Features:
- Continuous voice listening + non-blocking TTS
- Uses Ollama (gemma3:1b) for AI responses & code generation
- Selenium-driven Chrome search/open + voices confirmations + safe download flow
- Local memory (JarvisMemory.json) logging queries, searches, downloads, generated files
- "Write a code" -> generate code, save in JarvisProjects, open in VS Code
- "Teach me <language>" -> generate lessons and save as .txt and open in Notepad
- Language detection via langdetect and AI-based translation/responses
- Does NOT bypass or remove safety filters

Adjust CHROMEDRIVER_PATH and VS_CODE_PATH if needed.
"""

import os
import json
import time
import datetime
import subprocess
import webbrowser
from pathlib import Path

import speech_recognition as sr
from langdetect import detect, LangDetectException
import wikipedia
import ollama
from PIL import ImageGrab
import pyttsx3
import threading
# -------------------------------
# CHROME DRIVER SETUP
# -------------------------------
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import subprocess

import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import comtypes.client


# ------------------ Spotify Control Setup ------------------
import spotipy
from spotipy.oauth2 import SpotifyOAuth

SPOTIFY_CLIENT_ID = "01218bb85bb24f788f1e0b9516e681c7"
SPOTIFY_CLIENT_SECRET = "41bdeca7c4464d3abcaab1c6cfc67aaf"
SPOTIFY_REDIRECT_URI = "https://open.spotify.com/?utm_source=pwa_install"

driver = None
last_command = None


# Spotify scopes (permissions)
SPOTIFY_SCOPE = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "user-library-read "
    "playlist-read-private "
    "playlist-read-collaborative "
    "user-top-read"
)

sp = None
try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE
    ))
    print("✅ Spotify connected.")
except Exception as e:
    print("⚠️ Spotify connection failed:", e)



# ----- Optional embedding libs -----
USE_SENT_TRANSFORMERS = True
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("SentenceTransformers not available or failed to load — falling back to TF-IDF. Error:", e)
    USE_SENT_TRANSFORMERS = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

# ------------------ Config ------------------
BASE_DIR = Path.cwd()
PROJECTS_DIR = BASE_DIR / "JarvisProjects"
PROJECTS_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR = PROJECTS_DIR / "Downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR = PROJECTS_DIR / "Transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

MEMORY_FILE = BASE_DIR / "JarvisMemory.json"
CHROME_PROFILE = None  # or r"C:\Users\<you>\AppData\Local\Google\Chrome\User Data"
CHROMEDRIVER_PATH = "chromedriver"  # or full path
VS_CODE_PATH = os.path.expandvars(r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe")
NOTEPAD_EXE = "notepad.exe"
OLLAMA_MODEL = "gemma3:1b"

# ------------------ Load / Init Memory ------------------
if MEMORY_FILE.exists():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except Exception:
        memory = {"entries": []}
else:
    memory = {"entries": []}

def persist_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

# If using TF-IDF fallback, maintain vectorizer state in memory as well
if not USE_SENT_TRANSFORMERS:
    # store raw texts and vectorizer will be fit on demand
    memory.setdefault("tfidf_texts", [])

# ------------------ TTS (Safe Non-blocking Speech) ------------------
speak_lock = threading.Lock()

def say(text: str, block: bool = False):
    """Speak text safely — reinitializes engine each time to prevent silence after first use."""
    print("Jarvis:", text)

    def _speak_inner(message):
        with speak_lock:
            engine = pyttsx3.init('sapi5')
            engine.setProperty('rate', 165)
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[0].id)  # 0 = male, 1 = female
            engine.say(message)
            engine.runAndWait()
            engine.stop()

    thread = threading.Thread(target=_speak_inner, args=(text,), daemon=True)
    thread.start()
    if block:
        thread.join()


# ------------------ Selenium Chrome helper ------------------
def make_chrome_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    if CHROME_PROFILE:
        options.add_argument(f"--user-data-dir={CHROME_PROFILE}")
    prefs = {"download.default_directory": str(DOWNLOADS_DIR), "profile.default_content_setting_values.automatic_downloads": 1}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=options)
    return driver

# ------------------ Ollama helpers ------------------
def ai_chat(prompt, max_tokens=512):
    try:
        resp = ollama.chat(model=OLLAMA_MODEL,
                           messages=[{"role":"system","content":"You are Jarvis, a helpful assistant."},
                                     {"role":"user","content":prompt}],
                           options={"num_predict": max_tokens})
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        print("Ollama error:", e)
        return None

def generate_code(prompt):
    return ai_chat(f"Write a complete, runnable code for: {prompt}\nProvide only the code block (no extra commentary).", max_tokens=1024)

# ------------------ Embedding & Semantic Memory ------------------
def embed_text(text):
    if USE_SENT_TRANSFORMERS:
        vec = sbert_model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()
    else:
        # TF-IDF fallback: vectors computed on-demand during search
        return None

def add_memory_entry(kind, content, metadata=None):
    entry = {
        "id": int(time.time()*1000),
        "timestamp": datetime.datetime.now().isoformat(),
        "type": kind,
        "content": content,
        "metadata": metadata or {}
    }
    # embed (if available)
    vec = embed_text(content)
    if vec is not None:
        entry["embedding"] = vec
    memory.setdefault("entries", []).append(entry)
    if not USE_SENT_TRANSFORMERS:
        memory.setdefault("tfidf_texts", []).append(content)
    persist_memory()

def build_tfidf_matrix():
    texts = memory.get("tfidf_texts", [])
    if not texts:
        return None, None
    vectorizer = TfidfVectorizer().fit(texts)
    mat = vectorizer.transform(texts).toarray()
    return vectorizer, mat

def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_search(query, k=5):
    """Return top-k memory entries relevant to query."""
    results = []
    if USE_SENT_TRANSFORMERS:
        qvec = sbert_model.encode([query], normalize_embeddings=True)[0]
        for e in memory.get("entries", []):
            emb = e.get("embedding")
            if emb:
                sim = cosine_sim(qvec, emb)
                results.append((sim, e))
    else:
        # TF-IDF fallback: fit vectorizer and compute cosine similarity with stored texts
        vectorizer, mat = build_tfidf_matrix()
        if vectorizer is None:
            return []
        qvec = vectorizer.transform([query]).toarray()[0]
        for idx, e in enumerate(memory.get("entries", [])):
            text_vec = mat[idx]
            sim = cosine_sim(qvec, text_vec)
            results.append((sim, e))
    results.sort(key=lambda x: x[0], reverse=True)
    topk = [r[1] for r in results[:k]]
    return topk

# ------------------ Transcripts ------------------
def append_transcript(user_text, assistant_text=None, action=None):
    date_str = datetime.date.today().isoformat()
    file_path = TRANSCRIPTS_DIR / f"{date_str}.txt"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] USER: {user_text}\n")
        if assistant_text:
            f.write(f"[{ts}] JARVIS: {assistant_text}\n")
        if action:
            f.write(f"[{ts}] ACTION: {action}\n")
        f.write("\n")

# ------------------ Utilities ------------------
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

def log_and_respond(user_text, response_text=None, action=None):
    # memory entry
    add_memory_entry("interaction", {"user": user_text, "assistant": response_text or ""})
    append_transcript(user_text, response_text, action)

# ------------------ Browser helpers (search & download) ------------------
def make_chrome_driver():
    """Create a Chrome WebDriver instance using the Service class (no executable_path)."""
    chrome_driver_path = r"C:\Program Files\Google\Chrome\Application\chromedriver.exe"
    if not os.path.exists(chrome_driver_path):
        say("Chrome driver not found, sir. Please install ChromeDriver or update the path.")
        raise FileNotFoundError("ChromeDriver not found")

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


# -------------------------------
# CHROME SEARCH FUNCTION
# -------------------------------
def chrome_search_and_open(query, retries=2, keep_open=True):
    """
    Opens Chrome, searches the given query, and keeps the tab open
    until the user or Jarvis explicitly closes it.
    Returns (results, driver) so handle_query() stays compatible.
    """
    say(f"Jarvis: Searching Chrome for {query}")
    driver = None

    for attempt in range(1, retries + 1):
        try:
            options = Options()
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--start-maximized")
            prefs = {
                "download.default_directory": str(DOWNLOADS_DIR),
                "profile.default_content_setting_values.automatic_downloads": 1
            }
            options.add_experimental_option("prefs", prefs)
            if CHROME_PROFILE:
                options.add_argument(f"--user-data-dir={CHROME_PROFILE}")

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            say("Opening browser...")
            print(f"Attempt {attempt}: Launching Chrome for query '{query}'")

            if "youtube" in query.lower():
                search_term = query.lower().replace("youtube", "").replace("and play it", "").strip()
                driver.get(f"https://www.youtube.com/results?search_query={search_term}")
                say(f"Playing {search_term} on YouTube.")
                # Auto-play first video
                try:
                    time.sleep(3)
                    first_video = driver.find_element("xpath", '(//a[@id="video-title"])[1]')
                    first_video.click()
                except Exception:
                    say("Couldn't auto-play the first video, sir.")
            else:
                driver.get(f"https://www.google.com/search?q={query}")

            say("Search completed, sir.")
            print("✅ Chrome opened and will stay open.")

            if keep_open:
                return [("Search executed", driver.current_url)], driver

            time.sleep(10)
            driver.quit()
            return [("Search executed", driver.current_url)], None

        except Exception as e:
            print(f"⚠️ Chrome search attempt {attempt} failed: {e}")
            say(f"Search attempt {attempt} failed. Retrying...")
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    say("Sorry sir, I was unable to complete the search after several attempts.")
    print("❌ All attempts to open Chrome failed.")
    return [], None



def confirm_and_download(driver, file_url=None):
    if not driver:
        say("No browser session available to download from.")
        return False
    candidate = None
    if file_url:
        candidate = file_url
    else:
        anchors = driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            href = a.get_attribute("href")
            if href and any(href.lower().endswith(ext) for ext in [".exe", ".msi", ".zip", ".tar.gz", ".whl", ".pdf"]):
                candidate = href
                break
    if not candidate:
        say("I couldn't find a downloadable file on this page, sir.")
        return False
    say(f"I found a file: {candidate}. Should I download it? Say yes or no.", block=False)
    confirmed = listen_for_confirmation(timeout=8)
    if confirmed:
        try:
            driver.get(candidate)
            add_memory_entry("download", {"url": candidate, "status": "started"})
            say("Download started and will be saved to JarvisProjects/Downloads.", block=False)
            return True
        except Exception as e:
            print("Download navigation error:", e)
            say("Failed to start download, sir.")
            return False
    else:
        say("Download cancelled, sir.")
        return False

# ------------------ Listening helpers ------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_once(timeout=8, phrase_time_limit=8):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.4)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print("Listen error:", e)
            return None

def listen_for_confirmation(timeout=6):
    txt = listen_once(timeout=timeout, phrase_time_limit=4)
    if not txt:
        return False
    txt = txt.lower()
    if any(w in txt for w in ["yes", "yeah", "yup", "sure", "do it", "download", "okay", "ok"]):
        return True
    return False

# ------------------ Teach & Code helpers ------------------
def teach_language(language, topic=None):
    prompt = f"Create a beginner lesson for learning {language}. Include pronunciation notes, common phrases, a short grammar overview, and 10 practice exercises."
    if topic:
        prompt += f" Focus on topic: {topic}."
    say(f"Generating your {language} lesson, sir.")
    lesson = ai_chat(prompt)
    if not lesson:
        say("I couldn't generate the lesson now.")
        return
    project_path = PROJECTS_DIR / f"teach_{language}_{int(time.time())}"
    files_path = project_path / "files"
    files_path.mkdir(parents=True, exist_ok=True)
    lesson_file = files_path / f"{language}_lesson.txt"
    lesson_file.write_text(lesson, encoding="utf-8")
    add_memory_entry("teach", {"language": language, "file": str(lesson_file)})
    append_transcript(f"Teach request: {language} {topic or ''}", lesson, action=f"Saved lesson to {lesson_file}")
    say(f"Lesson saved to {lesson_file}. Opening in Notepad, sir.")
    subprocess.Popen([NOTEPAD_EXE, str(lesson_file)])

# ------------------ Code Project Writer (Enhanced from Jarvis v4) ------------------
def write_project_code(language: str, idea: str):
    """Creates a folder, writes code, and opens it in VS Code."""
    say(f"Creating {language} project for {idea}, sir.")

    # Folder setup
    base_dir = PROJECTS_DIR
    folder_name = "_".join(idea.lower().split()[:3])
    project_path = base_dir / folder_name
    project_path.mkdir(exist_ok=True)

    # Choose filename and code content
    if language.lower() == "python":
        filename = project_path / "main.py"
        code = generate_python_template(idea)
    elif language.lower() in ["html", "web", "website"]:
        filename = project_path / "index.html"
        code = generate_html_template(idea)
    elif language.lower() in ["javascript", "js"]:
        filename = project_path / "script.js"
        code = generate_js_template(idea)
    else:
        filename = project_path / "main.txt"
        code = f"# Idea: {idea}\n# Language: {language}\n# Describe your project here."

    filename.write_text(code, encoding="utf-8")

    # Log to memory and transcripts
    add_memory_entry("code_project", {"language": language, "idea": idea, "file": str(filename)})
    append_transcript(f"Code project: {idea}", action=f"Saved {filename}")
    say(f"{language} project for {idea} created successfully.")
    subprocess.Popen([VS_CODE_PATH, str(project_path)], shell=True)

def generate_python_template(idea: str) -> str:
    """Basic Python templates."""
    if "calculator" in idea.lower():
        return """def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return a / b if b != 0 else 'Error'

print('Simple Calculator')
a = float(input('Enter first number: '))
b = float(input('Enter second number: '))
op = input('Enter operation (+ - * /): ')

if op == '+': print(add(a,b))
elif op == '-': print(subtract(a,b))
elif op == '*': print(multiply(a,b))
elif op == '/': print(divide(a,b))
else: print('Invalid operator')
"""
    elif "game" in idea.lower():
        return """import random
print('Number Guessing Game')
num = random.randint(1, 10)
guess = int(input('Guess a number 1-10: '))
if guess == num:
    print('Correct!')
else:
    print(f'Wrong! It was {num}')
"""
    else:
        return f"# Python project generated by Jarvis\nprint('Project idea: {idea}')"

def generate_html_template(idea: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>{idea.title()}</title>
</head>
<body>
  <h1>{idea.title()}</h1>
  <p>This page was generated by Jarvis.</p>
</body>
</html>"""

def generate_js_template(idea: str) -> str:
    return f"""// {idea.title()} Project
console.log("Project: {idea}");
alert("Hello from your {idea} project!");
"""

# -----------------------------
# SYSTEM CONTROL FUNCTIONS
# -----------------------------

def set_brightness(level):
    """Set brightness (0–100)"""
    try:
        level = int(level)
        level = max(0, min(100, level))
        sbc.set_brightness(level)
        say(f"Brightness set to {level} percent, sir.")
    except Exception as e:
        say("Failed to adjust brightness, sir.")
        print("Brightness error:", e)

def change_brightness(direction):
    """Increase or decrease brightness"""
    try:
        current = sbc.get_brightness(display=0)[0]
        if direction == "up":
            new = min(100, current + 10)
        else:
            new = max(0, current - 10)
        sbc.set_brightness(new)
        say(f"Brightness adjusted to {new} percent, sir.")
    except Exception as e:
        say("Couldn’t change brightness, sir.")
        print("Brightness change error:", e)


def set_volume(level):
    """Set volume (0–100)"""
    try:
        level = float(level) / 100
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMasterVolume(level, None)
        say(f"Volume set to {int(level * 100)} percent, sir.")
    except Exception as e:
        say("Failed to set volume, sir.")
        print("Volume error:", e)


def change_volume(direction):
    """Increase or decrease system volume"""
    try:
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            current = volume.GetMasterVolume()
            if direction == "up":
                new = min(1.0, current + 0.1)
            else:
                new = max(0.0, current - 0.1)
            volume.SetMasterVolume(new, None)
        say(f"Volume {'increased' if direction == 'up' else 'decreased'}, sir.")
    except Exception as e:
        say("Couldn’t adjust volume, sir.")
        print("Volume change error:", e)


def mute_volume():
    """Mute system volume"""
    try:
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMute(1, None)
        say("Volume muted, sir.")
    except Exception as e:
        say("Couldn’t mute volume, sir.")
        print("Mute error:", e)


# ------------------ Spotify Voice Handler ------------------
# ---------------------------------------------
# ADVANCED SPOTIFY CONTROL (Premium required)
# ---------------------------------------------
def handle_spotify_command(command):
    """Handle Spotify commands using the official API (Premium required)."""
    say("Jarvis: Connecting to Spotify...")

    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id="01218bb85bb24f788f1e0b9516e681c7",
            client_secret="41bdeca7c4464d3abcaab1c6cfc67aaf",
            redirect_uri="https://open.spotify.com/?utm_source=pwa_install",
            scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing,playlist-read-private,user-library-read"
        ))

        # Ensure Spotify Premium is available
        user = sp.current_user()
        if not user:
            say("I couldn’t connect to Spotify, sir.")
            return
        say(f"Connected to Spotify as {user['display_name']}.")

        cmd = command.lower()

        # Handle "play song" command
        if "play" in cmd:
            song = cmd.split("play", 1)[1].strip()
            if song:
                results = sp.search(q=song, type="track", limit=1)
                if results["tracks"]["items"]:
                    track_uri = results["tracks"]["items"][0]["uri"]
                    sp.start_playback(uris=[track_uri])
                    say(f"Now playing {song}, sir.")
                else:
                    say(f"I couldn’t find {song} on Spotify.")
            else:
                sp.start_playback()
                say("Resuming playback, sir.")

        elif "pause" in cmd:
            sp.pause_playback()
            say("Playback paused, sir.")

        elif "resume" in cmd or "continue" in cmd:
            sp.start_playback()
            say("Resumed playback, sir.")

        elif "next" in cmd:
            sp.next_track()
            say("Skipped to the next track, sir.")

        elif "previous" in cmd or "back" in cmd:
            sp.previous_track()
            say("Went back to the previous track, sir.")

        elif "volume" in cmd:
            import re
            level = re.findall(r'\d+', cmd)
            if level:
                vol = int(level[0])
                sp.volume(vol)
                say(f"Set the volume to {vol}%, sir.")
            else:
                say("Please tell me the volume level, sir.")

        elif "what's playing" in cmd or "current" in cmd:
            track = sp.current_playback()
            if track and track['item']:
                name = track['item']['name']
                artist = track['item']['artists'][0]['name']
                say(f"Currently playing {name} by {artist}.")
            else:
                say("Nothing is currently playing, sir.")

        else:
            say("Please specify what to do on Spotify, sir — play, pause, next, or volume.")

    except Exception as e:
        say("There was a problem controlling Spotify.")
        print("Spotify error:", e)




# ------------------ Core query handler (with semantic recall) ------------------
def handle_query(query):
    global driver
    global last_command
    if query == last_command:
        return
    last_command = query

    if not query:
        return
    user_text = query.strip()
    add_memory_entry("voice", user_text)
    append_transcript(user_text, action=None)

    q = user_text.lower().strip()
    # Basic commands
    if any(k in q for k in ["exit", "quit", "shutdown", "stop"]):
        say("Goodbye, sir.", block=True)
        persist_memory()
        os._exit(0)

    if q.startswith("teach me ") or q.startswith("teach "):
        words = q.replace("teach me", "").replace("teach", "").strip()
        lang = words.split()[0] if words else None
        topic = " ".join(words.split()[1:]).strip() or None
        if lang:
            teach_language(lang.capitalize(), topic)
        else:
            say("Which language do you want to learn, sir?")
        return

    if any(x in q for x in ["write", "create", "make"]) and any(y in q for y in ["code", "project", "program"]):
        # Detect language keywords
        lang = None
        for keyword in ["python", "html", "javascript", "js", "web", "website"]:
            if keyword in q:
                lang = keyword
                break
        # Extract the idea (remove filler words)
        desc = q
        for s in ["write", "create", "make", "project", "program", "code", "in python", "in html", "in javascript",
                  "in js"]:
            desc = desc.replace(s, "")
        desc = desc.strip() or "example"
        language = lang or "python"
        write_project_code(language, desc)
        return

    if driver:
        say("Browser is active. You can say 'scroll up', 'scroll down', or 'skip ad', sir.")
        while True:
            cmd = listen_once(timeout=8, phrase_time_limit=4)
            if not cmd:
                break
            cmd = cmd.lower()
            if "skip" in cmd and "ad" in cmd:
                skip_ads(driver)
            elif "scroll" in cmd:
                if "up" in cmd:
                    scroll_page(driver, "up")
                elif "down" in cmd:
                    scroll_page(driver, "down")
                elif "top" in cmd:
                    scroll_page(driver, "top")
                elif "bottom" in cmd:
                    scroll_page(driver, "bottom")
            elif "exit" in cmd or "close" in cmd:
                say("Closing browser, sir.")
                driver.quit()
                break

    # ------------------ Browser Interaction Enhancements ------------------
    def skip_ads(driver):
        """Attempt to skip YouTube or site ads automatically."""
        try:
            say("Checking for ads, sir.")
            # For YouTube skip ad buttons
            ad_buttons = driver.find_elements(By.XPATH, "//button[contains(@class,'ytp-ad-skip-button')]")
            if ad_buttons:
                ad_buttons[0].click()
                say("Ad skipped successfully, sir.")
                return True

            # Generic skip buttons
            generic_skips = driver.find_elements(By.XPATH, "//button[contains(text(),'Skip') or contains(text(),'Close')]")
            for btn in generic_skips:
                try:
                    btn.click()
                    say("Skipped an ad or popup, sir.")
                    return True
                except Exception:
                    continue
            say("No ads to skip right now, sir.")
            return False
        except Exception as e:
            print("Skip ad error:", e)
            say("Unable to skip ad at the moment, sir.")
            return False

    def scroll_page(driver, direction="down"):
        """Scroll page using voice commands."""
        try:
            if direction == "down":
                driver.execute_script("window.scrollBy(0, window.innerHeight);")
                say("Scrolled down, sir.")
            elif direction == "up":
                driver.execute_script("window.scrollBy(0, -window.innerHeight);")
                say("Scrolled up, sir.")
            elif direction == "top":
                driver.execute_script("window.scrollTo(0, 0);")
                say("Scrolled to the top, sir.")
            elif direction == "bottom":
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                say("Scrolled to the bottom, sir.")
            else:
                say("Please say scroll up, scroll down, scroll top, or scroll bottom, sir.")
        except Exception as e:
            print("Scroll error:", e)
            say("I couldn’t scroll right now, sir.")

    if "screenshot" in q or "take screenshot" in q:
        try:
            img_path = PROJECTS_DIR / f"screenshot_{int(time.time())}.png"
            img = ImageGrab.grab()
            img.save(img_path)
            say(f"Screenshot saved to {img_path}", block=False)
            add_memory_entry("screenshot", str(img_path))
            append_transcript("screenshot", action=f"Saved {img_path}")
        except Exception as e:
            print("Screenshot error:", e)
            say("Failed to take screenshot, sir.")
        return

    if "spotify" in q:
        handle_spotify_command(q)
        return

    if any(k in q for k in ["wikipedia", "who is", "what is", "tell me about"]):
        topic = q
        for s in ["wikipedia", "tell me about", "who is", "what is", "search for"]:
            topic = topic.replace(s, "")
        topic = topic.strip()
        try:
            summary = wikipedia.summary(topic, sentences=2)
            say(summary)
            add_memory_entry("wiki", {"query": topic, "summary": summary})
            append_transcript(topic, assistant_text=summary)
        except Exception:
            say("I couldn't fetch it from Wikipedia, sir.")
        return

    # Semantic recall: retrieve up to 3 most similar past interactions and include short context
    similar = semantic_search(user_text, k=3)
    context_snippets = []
    for s in similar:
        # only include ones with substantial content
        content = s.get("content")
        if isinstance(content, dict):
            snippet = content.get("user") or content.get("assistant") or str(content)
        else:
            snippet = str(content)
        if snippet and len(snippet) > 20:
            context_snippets.append(snippet[:300])
    context_prompt = ""
    if context_snippets:
        context_prompt = "Previous related interactions:\n" + "\n---\n".join(context_snippets) + "\n\nUse this context to be consistent.\n"

    # detect language and instruct AI to reply same language when non-english
    lang = detect_language(user_text)
    ai_prompt = user_text
    if context_prompt:
        ai_prompt = context_prompt + "\nUser: " + user_text

    if lang and lang != "en":
        # instruct assistant to reply in the same language
        ai_prompt = f"Reply in {lang} to the following:\n{ai_prompt}"

    say("Thinking...", block=False)
    response = ai_chat(ai_prompt)
    if response:
        say(response)
        add_memory_entry("ai_response", {"prompt": user_text, "response": response})
        append_transcript(user_text, assistant_text=response)
    else:
        say("I couldn't get an answer right now, sir.")
        append_transcript(user_text, assistant_text="(no response)")

    # -----------------------------
    # DEVICE CONTROL VIA VOICE
    # -----------------------------
    if any(word in q for word in ["brightness", "bright", "light level"]):
        import re
        if "increase" in q or "up" in q or "raise" in q:
            change_brightness("up")
            current = sbc.get_brightness(display=0)[0]
            say(f"Brightness increased to {current} percent, sir.")
            return
        elif "decrease" in q or "down" in q or "reduce" in q or "lower" in q:
            change_brightness("down")
            current = sbc.get_brightness(display=0)[0]
            say(f"Brightness reduced to {current} percent, sir.")
            return
        else:
            match = re.search(r"brightness\s*(\d+)", q)
            if match:
                level = int(match.group(1))
                set_brightness(level)
                say(f"Brightness set to {level} percent, sir.")
                return
            say("Please tell me what brightness level to set, sir.")
            return

    if any(word in q for word in ["volume", "sound", "audio"]):
        import re
        if "increase" in q or "up" in q or "raise" in q:
            change_volume("up")
            say("Volume increased, sir.")
            return
        elif "decrease" in q or "down" in q or "reduce" in q or "lower" in q:
            change_volume("down")
            say("Volume decreased, sir.")
            return
        elif "mute" in q or "silent" in q:
            mute_volume()
            return
        else:
            match = re.search(r"(?:volume|sound|audio)\s*(\d+)", q)
            if match:
                level = int(match.group(1))
                set_volume(level)
                say(f"Volume set to {level} percent, sir.")
                return
            say("Please tell me what volume level to set, sir.")
            return


# ------------------ Continuous listener ------------------
def listen_continuous():
    say("Jarvis v3 online and listening.", block=False)
    while True:
        try:
            print("🎤 Listening...")
            query = listen_once(timeout=10, phrase_time_limit=8)
            if query:
                print("You said:", query)
                handle_query(query)
            else:
                continue
        except Exception as e:
            print("Listener error:", e)
            time.sleep(1)

# ------------------ Main ------------------
if __name__ == "__main__":
    say("Hello sir, Jarvis version 3 ready.")
    listener_thread = threading.Thread(target=listen_continuous, daemon=True)
    listener_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        persist_memory()
        say("Shutting down, sir.", block=True)
