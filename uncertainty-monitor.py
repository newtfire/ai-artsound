# Uncertainty Monitor for Ollama LLMs 
# This script is ultimately for monitoring, visualizing, and sonifying
# a language model's uncertainty while it is calculating how to generate text.
# * Designed to work with Ollama + Qwen
# * Computes an uncertainty score at each stage in response genration to a prompt.
# * Designed to pause variably to produce dramatic effect during high uncertainty.

# Requirements:  pip install requests
# Run with: 
#     python uncertainty-monitor.py
#     python uncertainty-monitor.py "Your custom prompt here"

# This code was prepared and adapted with assistance from Claude Sonnet 4.6 Extended.


import requests, json, math, time, sys
# Enable ANSI color support on Windows
import os
if os.name == "nt":
    os.system("")   # triggers Windows 10+ ANSI mode in CMD/PowerShell
import platform
import socket
import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ── Log Sessions ────────────────────────────────────────────────────────────
class SessionLogger:
    """
    Logs each session to a separate XML file in LOG_DIR.
    Each session file has a <session> root with one <log> child per run.
    """
    def __init__(self, model: dict):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.model = model
        self.session_start = datetime.datetime.now()
        timestamp_str = self.session_start.isoformat(timespec="seconds")
        # This should give us a timestamp like this: "2026-04-18T14:20:00"
        model_slug = model["id"].replace("/", "-").replace(":", "-")
        filename = f"{timestamp_str}_{model_slug}.xml"
        self.filepath = os.path.join(LOG_DIR, filename)
        # Build the root element and write the initial file
        self.root = ET.Element("session")
        # Operating system information
        self.root.set("hostname", socket.gethostname())
        self.root.set("os", platform.system())  # e.g. "Darwin", "Windows", "Linux"
        self.root.set("os_version", platform.version())  # detailed OS version string
        self.root.set("os_release", platform.release())  # e.g. "14.4.1" on macOS
        self.root.set("machine", platform.machine())  # e.g. "arm64", "x86_64"
        self.root.set("processor", platform.processor())  # e.g. "arm", "Intel64"
        self.root.set("python", platform.python_version())  # e.g. "3.13.0"
        # Timestamp and model info
        self.root.set("timestamp", self.session_start.isoformat(timespec="seconds"))
        self.root.set("model_id", model["id"])
        self.root.set("model_label", model["label"])
        self._write()
        print(f"{DIM}Logging to: {self.filepath}{RESET}")

    def log_response(self, prompt: str, token_log: list, settings: dict):
        """Call this after each completed generation."""
        response_text = "".join(tok for tok, _, _ in token_log)
        scores = [sc for _, sc, _ in token_log if sc > 0]
        avg_uncertainty = sum(scores) / len(scores) if scores else 0
        hesitations = sum(1 for sc in scores if sc > UNCERTAINTY_THRESHOLD)

        # Build <log> element
        log_el = ET.SubElement(self.root, "log")
        log_el.set("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
        log_el.set("temperature", str(settings.get("temperature", "")))
        log_el.set("num_predict", str(settings.get("num_predict", "")))
        log_el.set("repeat_penalty", str(settings.get("repeat_penalty", "")))

        # <prompt> child
        prompt_el = ET.SubElement(log_el, "prompt")
        prompt_el.text = prompt

        # <response> child
        response_el = ET.SubElement(log_el, "response")
        response_el.set("token_count", str(len(token_log)))
        response_el.set("hesitation_count", str(hesitations))
        response_el.set("mean_uncertainty", str(round(avg_uncertainty, 4)))
        response_el.text = response_text

        self._write()

    def _write(self):
        """Serialize the current tree to a pretty-printed XML file."""
        raw = ET.tostring(self.root, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="    ")
        # toprettyxml adds its own declaration; keep it clean
        lines = pretty.split("\n")
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

# ── Model Registry ────────────────────────────────────────────────────────────
# Add or remove models from the python dictionary here as you experiment.
# "label" is what visitors see; "id" is the Ollama model identifier.
# "temperature" gives a default temp and "num_predict" sets the number of tokens to output.
# repeat_penalty tries to prevent a model from looping/repeating chunks of its response.

MODELS = [
    {
        "label": "Qwen 2.5 (1.5B) — general language, web-trained",
        "id":    "qwen2.5:1.5b",
        "temperature": 0.8,
        "num_predict": 200,
    },
    {
        "label": "OLMo 2 (7B) — Allen AI, open science, transparent training",
        "id":    "olmo2:7b",
        "temperature": 0.8,
        "num_predict": 200,
    },
    {
        "label": "OLMo 2 (13B) — Allen AI, open science, larger model",
        "id":    "olmo2:13b",
        "temperature": 0.8,
        "num_predict": 200,
    },
    {
        "label": "Pleias 350m — copyright-free, literary register",
        "id":    "pleias-350m:latest",
        "temperature": 0.2,    # PleIAs recommends low temperature
        "repeat_penalty": 1.2,
        "num_predict": 80,     # base model doesn't self-terminate
    },
    {
        "label": "Pleias Nano (1.2B) — copyright-free, RAG-specialized",
        "id":    "pleias-nano:latest",
        "temperature": 0.2,
        "repeat_penalty": 1.2,
        "num_predict": 60,     # very tight cap — this one runs forever
    },
    {
        "label": "Llama 3.2 (3B) — Meta, general language",
        "id":    "llama3.2:3b",
    },
    {
        "label": "Llama 3.2 (1B) — Meta, general language, very small",
        "id":    "llama3.2:1b",
    },
]

# Models that require manual setup (can't be pulled via Ollama)
MANUAL_SETUP_MODELS = ["pleias-350m:latest", "pleias-nano:latest"]

DEFAULT_MODEL_INDEX = 0   # which model to use if visitor skips selection

# ── Check if you have the  models  ────────────────────────────────────────────────────────────
# ebb: This next function checks if you have a model locally, and if you don't, it pulls it in
# via ollama if it's possible to do that. But if it's a
def ensure_model_available(model_id: str) -> bool:
    """
    Check if a model is available locally. If not, offer to pull it.
    Returns True if the model is ready, False if unavailable.
    """
    # Ask Ollama what models are installed
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        installed = [m["name"] for m in response.json().get("models", [])]
    except requests.exceptions.ConnectionError:
        print(f"\n{COLOR_ERROR}Cannot reach Ollama at {OLLAMA_URL}{RESET}")
        print("Start it with:  ollama serve")
        sys.exit(1)

    if model_id in installed:
        return True
    else:
        if model_id in MANUAL_SETUP_MODELS:
            print(f"\n{COLOR_UNCERTAIN}'{model_id}' requires manual setup.{RESET}")
            print(f"{DIM}See README: Installing and Configuring with PleIAs Models{RESET}")
            return False

    # Model not found — offer to pull it
    print(f"\n{COLOR_UNCERTAIN}Model '{model_id}' is not installed.{RESET}")
    answer = input(f"{COLOR_LABEL}Pull it now? This may take several minutes. [y/n]: {RESET}").strip().lower()
    if answer != "y":
        return False

    print(f"{DIM}Pulling {model_id}...{RESET}")
    try:
        pull_response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_id},
            stream=True
        )
        pull_response.raise_for_status()
        for line in pull_response.iter_lines():
            if line:
                status = json.loads(line)
                msg = status.get("status", "")
                completed = status.get("completed", 0)
                total = status.get("total", 0)
                if total:
                    pct = completed / total * 100
                    print(f"\r{DIM}{msg}: {pct:.1f}%{RESET}", end="", flush=True)
                else:
                    print(f"\r{DIM}{msg}{RESET}", end="", flush=True)
        print(f"\n{COLOR_CONFIDENT}Done.{RESET}\n")
        return True
    except Exception as e:
        print(f"\n{COLOR_ERROR}Pull failed: {e}{RESET}")
        return False

# ── "Big Picture" Parameters  ──────────────────────────────────────────────────────────

OLLAMA_URL            = "http://localhost:11434/api/generate"
TOP_K_CANDIDATES      = 5
UNCERTAINTY_THRESHOLD = 0.55
MAX_PAUSE_SECONDS     = 2.0
PROMPT = (
    "Describe the feeling of standing at the edge of a vast, dark ocean "
    "at night, in two or three sentences."
)


# ── ANSI (Shell) Display / RGB Color Variables ─────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


# These color settings are for running in the shell.
# They set true-color ANSI foreground for programming colors in
# the terminal.  We'll have to replace this function's body
# when ready to send color to external hardware."""
def rgb(r: int, g: int, b: int) -> str:
    """These color settings are for running in the shell.
    Replace the return value with hardware output when ready."""
    return f"\033[38;2;{r};{g};{b}m"

    # FOR EXHIBIT: Add a line to send RGB values to external hardware.
    # send_to_hardware(r, g, b)   # your Arduino/MCP call
    # return f"\033[38;2;{r};{g};{b}m"   # still colors the terminal too.


# ─── RGB Color Palette! (Edit/adjust) ───────────────────────

COLOR_CONFIDENT = rgb(100, 200, 255)  # cool blue  — low uncertainty
COLOR_UNCERTAIN = rgb(255, 200, 60)  # warm amber — mid uncertainty
COLOR_HESITATING = rgb(255, 60, 40)  # hot red    — high uncertainty
COLOR_LABEL = rgb(180, 180, 180)  # soft grey  — UI chrome
COLOR_HIGHLIGHT = rgb(220, 120, 255)  # violet     — hesitation marker
COLOR_ERROR = rgb(255, 60, 40)  # same red as COLOR_HESITATING, or adjust to taste
COLOR_HEADER = rgb(100, 220, 200)  # teal — section titles(?) or something else :-)


# Function to select a model! 
def select_model() -> dict:
    """Prompt the user to choose a model. Returns the selected model dict."""
    print(f"\n{BOLD}{COLOR_HEADER}Available models:{RESET}\n")
    for i, m in enumerate(MODELS):
        print(f"  {COLOR_LABEL}{i + 1}.{RESET} {m['label']}")
        print(f"     {DIM}{m['id']}{RESET}")
    print()

    while True:
        raw = input(
            f"{COLOR_LABEL}Choose a model [1–{len(MODELS)}, "
            f"or Enter for default ({DEFAULT_MODEL_INDEX + 1})]: {RESET}"
        ).strip()

        if not raw:
            chosen = MODELS[DEFAULT_MODEL_INDEX]
            if ensure_model_available(chosen["id"]):
                print(f"{DIM}Using default: {chosen['label']}{RESET}")
                return chosen
            else:
                print(f"{COLOR_ERROR}Default model unavailable. Please select another.{RESET}")
                continue
        if raw.isdigit() and 1 <= int(raw) <= len(MODELS):
            chosen = MODELS[int(raw) - 1]
            if ensure_model_available(chosen["id"]):
                return chosen
            else:
                print(f"{COLOR_ERROR}Skipping — please choose another model.{RESET}")
                continue  # loop back to prompt again
        print(f"{COLOR_ERROR}Please enter a number between 1 and {len(MODELS)}.{RESET}")

# ebb: Function to let the user adjust the temperature and prompt size
def adjust_settings(model: dict) -> dict:
    """
    Give the user the options to override temperature and token limit
    for the current model. Returns a copy of the model dict with
    updated settings, leaving the registry defaults untouched.
    """
    import copy
    m = copy.deepcopy(model)

    print(f"\n{BOLD}{COLOR_HEADER}Model settings:{RESET}")
    print(f"  {COLOR_LABEL}Temperature :{RESET} {m['temperature']}  "
          f"{DIM}(lower = more predictable, higher = more creative){RESET}")
    print(f"  {COLOR_LABEL}Max tokens  :{RESET} {m['num_predict']}  "
          f"{DIM}(hard cap on response length){RESET}")
    print(f"{DIM}Press Enter to keep current values.{RESET}\n")

    raw = input(f"{COLOR_LABEL}Temperature [Enter to keep {m['temperature']}]: {RESET}").strip()
    if raw:
        try:
            m["temperature"] = float(raw)
        except ValueError:
            print(f"{COLOR_ERROR}Invalid value, keeping {m['temperature']}{RESET}")

    raw = input(f"{COLOR_LABEL}Max tokens  [Enter to keep {m['num_predict']}]: {RESET}").strip()
    if raw:
        try:
            m["num_predict"] = int(raw)
        except ValueError:
            print(f"{COLOR_ERROR}Invalid value, keeping {m['num_predict']}{RESET}")

    return m

# ── Uncertainty Calculation ───────────────────────────────────────────────────
def compute_uncertainty(top_logprobs: dict) -> float:
    """Shannon entropy of the top-k distribution, normalized to [0, 1]."""
    if not top_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in top_logprobs.values()]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0



def uncertainty_color(score: float) -> str:
    """
    This converts uncertainty data into a color gradient 
    from cool blue → amber → hot red as uncertainty increases.
    """

    if score < 0.5:
        # blue → amber
        t = score / 0.5
        r = int(100 + t * (255 - 100))
        g = int(200 + t * (200 - 200))
        b = int(255 + t * ( 60 - 255))

    else:
        # amber → red
        t = (score - 0.5) / 0.5
        r = int(255)
        g = int(200 + t * ( 60 - 200))
        b = int( 60 + t * ( 40 -  60))
    return rgb(r, g, b)

def uncertainty_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{uncertainty_color(score)}{bar}{RESET}"

def format_candidates(top_logprobs, chosen):
    # This formats the top candidate tokens and their probabilities for display.
    # So we can see not just the uncertainty score, but also 
    # which tokens are being considered and how close they are to being chosen.
    # top_logprobs is a dictionary of token → logprob for the top candidates like
    # this: {"dark": -0.48, "vast": -1.55, "cold": -2.21, "deep": -2.81}
    # chosen is the token that was actually selected.

    parts = []
    # This for loop sorts the candidates by logprob (highest first) and takes the top 4. 
    # (Adjust to take more of them!) 
    # The function formats each one with a marker to show which was chosen, and the probability as a percentage.
    for token, lp in sorted(top_logprobs.items(), key=lambda x: -x[1])[:4]:
        # The marker "→" indicates which token was chosen, so we can see how close the others were.
        marker = "→" if token == chosen else " "
        parts.append(f"{marker}{repr(token):15} {math.exp(lp)*100:5.1f}%")
        # {math.exp(lp)*100:5.1f}%") expresses the log probabiliy (0 - 1) in a percentage (.62 to 62.0%) with one decimal place, and a width of 5 characters for alignment.
    return "  ".join(parts)

# ── TBD: Exhibit Trigger MCP Hook ──────────────────────────────────────────────────────

def on_uncertainty_event(token: str, score: float, candidates: dict):
    """
    *** DEVELOP OUR EXHIBIT TRIGGER HERE ***

    This fires on every token. Replace the pass below with:
      - An MCP tool call  →  trigger lights / sounds
      - serial.write()    →  Arduino
      - OSC message       →  sound system

    Args:
        token      : the token the model just chose
        score      : 0.0 (confident) → 1.0 (maximally uncertain)
        candidates : {token_string: log_prob} for the top-k alternatives
    """
    pass  # ← your trigger code here


# ── Main Streaming Function! ─────────────────────────────────────────────────────────────

def stream_with_uncertainty(prompt: str, model: dict, logger: SessionLogger = None):
    print(f"\n{BOLD}{COLOR_LABEL}━━━ Uncertainty Monitor ━━━{RESET}")
    print(f"{DIM}Model: {model['label']}{RESET}")
    print(f"{DIM}Threshold: {UNCERTAINTY_THRESHOLD}{RESET}\n")
    print(f"{BOLD}Prompt:{RESET} {prompt}\n")
    print(f"{BOLD}{'─'*80}{RESET}\n")
    print(f"{COLOR_CONFIDENT}blue = confident{RESET}  "
      f"{COLOR_UNCERTAIN}amber = uncertain{RESET}  "
      f"{COLOR_HESITATING}red = hesitating{RESET}\n")

    payload = {
        "model": model['id'], "prompt": prompt, "stream": True,
        "logprobs": True,
        "top_logprobs": TOP_K_CANDIDATES,   # ← top level, not inside options
        "options": {
            "temperature": model.get("temperature", 0.8),
            "repeat_penalty": model.get("repeat_penalty", 1.1),
            # ebb: repeat penalty should help prevent looping in responses.
            "top_k": TOP_K_CANDIDATES,
            "num_predict": model.get("num_predict", 200),
                    },
}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    response.raise_for_status()

    token_log = []

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        chunk = json.loads(raw_line)
        # print(f"\nDEBUG: {chunk}")   # for debugging the logprobs output format from Ollama. Remove or comment out when ready.
        token = chunk.get("response", "")

        # Parse logprobs (Ollama format varies slightly by version)
        logprob_data = chunk.get("logprobs") or []
        top_logprobs = {}
        if isinstance(logprob_data, list) and logprob_data:
            entry = logprob_data[0]
            for candidate in entry.get("top_logprobs", []):
                top_logprobs[candidate["token"]] = candidate["logprob"]

        score = compute_uncertainty(top_logprobs)
        token_log.append((token, score, top_logprobs))

        # Trigger hook for exhibit response here:
        on_uncertainty_event(token, score, top_logprobs)


        # Pacing: Dramatic pause: scales with uncertainty above threshold 
        if score > UNCERTAINTY_THRESHOLD:
            pause = MAX_PAUSE_SECONDS * (score - UNCERTAINTY_THRESHOLD) / (1.0 - UNCERTAINTY_THRESHOLD)
            time.sleep(pause)
        # Print token colored by uncertainty
        sys.stdout.write(f"{uncertainty_color(score)}{token}{RESET}")
        sys.stdout.flush()

        if chunk.get("done"):
            break

# Post-generation report ────────────────────────────────────────────────
    print(f"\n\n{BOLD}{'─'*80}{RESET}")
    print(f"\n{BOLD}{COLOR_LABEL}Token-by-token log:{RESET}\n")
    print(f"{'TOKEN':20} {'SCORE':8} {'BAR':22} TOP CANDIDATES")
    print(f"{'─'*20} {'─'*8} {'─'*22} {'─'*40}")
    
    hesitations = 0
    for tok, sc, cands in token_log:
        if not tok.strip(): continue
        flag = f"{COLOR_HIGHLIGHT} ◀ HESITATION{RESET}" if sc > UNCERTAINTY_THRESHOLD else ""
        cand_str = format_candidates(cands, tok) if cands else "(no logprob data)"
        print(f"{repr(tok):20} {sc:8.3f}  {uncertainty_bar(sc)}  {cand_str}{flag}")
        if sc > UNCERTAINTY_THRESHOLD: hesitations += 1

    avg = sum(s for _, s, _ in token_log) / len(token_log) if token_log else 0
    print(f"\n{BOLD}Summary:{RESET}  {len(token_log)} tokens  |  "
          f"{hesitations} hesitation events  |  mean uncertainty {avg:.3f}")
    print(f"\n{DIM}Tune: UNCERTAINTY_THRESHOLD, MAX_PAUSE_SECONDS, PROMPT at top of file.")
    print(f"Wire up: add trigger code to on_uncertainty_event().{RESET}\n")

# ── Logging ──────────────────────────────────────────────────────────────
    if logger:
        settings = {
            "temperature":    model.get("temperature", 0.8),
            "num_predict":    model.get("num_predict", 200),
            "repeat_penalty": model.get("repeat_penalty", 1.1),
        }
        logger.log_response(prompt, token_log, settings)

# ── User Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # One-shot mode: model selection not supported, uses default
        prompt = " ".join(sys.argv[1:])
        try:
            stream_with_uncertainty(prompt, MODELS[DEFAULT_MODEL_INDEX])
        except requests.exceptions.ConnectionError:
            print(f"\n{COLOR_ERROR}Cannot reach Ollama at {OLLAMA_URL}{RESET}")
            print("Start it with:  ollama serve")
            sys.exit(1)
    else:
        print(f"\n{BOLD}{COLOR_HEADER}━━━ Uncertainty Monitor — Interactive Mode ━━━{RESET}")
        print(f"{DIM}Press Enter to use defaults. Type 'quit' to exit.{RESET}")

        current_model = select_model()   # ← select once at startup
        logger = SessionLogger(current_model) # initiate the log!

        while True:
            try:
                user_input = input(f"\n{COLOR_LABEL}Enter a prompt "
                                   f"(or 'm' to switch model, 's' to adjust settings): {RESET}").strip()

                if user_input.lower() in ("quit", "exit", "q"):
                    print(f"\n{DIM}Goodbye.{RESET}\n")
                    break
                if user_input.lower() == "m":
                    current_model = select_model()
                    logger = SessionLogger(current_model)  # Start a new log session file for new model
                    continue
                if user_input.lower() == "s":
                    current_model = adjust_settings(current_model)
                    continue
                prompt = user_input if user_input else PROMPT
                if not user_input:
                    print(f"{DIM}Using default prompt.{RESET}")
                stream_with_uncertainty(prompt, current_model, logger)
                print()

            except requests.exceptions.ConnectionError:
                print(f"\n{COLOR_ERROR}Cannot reach Ollama at {OLLAMA_URL}{RESET}")
                print("Start it with:  ollama serve\n")
            except KeyboardInterrupt:
                print(f"\n\n{DIM}Interrupted.{RESET}\n")
                break