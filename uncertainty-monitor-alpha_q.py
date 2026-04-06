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
# from datetime import datetime, timedelta
# from dateutil import tz

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL              = "qwen2.5:1.5b"
OLLAMA_URL         = "http://localhost:11434/api/generate"
TOP_K_CANDIDATES   = 5
UNCERTAINTY_THRESHOLD = 0.55   # 0.0–1.0; tune by watching output
MAX_PAUSE_SECONDS  = 2.0       # pause length at maximum uncertainty
PROMPT = (
    "Describe the feeling of standing at the edge of a vast, dark ocean "
    "at night, in two or three sentences."
)

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

# ── ANSI (Shell) Display / RGB Color Variables ─────────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

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

COLOR_CONFIDENT  = rgb(100, 200, 255)   # cool blue  — low uncertainty
COLOR_UNCERTAIN  = rgb(255, 200,  60)   # warm amber — mid uncertainty  
COLOR_HESITATING = rgb(255,  60,  40)   # hot red    — high uncertainty
COLOR_LABEL      = rgb(180, 180, 180)   # soft grey  — UI chrome
COLOR_HIGHLIGHT  = rgb(220, 120, 255)   # violet     — hesitation marker
COLOR_ERROR = rgb(255, 60, 40)   # same red as COLOR_HESITATING, or adjust to taste
COLOR_HEADER = rgb(100, 220, 200)   # teal — section titles
# COLOR_HEADER is reserved for future use

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


# ── Main Stream Loop ─────────────────────────────────────────────────────────────

def stream_with_uncertainty(prompt: str):
    print(f"\n{BOLD}{COLOR_LABEL}━━━ Uncertainty Monitor ━━━{RESET}")
    print(f"{DIM}Model: {MODEL}  |  Threshold: {UNCERTAINTY_THRESHOLD}{RESET}\n")
    print(f"{BOLD}Prompt:{RESET} {prompt}\n")
    print(f"{BOLD}{'─'*80}{RESET}\n")
    print(f"{COLOR_CONFIDENT}blue = confident{RESET}  "
      f"{COLOR_UNCERTAIN}amber = uncertain{RESET}  "
      f"{COLOR_HESITATING}red = hesitating{RESET}\n")

    payload = {
        "model": MODEL, "prompt": prompt, "stream": True,
        "logprobs": True,
        "top_logprobs": TOP_K_CANDIDATES,   # ← top level, not inside options
        "options": {"temperature": 0.8,
                "top_k": TOP_K_CANDIDATES, },
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

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # One-shot mode: python uncertainty-monitor.py "your prompt here"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        try:
            stream_with_uncertainty(prompt)
        except requests.exceptions.ConnectionError:
            print(f"\n{COLOR_ERROR}Cannot reach Ollama at {OLLAMA_URL}{RESET}")
            print("Start it with:  ollama serve")
            sys.exit(1)

    # Interactive mode: just run python uncertainty-monitor.py
    else:
        print(f"\n{BOLD}{COLOR_HEADER}━━━ Uncertainty Monitor — Interactive Mode ━━━{RESET}")
        print(f"{DIM}Press Enter to use the default prompt. Type 'quit' to exit.{RESET}\n")
        while True:
            try:
                user_input = input(f"{COLOR_LABEL}Enter a prompt:{RESET} ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print(f"\n{DIM}Goodbye.{RESET}\n")
                    break
                prompt = user_input if user_input else PROMPT
                if not user_input:
                    print(f"{DIM}Using default prompt.{RESET}")
                stream_with_uncertainty(prompt)
                print()  # breathing room between runs
            except requests.exceptions.ConnectionError:
                print(f"\n{COLOR_ERROR}Cannot reach Ollama at {OLLAMA_URL}{RESET}")
                print("Start it with:  ollama serve\n")
            except KeyboardInterrupt:
                print(f"\n\n{DIM}Interrupted.{RESET}\n")
                break