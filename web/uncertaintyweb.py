#Imports - note: Pyscript and Pyodide throw errors but still maintain functionality
import js, json, math, asyncio, datetime, socket, platform
from pyscript import when, display, window
from pyscript.web import page
from pyodide.http import pyfetch
from js import Blob, URL, document
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


# ------------------------
# Tuning constants.
# These are passed into the JS engine on the first Run click.
# ---------------------------------------------------------------------------


PENTATONIC_INTERVALS = [0, 2, 4, 7, 9]  # semitones: C D E G A

ATTACK_SECONDS = 0.060  # ramp up (longer = softer attack, avoids clicks) (originally 0.03)
DECAY_RATE = 12.0  # marimba-style exponential decay speed (8–20)
RELEASE_SECONDS = 0.22  # punctuation fade-out length in seconds
_VOLUME = 0.3  # master volume 0.0–1.0

PUNCTUATION = set('.!?;:')

# -----------------------------
# Set up the pentatonic scale
# -----------------------------

def build_pentatonic_scale(start_midi=48, end_midi=93):
    """Returns MIDI note numbers for C major pentatonic from start to end."""
    notes = []
    octave = 0
    while True:
        for interval in PENTATONIC_INTERVALS:
            note = start_midi + octave * 12 + interval
            if note > end_midi:
                return notes
            notes.append(note)
        octave += 1


PENTATONIC_SCALE = build_pentatonic_scale()


def midi_to_freq(midi_note: int) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def score_to_note(score: float) -> int:
    """Map uncertainty score 0.0–1.0 to a MIDI note in the pentatonic scale."""
    index = int(score * (len(PENTATONIC_SCALE) - 1))
    index = max(0, min(index, len(PENTATONIC_SCALE) - 1))
    return PENTATONIC_SCALE[index]


def top_two_gap(top_logprobs: dict) -> float:
    """
    Probability gap between the #1 and #2 candidates, normalized to 0.0–1.0.
      Near 1.0 = clear winner (model was decisive)
      Near 0.0 = near-tie   (model was torn between top two)
    This drives release time: decisive → short release, torn → long release.
    """
    if len(top_logprobs) < 2:
        return 1.0
    sorted_probs = sorted(
        [math.exp(lp) for lp in top_logprobs.values()], reverse=True
    )
    gap = sorted_probs[0] - sorted_probs[1]
    return min(gap / sorted_probs[0], 1.0)


def gap_to_release(gap: float, min_r=0.04, max_r=0.45) -> float:
    """
    Invert the gap so:
      gap ~1.0 (decisive) → short release (min_r seconds, clean note)
      gap ~0.0 (torn)     → long release  (max_r seconds, notes blur together)
    Adjust min_r and max_r to taste.
    """
    return min_r + (1.0 - gap) * (max_r - min_r)


def make_envelope(total_samples, attack=0.01, decay=0.05, sustain=0.7,
                  release=0.08, sample_rate=44100):
    """
    # 2026-06-17 ebb: THIS NEXT CODE IS UNUSED:
    ADSR envelope as a numpy array.
      attack  : seconds to ramp from 0 → 1.0
      decay   : seconds to fall from 1.0 → sustain level
      sustain : volume level held during the body (0.0–1.0)
      release : seconds to fade from sustain → 0  (governed by top-two gap)
    Claude Sonnet's NOTE: This function is retained from the original for reference / future use
    (e.g. offline rendering or visualization). In the web version the equivalent
    envelope is computed by the Web Audio API inside the JS engine above.
    """
    # numpy may not be available in Pyodide without explicit import
    try:
        import numpy as np
    except ImportError:
        return None
    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)
    s = max(0, total_samples - a - d - r)
    envelope = np.concatenate([
        np.linspace(0.0, 1.0, a),
        np.linspace(1.0, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0.0, r),
    ])
    if len(envelope) < total_samples:
        envelope = np.append(envelope, np.zeros(total_samples - len(envelope)))
    return envelope[:total_samples]
# ----- ebb: END UNUSED PORTION -----


# ---------------------------------------------------------------------------
# Web Audio output functions — replace set_pitch / trigger_release / play_tone
# from the terminal version.  The signatures and call sites in
# on_uncertainty_event() are kept identical so the logic layer is untouched.
# ---------------------------------------------------------------------------

def set_pitch(freq: float, release_seconds: float = RELEASE_SECONDS):
    """Switch to a new pitch and reset the envelope for a fresh attack."""
    window.playNote(freq, release_seconds)


def trigger_release(release_seconds: float = RELEASE_SECONDS):
    """Fade out over release_seconds (called on punctuation / word boundaries)."""
    window.triggerRelease(release_seconds)


def play_tone(freq: float, release_seconds: float = RELEASE_SECONDS, **kwargs):
    """Update pitch and rearticulate (reset envelope) for each token."""
    set_pitch(freq, release_seconds)


# ---------------------------------------------------------------------------
# on_uncertainty_event — logic layer preserved verbatim from original.
#
# The only change: time.sleep() → await asyncio.sleep() because the web
# version runs in an async context. The function is now a coroutine so it
# must be awaited at the call site (see handle_click below).
# ---------------------------------------------------------------------------

async def on_uncertainty_event(token: str, score: float, candidates: dict):
    """
    Fires on every token.
    - Punctuation (. ! ? ; :) triggers a release (fade to silence).
    - Tokens starting with a space mark a true word boundary — also release
      briefly before the new note, so compound sub-word tokens (e.g. "stand"
      + "ing") flow into each other with sustain pedal effect, only cutting
      off cleanly at the next real word boundary.
    - All other tokens play their note immediately after the previous one.

    Args:
        token      : the word the model just chose
        score      : 0.0 (confident) -> 1.0 (maximally uncertain)
        candidates : {token_string: log_prob} for the top-k alternatives
    """
    gap = top_two_gap(candidates)
    release_time = gap_to_release(gap)

    # Punctuation: full release to silence
    if any(p in token for p in PUNCTUATION):
        trigger_release(release_time)
        return

    if not token.strip():
        return

    # A leading space means this is a new word (not a sub-word continuation).
    # Trigger a very brief release so the previous word cuts off cleanly,
    # then immediately start the new note — like lifting and re-pressing a
    # piano key with the sustain pedal still down.
    if token.startswith(' '):
        trigger_release(release_time)
        await asyncio.sleep(ATTACK_SECONDS * 2)  # tiny gap between words

    midi_note = score_to_note(score)
    freq = midi_to_freq(midi_note)
    play_tone(freq, release_time)


# Taken directly from uncertainty-monitor.py that runs in the shell:
OLLAMA_URL            = "http://localhost:11434/api/generate" #must be localhost for docker
TOP_K_CANDIDATES      = 5
UNCERTAINTY_THRESHOLD = 0.55 #make adjustable?
MAX_PAUSE_SECONDS     = 2.0

#Shannon Entropy - see original Python script for details
def compute_uncertainty(top_logprobs: dict) -> float:
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

#log creation
def createLog(token_count, hesitation_count, mean_uncertainty):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    session = ET.Element("session", hostname=socket.gethostname(), os=platform.system(), os_version=platform.version(), os_release=platform.release(), machine=platform.machine(), processor=platform.processor(), python=platform.python_version(), model_id=str(page["#modelSelect"].value), model_label=page["#modelDesc"].textContent)
    log = ET.SubElement(session, "log", timestamp=timestamp, temperature=page["#temp"].value, num_predict=page["#maxTokens"].value, repeat_penalty="1.1")
    prompt = ET.SubElement(log, "prompt")
    prompt.text = str(page["#promptConfirm"].textContent)
    response = ET.SubElement(log, "response", token_count=str(token_count), hesitation_count=str(hesitation_count), mean_uncertainty=str(round(mean_uncertainty, 4)))
    response.text = ""
    for span in js.document.querySelectorAll("#output .token"):
        response.text += span.textContent
    string = ET.tostring(session, encoding="utf-8", method="xml").decode("utf-8")
    parse = minidom.parseString(string)
    finalstring = parse.toprettyxml(indent="  ")
    blob = js.Blob.new([finalstring], {"type": "application/xml"})
    url = js.URL.createObjectURL(blob)

    p = js.document.createElement("p")
    p.id = "centContainer"
    link = js.document.createElement("a")
    link.href = url
    link.id = "logLink"
    link.download = f"log_{timestamp}.xml"
    link.textContent = "Download this log as XML"
    htmlbreak = js.document.createElement("br")
    js.document.getElementById("output").appendChild(htmlbreak)
    js.document.getElementById("output").appendChild(p)
    js.document.getElementById("centContainer").appendChild(link)


@when("click", "#run") # retrieves values from webpage
async def handle_click():
    # initAudio() MUST be called from a user-gesture handler (this click) to
    # satisfy the browser's autoplay policy.  We pass in all tuning constants
    # from Python so the JS engine stays in sync with the values above.
    window.initAudio(ATTACK_SECONDS, DECAY_RATE, RELEASE_SECONDS, _VOLUME)
    model = str(page["#modelSelect"].value)
    prompt   = str(page["#promptConfirm"].textContent) #uses confirm <p> so default prompt is chosen w/o input
    temp     = float(page["#temp"].value)
    tokens   = int(page["#maxTokens"].value)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "logprobs": True,
        "top_logprobs": TOP_K_CANDIDATES,
        "options": {
            "temperature": temp,
            "repeat_penalty": 1.1, #make adjustable?
            "top_k": TOP_K_CANDIDATES, #make adjustable?
            "num_predict": tokens,
        },
    }

    output = js.document.getElementById("output")
    outTable = js.document.getElementById("outTable")
    fromLog = document.querySelectorAll('.fromLog')
    for element in fromLog:
        element.remove()

    print(fromLog)
    output.innerHTML = ""

    #logging - could implement in output?
    hesitation_count = 0
    total_uncertainty = 0.0
    token_count = 0

# original script incompatible with web format - adapted to use PyScript and async functions
    try:
        response = await pyfetch(
            OLLAMA_URL,
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload)
        )

        reader = response.js_response.body.getReader()
        decoder = js.TextDecoder.new("utf-8")
        buffer = ""

        while True:
            chunk = await reader.read()
            if chunk.done:
                break

            buffer += decoder.decode(chunk.value)

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                token = data.get("response", "")

                addHesitation = 0
                logprob_data = data.get("logprobs") or []
                top_logprobs = {}
                if isinstance(logprob_data, list) and logprob_data:
                    entry = logprob_data[0]
                    top_logprobs[entry["token"]] = entry["logprob"]
                    for candidate in entry.get("top_logprobs", []):
                        top_logprobs[candidate["token"]] = candidate["logprob"]

                score = compute_uncertainty(top_logprobs)

                # Dramatic pause from uncertainty monitor (Adjust?)
                total_uncertainty += score
                if score > UNCERTAINTY_THRESHOLD:
                    hesitation_count += 1
                    addHesitation = 1
                    pause = MAX_PAUSE_SECONDS * (score - UNCERTAINTY_THRESHOLD) / (1.0 - UNCERTAINTY_THRESHOLD)
                    await asyncio.sleep(pause)

                # Real-time appearing + color coding
                if token:
                    token_count += 1
                    numbering = js.document.createElement("td")
                    numbering.textContent = (str(token_count))
                    span = js.document.createElement("span")
                    tableOptions = js.document.createElement("td")
                    tableBar = js.document.createElement("td")
                    tableRow = js.document.createElement("tr")
                    tableRow.className = "fromLog"
                    span.textContent = token
                    tableToken = js.document.createElement("td")
                    tableToken.textContent = token
                    span.dataset.score = str(round(score, 4))
                    tableScore = js.document.createElement("td")
                    tableScore.textContent= str(round(score, 4))
                    span.className = "token"
                    span.dataset.possibilities = "\n".join(
    f"{possibles}  ({math.exp(probs)*100:.1f}%)"
    for possibles, probs in sorted(top_logprobs.items(), key=lambda x: -x[1])
)
                    tableOptions.textContent = "\t".join(
                        f"{possibles}  ({math.exp(probs) * 100:.1f}%)"
                        for possibles, probs in sorted(top_logprobs.items(), key=lambda x: -x[1])
                    )
                    if score < 0.5:
                        t = score / 0.5
                        r = int(100 + t * (255 - 100))
                        g = int(200 + t * (200 - 200))
                        b = int(255 + t * (60 - 255))

                        rgb_string = f"rgb({r}, {g}, {b})"
                        span.style.color = rgb_string
                        tableBar.style.color = rgb_string

                    else:
                        t = (score - 0.5) / 0.5
                        r = int(255)
                        g = int(200 + t * (60 - 200))
                        b = int(60 + t * (40 - 60))

                        rgb_string = f"rgb({r}, {g}, {b})"
                        span.style.color = rgb_string
                        tableBar.style.color = rgb_string


                    tableRow.id = str(token_count)
                    outHesitation = js.document.createElement('td')
                    output.appendChild(span)
                    outTable.appendChild(tableRow)
                    tableRow.appendChild(numbering)
                    tableRow.appendChild(tableToken)
                    tableRow.appendChild(tableScore)
                    for _ in range(math.trunc(score*10)):
                        tableBar.textContent += "▓"
                    for _ in range(10-math.trunc(score*10)):
                        tableBar.textContent += "░"
                    tableRow.appendChild(tableBar)
                    tableRow.appendChild(tableOptions)
                    if addHesitation == 1:
                        outHesitation.textContent = "< Hesitation Event"
                        outHesitation.style.color = "purple"
                        outHesitation.className = "hesitation"
                        tableRow.appendChild(outHesitation)
                    else:
                        outHesitation.textContent = ""
                        tableRow.appendChild(outHesitation)

                    await on_uncertainty_event(token, score, top_logprobs)

                if data.get("done"):
                    break

    except Exception as e:
        output.textContent = f"Error: {str(e)}"

    createLog(token_count, hesitation_count, total_uncertainty/token_count if token_count > 0 else 0.0)
    svgContainer = js.document.createElement("a")
    svgContainer.id = "svgContainer"
    svgLink = js.document.createElement("p")
    svgLink.textContent = "Download this log as SVG table"
    svgLink.className = "centContainer"
    js.document.getElementById("output").append(svgContainer)
    js.document.getElementById("svgContainer").append(svgLink)
