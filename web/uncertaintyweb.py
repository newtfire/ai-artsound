#Imports - note: Pyscript and Pyodide throw errors but still maintain functionality
import json, math, asyncio, datetime, socket, platform
from pyscript import when, display, window
from pyscript.web import page
from pyodide.http import pyfetch
from js import Blob, URL, document
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
4

# Taken directly from uncertainty.monitor
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
    link.textContent = "Download This Log"
    htmlbreak = js.document.createElement("br")
    js.document.getElementById("output").appendChild(htmlbreak)
    js.document.getElementById("output").appendChild(p)
    js.document.getElementById("centContainer").appendChild(link)


@when("click", "#run") # retrieves values from webpage
async def handle_click():
    audio_ctx = window.AudioContext.new()
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


                logprob_data = data.get("logprobs") or []
                top_logprobs = {}
                if isinstance(logprob_data, list) and logprob_data:
                    entry = logprob_data[0]
                    top_logprobs[entry["token"]] = entry["logprob"]
                    for candidate in entry.get("top_logprobs", []):
                        top_logprobs[candidate["token"]] = candidate["logprob"]

                score = compute_uncertainty(top_logprobs)

                # Dramatic pause from uncertainty monitor (Adjust?)
                token_count += 1
                total_uncertainty += score
                if score > UNCERTAINTY_THRESHOLD:
                    hesitation_count += 1
                    pause = MAX_PAUSE_SECONDS * (score - UNCERTAINTY_THRESHOLD) / (1.0 - UNCERTAINTY_THRESHOLD)
                    await asyncio.sleep(pause)

                # Real-time appearing + color coding
                if token:
                    span = js.document.createElement("span")
                    span.textContent = token
                    span.dataset.score = str(round(score, 4))
                    span.className = "token"
                    span.dataset.possibilities = "\n".join(
    f"{possibles}  ({math.exp(probs)*100:.1f}%)"
    for possibles, probs in sorted(top_logprobs.items(), key=lambda x: -x[1])
)
                    if score < 0.5:
                        t = score / 0.5
                        r = int(100 + t * (255 - 100))
                        g = int(200 + t * (200 - 200))
                        b = int(255 + t * (60 - 255))

                        rgb_string = f"rgb({r}, {g}, {b})"
                        span.style.color = rgb_string

                    else:
                        t = (score - 0.5) / 0.5
                        r = int(255)
                        g = int(200 + t * (60 - 200))
                        b = int(60 + t * (40 - 60))

                        rgb_string = f"rgb({r}, {g}, {b})"
                        span.style.color = rgb_string
                    output.appendChild(span)
                    #FUNCTION FOR SOUND GO HERE!!!!!!

                if data.get("done"):
                    break

    except Exception as e:
        output.textContent = f"Error: {str(e)}"

    createLog(token_count, hesitation_count, total_uncertainty/token_count if token_count > 0 else 0.0)



# Sound?
PENTATONIC_INTERVALS = [0, 2, 4, 7, 9]   # semitones: C D E G A

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
    ADSR envelope as a numpy array.
      attack  : seconds to ramp from 0 → 1.0
      decay   : seconds to fall from 1.0 → sustain level
      sustain : volume level held during the body (0.0–1.0)
      release : seconds to fade from sustain → 0  (governed by top-two gap)
    """
    a = int(attack  * sample_rate)
    d = int(decay   * sample_rate)
    r = int(release * sample_rate)
    s = max(0, total_samples - a - d - r)

    envelope = np.concatenate([
        np.linspace(0.0,     1.0,     a),   # attack
        np.linspace(1.0,     sustain, d),   # decay
        np.full(s,           sustain),       # sustain
        np.linspace(sustain, 0.0,     r),   # release
    ])
    # Trim or pad to exact length
    if len(envelope) < total_samples:
        envelope = np.append(envelope, np.zeros(total_samples - len(envelope)))
    return envelope[:total_samples]










# ── Continuous Audio Engine ──────────────────────────────────────────────────
#
# A single audio stream runs continuously.
# Each token instantly updates pitch AND resets the amplitude envelope,
# so every word gets a fresh attack even if the pitch repeats.
# Punctuation triggers a release (fade to silence).
#
# Envelope per token:
#   Attack  : very short ramp up (avoids click)
#   Decay   : piano-style exponential fade over the token's lifetime
#   Release : smooth fade to silence on punctuation

import threading as _threading

PUNCTUATION      = set('.!?;:')
STREAM_CHUNK     = 256        # samples per callback (~6ms at 44100)
SAMPLE_RATE      = 44100
ATTACK_SECONDS   = 0.020      # longer attack — prevents speaker clicks
DECAY_RATE       = 12.0       # marimba: fast decay (try 8.0–20.0)
RELEASE_SECONDS  = 0.22       # fade-out length at punctuation

_current_freq    = 0.0        # 0 = silent
_volume          = 0.3        # master volume 0.0–1.0
_phase           = 0.0        # phase accumulator for fundamental
_phase2          = 0.0        # phase accumulator for overtone

# Envelope state
_env_amplitude   = 0.0        # current envelope level
_env_attack_left = 0          # samples remaining in attack
_releasing       = False
_release_samples_left = 0
_release_start_amp    = 0.0

_audio_lock      = _threading.Lock()

def _audio_callback(outdata, frames, time_info, status):
    global _phase, _phase2, _env_amplitude, _env_attack_left
    global _releasing, _release_samples_left, _release_start_amp, _current_freq

    with _audio_lock:
        freq          = _current_freq
        releasing     = _releasing
        rel_left      = _release_samples_left
        rel_start_amp = _release_start_amp
        atk_left      = _env_attack_left
        amp           = _env_amplitude

    chunk = np.zeros(frames, dtype=np.float32)

    if freq > 0 or releasing:
        amps = np.empty(frames, dtype=np.float32)
        for i in range(frames):
            if releasing:
                if rel_left > 0:
                    amps[i] = rel_start_amp * (rel_left / (RELEASE_SECONDS * SAMPLE_RATE))
                    rel_left -= 1
                else:
                    amps[i] = 0.0
                    freq = 0.0
            elif atk_left > 0:
                progress = 1.0 - (atk_left / (ATTACK_SECONDS * SAMPLE_RATE))
                amps[i]  = progress
                amp      = progress
                atk_left -= 1
            else:
                # Marimba: fast exponential decay
                amp     *= (1.0 - DECAY_RATE / SAMPLE_RATE)
                amp      = max(amp, 0.0)
                amps[i]  = amp

        if freq > 0:
            t = np.arange(frames) / SAMPLE_RATE
            # Fundamental sine
            phases1  = _phase  + 2 * np.pi * freq       * t
            # Overtone: 2 octaves up (4x freq), quieter and decays faster
            phases2  = _phase2 + 2 * np.pi * freq * 4.0 * t
            sine1    = np.sin(phases1).astype(np.float32)
            sine2    = np.sin(phases2).astype(np.float32) * 0.25 * (amps ** 1.5)
            chunk    = (sine1 * amps + sine2) * _volume
            _phase   = phases1[-1] % (2 * np.pi)
            _phase2  = phases2[-1] % (2 * np.pi)

        with _audio_lock:
            _env_amplitude        = amp
            _env_attack_left      = atk_left
            _releasing            = releasing and rel_left > 0
            _release_samples_left = rel_left
            if not _releasing:
                _current_freq     = freq

    outdata[:, 0] = chunk
    outdata[:, 1] = chunk

# NEED ROUGH EQUIVALENT
_stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=2,
    dtype='float32',
    blocksize=STREAM_CHUNK,
    callback=_audio_callback,
)
_stream.start()

AUDIO_AVAILABLE = True

def set_pitch(freq: float):
    """Switch to a new pitch and reset the envelope for a fresh attack."""
    global _current_freq, _env_amplitude, _env_attack_left, _releasing, _phase, _phase2
    with _audio_lock:
        _current_freq    = freq
        _env_amplitude   = 0.0
        _env_attack_left = int(ATTACK_SECONDS * SAMPLE_RATE)
        _releasing       = False
        _phase           = 0.0   # reset phase on new note for clean attack
        _phase2          = 0.0

def trigger_release():
    """Fade out over RELEASE_SECONDS (called on punctuation tokens)."""
    global _releasing, _release_samples_left, _release_start_amp
    with _audio_lock:
        if _current_freq > 0 or _env_amplitude > 0:
            _releasing            = True
            _release_samples_left = int(RELEASE_SECONDS * SAMPLE_RATE)
            _release_start_amp    = _env_amplitude

def play_tone(freq: float, **kwargs):
    """Update pitch and rearticulate (reset envelope) for each token."""
    set_pitch(freq)

# ── Exhibit Trigger Hook ──────────────────────────────────────────────────────

def on_uncertainty_event(token: str, score: float, candidates: dict):
    """
    Fires on every token.
    - Punctuation (. ! ? ; :) triggers a release (fade to silence).
    - Tokens starting with a space mark a true word boundary — also release
      briefly before the new note, so compound sub-word tokens (e.g. "stand"
      + "ing") flow into each other with sustain pedal effect, only cutting
      off cleanly at the next real word boundary.
    - All other tokens play their note immediately after the previous one.
    Add Arduino / OSC / MCP calls here as the exhibit develops.

    Args:
        token      : the word the model just chose
        score      : 0.0 (confident) -> 1.0 (maximally uncertain)
        candidates : {token_string: log_prob} for the top-k alternatives
    """
    # Punctuation: full release to silence
    if any(p in token for p in PUNCTUATION):
        trigger_release()
        return

    if not token.strip():
        return

    # A leading space means this is a new word (not a sub-word continuation).
    # Trigger a very brief release so the previous word cuts off cleanly,
    # then immediately start the new note — like lifting and re-pressing a
    # piano key with the sustain pedal still down.
    if token.startswith(' '):
        trigger_release()
        time.sleep(ATTACK_SECONDS * 2)   # tiny gap between words

    midi_note = score_to_note(score)
    freq      = midi_to_freq(midi_note)
    play_tone(freq)




