#Imports - note: Pyscript and Pyodide throw errors but still maintain functionality
import json, math, asyncio, datetime, socket, platform
from pyscript import when, display
from pyscript.web import page
from pyodide.http import pyfetch
import js
from js import Blob, URL, document
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

# Taken directly from uncertainty.monitor
OLLAMA_URL            = "http://localhost:11434/api/generate" #must be localhost for docker
TOP_K_CANDIDATES      = 5
UNCERTAINTY_THRESHOLD = 0.55 #make adjustable?
MAX_PAUSE_SECONDS     = 2.0

#Shannon Entropy - see original python script for details
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

                if data.get("done"):
                    break

    except Exception as e:
        output.textContent = f"Error: {str(e)}"

    createLog(token_count, hesitation_count, total_uncertainty/token_count if token_count > 0 else 0.0)





