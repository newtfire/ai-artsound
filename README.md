# ai-artsound uncertainty monitor

A project to visualize and sonify the uncertainty of a language model's "thinking" process.
Designed for the Penn State Behrend DIGIT program and VARIA lab.

This is a tool for visualizing a language model's uncertainty in real time as it generates text. At the alpha stage, running only in the shell, each token is color-coded by confidence level: from cool blue (certain) through yellow/orange (hesitating) to red (genuinely torn between multiple possibilities). High-uncertainty moments trigger a dramatic pause, helping to visualize the model's "thinking process". As the project develops, we intend to build this an art exhibit with light and sound effects to dramatize the experience of uncertainty in the model's processing.

In the alpha version, it runs only in a shell environment running Python 3 (we recommend 3.13), and depending on the local presence of Ollama to pull in a Qwen model. The script is set to pull Qwen2:5:15b, but can readily be changed to access a different model.

We intend to develop this as an interactive art exhibit project. 

---

## How it works

As the model generates each word (token), the script retrieves the probability the model assigned to that choice alongside its top alternative candidates. It computes a [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) score from those probabilities: a measure of how evenly spread the possibilities were. High entropy means the model was genuinely uncertain; low entropy means it was confident. This score drives the color display and the pacing of the output.

---

## Requirements

- Python 3.13
- [Ollama](https://ollama.com) (runs the language model locally)

---

## Setup

These instructions work for both **macOS** and **Windows**. Commands are the same unless noted.

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com).

On **macOS**, Ollama installs as a menu bar app and starts automatically at login. You'll see a small llama icon in your menu bar when it's running. You do not need to run any command to start it.

On **Windows**, Ollama runs as a background service after installation. Check the system tray for the icon.

Once installed, pull the language model the project uses:

```bash
ollama pull qwen2.5:1.5b
```

This downloads about 1GB and only needs to be done once.

### 2. Get the project files

If you received a ZIP file, unzip it to a folder of your choice. If you're cloning from GitHub:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 3. Create a virtual environment

A virtual environment keeps the project's dependencies isolated from the rest of your Python installation.

**macOS / Linux:**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You'll know the virtual environment is active when your terminal prompt shows `(.venv)` at the start.

### 4. Install dependencies
a. you need the requests library 

```bash
pip install requests
```

```bash
pip install -r requirements.txt
```

---

## Running the script

Make sure your virtual environment is active (you'll see `(.venv)` in the prompt), then:

```bash
python uncertainty-monitor.py
```

This launches **interactive mode**. You'll be prompted to type a sentence for the model to continue. Press Enter with no text to use the built-in default prompt. Type `quit` or press Ctrl-C to exit.

You can also pass a prompt directly as a command-line argument for a single run:

```bash
python uncertainty-monitor.py "The thing that surprised me most was"
```

#### As you add new pip installations

```bash
pip freeze > requirements.txt
```

### Tips for interesting output

Prompts that work best are ones where multiple continuations are equally plausible — open emotional descriptions, incomplete sentences, ambiguous scenarios. Factual questions tend to produce low uncertainty throughout. Try prompts like:

- `"The feeling I get when I look at the night sky is"`
- `"What I remember most about that day is"`
- `"The strange thing about silence is"`

---

## Configuration

All tunable parameters are at the top of `uncertainty-monitor.py`:

| Variable | Default | What it does |
|---|---|---|
| `MODEL` | `qwen2.5:1.5b` | Which Ollama model to use |
| `UNCERTAINTY_THRESHOLD` | `0.55` | Score above which a hesitation pause triggers |
| `MAX_PAUSE_SECONDS` | `2.0` | Maximum pause length at peak uncertainty |
| `TOP_K_CANDIDATES` | `5` | How many alternative tokens to request per step |
| `PROMPT` | *(see file)* | Default prompt used when Enter is pressed with no input |

The color palette is also editable — look for the `COLOR_*` variables in the color section of the file.

---

## Connecting to hardware (next stage)

The function `on_uncertainty_event(token, score, candidates)` in the script fires on every single token. It currently does nothing (`pass`), but this is where lighting and sound triggers will go. It receives:

- `token` : the word the model just generated
- `score` : a float from 0.0 (confident) to 1.0 (maximally uncertain)
- `candidates` : a dictionary of the top alternative tokens and their log probabilities

Replace `pass` with your Arduino serial write, MCP tool call, OSC message, or any other hardware trigger.

---

## Troubleshooting

**`Cannot reach Ollama at http://localhost:11434`**
Ollama is not running. On macOS, check the menu bar for the llama icon. On Windows, check the system tray. If it's not there, open the Ollama application to start it.

**`(no logprob data)` in the output table**
Make sure `"top_logprobs"` is set in the request payload (it should be by default). Also confirm your Ollama version is 0.12.11 or later: run `ollama --version` to check.

**Colors don't display correctly in VS Code's terminal**
VS Code's integrated terminal can have issues rendering true-color ANSI. Run the script in your system terminal (Terminal.app on macOS, Windows Terminal on Windows) instead — this is the recommended environment for the exhibit anyway.

**PowerShell says "running scripts is disabled on this system"**
Run this command once to allow local scripts, then try activating the virtual environment again:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Acknowledgements

Developed with assistance from Claude Sonnet 4.6.
