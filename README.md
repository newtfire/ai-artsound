# ai-artsound uncertainty monitor

A project to visualize and sonify the uncertainty of a language model's "thinking" process.
Designed for the Penn State Behrend DIGIT program and VARIA lab.

This is a tool for visualizing a language model's uncertainty in real time as it generates text. At the alpha stage, running only in the shell, each token is color-coded by confidence level: from cool blue (certain) through yellow/orange (hesitating) to red (genuinely torn between multiple possibilities). High-uncertainty moments trigger a dramatic pause, helping to visualize the model's "thinking process". As the project develops, we intend to build this an art exhibit with light and sound effects to dramatize the experience of uncertainty in the model's processing.

In the alpha version, it runs only in a shell environment running Python 3 (we recommend 3.13), and depending on the local presence of Ollama to pull in a Qwen model. The script is set to pull Qwen2:5:15b, but can readily be changed to access a different model.

We intend to develop this as an interactive art exhibit project. 

---

## How it works

As the model generates each word (token), the script retrieves the probability the model assigned to that choice alongside its top alternative candidates. It computes a [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) score from those probabilities: a measure of how evenly spread the possibilities were. High entropy means the model was genuinely uncertain; low entropy means it was confident. This score drives the color display and the pacing of the output.

### A bit more detail

Shannon entropy is a calculation we're making *after* the model calculates probabilities for the next likely tokens. 

In our Python script we have this line, which is pulling in information from our language model about the array of probabilities:

```python
probs = [math.exp(lp) for lp in top_logprobs.values()]
```

Here's the chain of events of what's happening with with each token's prediction, and how it relates to the Shannon entropy we're working with:

```
neural network
      ↓
   logits  (raw scores, any real number)
      ↓
  softmax  (converts to probabilities summing to 1.0)
      ↓
 log-probs  (what Ollama sends us: the  natural log of those probabilities)
      ↓
math.exp()  (we reverse the log to get probabilities back)
      ↓
Shannon entropy  (we measure how spread the distribution is)
      ↓
uncertainty score  (normalized to 0–1 and applied to our color / pause / visualization / sonification logic. 
We're converting it to percentages for the token readouts)
```

#### Why log-probs instead of probabilities directly?

Ollama is sending us probabilities in natural log form rather than plain probabilities, because token probabilities can be super small (like 10⁻³⁰), difficult to work with. Logarithms keep the numbers in a manageable range. 

#### Adjusting the temperature

Here is another property we can tinker with: **temperature**, which adjusts the likelihood that low-probability responses are selected: Heightening the temperature can make for more "wild" random associations, and different models will have different recommended temperature settings for lucid responses.

  We can adjust the `temperature` in the script, and that will be applied *before* softmax, by dividing all the logits by the temperature value:

```
logits_adjusted = logits / temperature
```

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

### 1a. Pull in some models

Once installed, pull the language models we are using in the project. We started with Qwen but are now evaluating small language models transparently trained on data that is not under copyright. This includes PleIAs, trained on the [Common Corpus dataset](https://huggingface.co/datasets/PleIAs/common_corpus), a 2.3 trillion-token dataset trained entirely on open source data. 

#### To pull in Qwen with ollama (where we started):

```bash
ollama pull qwen2.5:1.5b
```

This downloads about 1GB and only needs to be done once.

##### To pull in Qwen with alternative models using Common Corpus
[See [Installing and Configuring with PleIAs Models](#installing-and-configuring-with-pleias-models) below! 
You need to have worked through the rest of the setup and configured a Python environment to work with other models. 

### 2. Get the project files on your computer

If you use git, clone this repo.
If you downloaded/recieved a ZIP file for this project, unzip it to a folder of your choice. If you're cloning from GitHub:

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
| `MODEL` | `??????` | Which Ollama model to use |
| `UNCERTAINTY_THRESHOLD` | `0.55` | Score above which a hesitation pause triggers |
| `MAX_PAUSE_SECONDS` | `2.0` | Maximum pause length at peak uncertainty |
| `TOP_K_CANDIDATES` | `5` | How many alternative tokens to request per step |
| `PROMPT` | *(see file)* | Default prompt used when Enter is pressed with no input |

The color palette is also editable — look for the `COLOR_*` variables in the color section of the file.

---

## Installing and Configuring with PleIAs Models

#### To pull in Pleias-350m 
(We will evaluate other PleIAs models, too, but this is a good starting point). 

The installation is a little more involved because Pleias models are not in Ollama's library and must be downloaded from HuggingFace and
imported manually. 

##### Step 1: Activate your Python environment and install huggingface-cli 

Make sure your virtual environment (.venv) is active:
Then:

```bash
pip install huggingface_hub
```

Confirm that it works wtih:

```bash
huggingface-cli --version
```

##### Step 2: Download the Pleias model weights

Run this from inside your project folder. The download is ~700MB and may take
several minutes depending on your connection:

```bash
huggingface-cli download PleIAs/Pleias-350m-Preview \
    --local-dir ./pleias-350m-src \
    --local-dir-use-symlinks False
```


The `--local-dir-use-symlinks False` flag is essential — without it, Ollama will
reject the files with an "insecure path" error.

When complete, verify you have a real file (not a symlink) by checking the size:

```bash
ls -lh pleias-350m-src/model.safetensors
```

You should see something around 700MB. If it shows a tiny file with an arrow (->),
the symlink flag didn't take — delete the folder and try again.

##### Step 3: Create the model in Ollama

A `Modelfile-pl-350m` is included in the repository. Run:

```bash
ollama create pleias-350m -f Modelfile-pl-350m
```

Ollama will convert the downloaded weights to its internal format. This may take
a minute or two and will show a progress indicator.

##### Step 4: Confirm the model is registered

```bash
ollama list
```

You should see `pleias-350m:latest` in the list.


##### Step 5: Run the monitor with Pleias

I have saved [a version of the uncertainty-monitor Python script](https://github.com/newtfire/ai-artsound/blob/main/uncertainty-monitor-alpha.py) to work with Pleias: **uncertainty-monitor-alpha.py** 

We can adjust the Pleias models in this script by changing the MODEL variable, using the value we see in `ollama list`. 

Currently, in uncertainty-monitor-alpha.py, this is set to Pleias 350m with 

```python
MODEL = "pleias-350m:latest"
```
Then run this script as usual:


```bash
python uncertainty-monitor.py
```

##### Recommended settings for Pleias in the script

These are already set in the uncertainty-monitor-alpha.py, NOT the same settings as for rhe Qwen model.

```python
"options": {
    "temperature": 0.2,       # PleIAs recommends low temperature
    "repeat_penalty": 1.2,    # prevents repetitive looping
    "top_k": TOP_K_CANDIDATES,
    "num_predict": 80,        # hard cap — model doesn't self-terminate cleanly
},
```

##### Cautions/Caveats with Pleias installations

- The `ollama run hf.co/PleIAs/...` shortcut does NOT work for these models
- Plain `curl` downloads a tiny Git LFS pointer file (~15 bytes), not the real model
- `huggingface-cli` must be used with `--local-dir-use-symlinks False`, otherwise
  Ollama will refuse to load the files with a security error
- The RAG-specialized variants (Pleias-Pico, Pleias-Nano) do not respond well to
  open-ended creative prompts (but are worth looking at just for fun).
  - We should probably work with the `Preview` base models for this exhibit (including Pleias-350m).
  
  
---
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
