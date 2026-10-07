# Project 3: What do AI models learn from chest X-ray reports?

## What this project is about

When a radiologist looks at a chest X-ray, they write a report. Hospitals have millions
of such reports, and they are now used to train AI systems: to label images, to search
old cases, or to write draft reports automatically.

Before an AI system can use a report, it has to convert the text into a form a
computer can calculate with. A language model reads the report and produces a
*representation* of it, also called an *embedding*: a fixed-length sequence of values
that stands for the whole text. Everything the AI system does afterwards works on this
representation, not on the words. So an important question for responsible AI is:
**what is actually inside a representation?** The medical content we want, for sure.
But maybe also the patient's age, the hospital the report comes from, the X-ray
machine, or the writing habits of one radiologist. Information like that can make a
model behave differently for different hospitals or patient groups without anyone
noticing.

In this three-week project you investigate this question on a real dataset of chest
X-ray reports.

| Week | What you do | Instructions |
|------|-------------|--------------|
| 1 | Measure what a report representation contains: the diseases we care about, and what else | [week1/README.md](week1/README.md) |
| 2 | To be announced | [week2/README.md](week2/README.md) |
| 3 | To be announced | [week3/README.md](week3/README.md) |

As in the previous projects, the submission is a poster. Keep the results and figures
from each week; the poster brings them together.

## The data

We use [ReXGradient-160K](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K),
a public dataset of chest X-ray studies from 79 hospitals in 3 US health systems. For
this course we prepared a smaller part of it as one CSV file: 10,000 studies from
6,964 patients. In Week 1 you use only the text and the metadata; the X-ray images are
not needed.

Each study in the file has:

- **Findings**: a few sentences in which the radiologist describes what they see in the
  image. *This is the text you work with.*
- **Impression**: the radiologist's short conclusion, usually one or two sentences.
- **Metadata**: the patient's sex and age, the hospital site and health system, the
  X-ray machine manufacturer, the year, the number of images, and a few technical fields.

**The labels.** The dataset itself has no ready-made yes/no labels for diseases, only
the free text. We created labels for you and put them in the CSV. For that we ran
[CheXbert](https://arxiv.org/abs/2004.09167), a model trained to read radiology
reports, on the *Impression* of every study. For 14 observations (edema, pleural
effusion, pneumonia, ...) it tells whether the Impression says the observation is
`positive`, `negative`, `uncertain` ("cannot exclude ..."), or does not mention it
(`blank`). You find them in the columns `chexbert_edema`, `chexbert_pleural_effusion`,
and so on. Labels made by a program instead of a doctor are called *silver labels*:
good enough to work with, but they contain some mistakes.

**Two rules that follow from this.**

1. The labels come from the Impression, so **the Impression is never your model
   input.** You work with the Findings. Otherwise you would predict a label from the
   very same text it was made from, which proves nothing.
2. For yes/no experiments use only the clear cases: `positive` against `negative`.
   Leave out `uncertain` and `blank`; do not count them as negatives. The helper
   `binary_label` in `src/data.py` does this for you.

## Getting started

You only need a computer with Python. No GPU, no git. The steps take about 15 minutes
plus download time.

**1. Python.** You need Python 3.10 or newer. Open a terminal (macOS: the Terminal app,
Windows: PowerShell) and type `python3 --version` (on Windows `python --version`). If
Python is missing or too old, install it from https://www.python.org/downloads/ and,
on Windows, tick "Add Python to PATH" during installation.

**2. The code.** On this GitHub page click the green **Code** button, then
**Download ZIP**. Unzip it. You get a folder `responsible-ai-rexgradient-main`; you can
rename it. Open a terminal inside that folder. (If you know git you can of course
`git clone` instead.)

**3. The packages.** The project needs a few Python packages, listed in
`requirements.txt`. Install them into a *virtual environment*, a private folder so
that nothing else on your computer is touched:

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

This downloads about 1 GB and takes a few minutes. When the environment is active, the
terminal line starts with `(.venv)`. Every time you open a new terminal, run the
second line again to activate it.

**4. The data.** Download `rexgradient_valid_metadata.csv` from the course webpage and
put it in the `data` folder of the project, so that the file is at
`data/rexgradient_valid_metadata.csv`.

**5. Check.** Run

```bash
python scripts/check_setup.py
```

This script checks that the data is in the right place, that the packages work, and
that the language model can be downloaded and run (about 400 MB, first time only). It
should print something like this and end with `Setup looks good.`:

```
10000 studies, 17007 image rows, 6964 patients
edema             CheXbert status counts: {'blank': 8780, 'positive': 436, 'negative': 415, 'uncertain': 369} -> 851 clear (positive/negative) studies
pleural_effusion  CheXbert status counts: {'blank': 9100, 'positive': 646, 'negative': 138, 'uncertain': 116} -> 784 clear (positive/negative) studies
device: cpu | loading BioClinicalBERT (downloads ~400 MB the first time)
embedding shape (5, 768) (expected (5, 768))
Setup looks good.
```

If it does, continue with [Week 1](week1/README.md). If you get stuck, ask at the
exercise session.

## Credits

Data: ReXGradient-160K (Gradient Health and Harvard Medical School). Labels: CheXbert
(Smit et al., 2020). Language model: Bio_ClinicalBERT (Alsentzer et al., 2019).
