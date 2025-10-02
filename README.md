# DIAAD — Digital Interface for Aggregate Analysis of Dialog

DIAAD is a small toolkit for batched dialog analysis that includes workflows for analyzing digital conversation turns and [POWERS](https://doi.org/10.3233/ACS-2013-20107) coding. It complements (and imports) the monologic speech analysis system [RASCAL](https://github.com/nmccloskey/RASCAL). 

---

## Overview (more details below)

- **Digital Conversation Turns Analysis**
   - tracking turn-taking in dialogs can reveal meaningful linguistic and psychosocial patterns [Tuomenoksa, et al., 2020](https://doi.org/10.1080/02687038.2020.1852518)
   - recording turns with a sequence of digits enables analysis of tallies and transition probabilities (see below) 
- **POWERS Coding**
   - Profile of Word Errors and Retrieval in Speech (POWERS) is an aphasiological coding system for analyzing dialogic speech (Herbet, et al., 2013)
   - DIAAD functionalities:
      - generates coder workbooks, automating most fields
      - summarizes coding and reports ICC2 values between coders
      - evaluates and optionally reselects reliability coding
---

## Web App

You can use DIAAD in your browser — no installation required:

👉 [Launch the DIAAD Web App](https://diaad-dialog.streamlit.app/)

---

## Installation

We recommend installing DIAAD into a dedicated virtual environment using Anaconda:

### 1. Create and activate your environment:

```bash
conda create --name diaad python=3.12
conda activate diaad
```

### 2. Install from PyPI:
```bash
pip install diaad
```

### ...or from GitHub:
```bash
pip install git+https://github.com/nmccloskey/diaad.git@main
```

### 3. Install the `en_core_web_trf` model (for automating POWERS):
```bash
python -m spacy download en_core_web_trf
```

---

## Setup

To prepare for running DIAAD, complete the following steps:

### 1. Create your working directory:

We recommend creating a fresh project directory where you'll run your analysis.

Example structure:
```
your_project/
├── config.yaml           # Configuration file (see below)
└── diaad_data/
    └── input/            # Place your .cha or .xlsx files here
                          # (DIAAD will make an output directory)
```

### 2. Provide a `config.yaml` file

This file specifies the directories, coders, reliability settings, and tier structure.

You can download the example config file from the repo or create your own like this:

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.2
automate_POWERS: true
just_c2_POWERS: false
exclude_participants:
coders:
- '1'
- '2'
- '3'
tiers:
  time:
    values:
    - PreTx
    - PostTx
    blind: true
  client_id:
    values: \d+
  setting:
    values:
    - LargeGroup
    - SmallGroup
```

> See [RASCAL](https://github.com/nmccloskey/RASCAL) for more information about the **tier** system for organizing data based on .cha file names.

---

## Quickstart — Command Line

DIAAD exposes a concise CLI with subcommands:

```bash
# Analyze digital conversation turns
diaad turns

# POWERS workflow
diaad powers make       # prepare POWERS coding files
diaad powers analyze    # analyze completed POWERS coding
diaad powers evaluate   # evaluate completed POWERS reliability coding
diaad powers reselect   # randomly reselect reliability subset
```
---
# Digital Conversation Turns (DCT) Protocol

DIAAD includes a lightweight system for analyzing **digital conversational turns** in group treatment sessions with people with aphasia.  
Instead of simple tallies, the DCT protocol records the **sequence of turns** compactly, enabling analysis of turn-taking dynamics and engagement, with optional markers for capturing turn qualities (e.g., length/substantiveness).

---

## Coding Procedure

### 1. Speaker Assignment
- `0` = Clinician(s) (all individuals not receiving treatment collapsed under this code)
- `1` = Participant 1
- `2` = Participant 2
- Continue incrementing (`3`, `4`, …) as needed.

### 2. Turn Entry with Markers
For each conversational turn, enter the assigned digit for the speaker (e.g., `0`, `1`, `2`).

Marking system:
- Digits are followed by one dot `.` (mark1), two dots `..` (mark2) or no dots
- Recommended usage:
   - Add `.` if the turn is *substantial* (contains an independent clause).  is   
   - Add `..` if the turn is *monologic* (contains at least two independent clauses)
   - Add no dots otherwise, or the turn is *minimal* (brief/no full idea)

### 3. Input Coding Table Format
- Turns are entered sequentially as a continuous string of digits and dots. 
- Bins are recommended for some temporal granularity (e.g., six 10-minute bins for a 1-hour conversation treatment session).

### Example: Digital Conversation Turns Coding Input

| site | session | group   | coder | bin | turns |
|------|---------|---------|-------|-----|-------|
| TU   | 12      | Dyad1   | NM    | 1   | `212012.02121210.10101.210.12.021212121210.210.2.1.010121.010.110.2102.12.` |
| TU   | 12      | Dyad1   | NM    | 2   | `0202.121212101.011101.2.12.120201.212101020202.10.21212.02.12010212.` |
| TU   | 12      | Dyad1   | NM    | 3   | `12..121.212.1212.0202.12120.201.210101..2012121.2121.2..1212.12.020.2.0` |
| TU   | 12      | Dyad1   | NM    | 4   | `010202.02121021020212101.01012101210010102.1210101010101010101010121020.1.` |
| TU   | 12      | Dyad1   | NM    | 5   | `0.121210.1010102120.102.02120212.0.2.020212121202121212.120.21010101212121` |
| TU   | 12      | Dyad1   | NM    | 6   | `2120210101212121212.10121202.12.02.1212010202.02.02.0202.020201202020.22.02012102002.012102` |
| TU   | 4       | LgGroup | NM    | 1   | `4.24.242424.0640.4.206.434343430606.060436.3706.0406.76760.602.502.326207.07.67.06767.3737.17.0701270606.06.54321007` |
| TU   | 4       | LgGroup | NM    | 2   | `763670.50505620507102..02404676.70101...010.707057574767.6..76717.01.7010141.4..1014.3401.671..61016161.721.77414.0` |
| TU   | 4       | LgGroup | NM    | 3   | `2.0.2.0.3.13.23.01313535737037.0.7.137314.` |
| TU   | 4       | LgGroup | NM    | 4   | `4.0.5.35.05.0.5..7575404.53436..40575754..24242..575.4375.45705.20.6.` |
| TU   | 4       | LgGroup | NM    | 5   | `06.007070767676050.21627.17.106063434607571270101.61.01016.161.2.0.1.01` |
| TU   | 4       | LgGroup | NM    | 6   | `0.607.2707.07.06..06.06.4603403212607201202..2702760276..020.1212606016..70.701702.1.70731313510.` |

---

## Analytic Opportunities
This richer symbolic format enables:
- **Turn counts & proportions** per participant  
- **Substantial vs. monologic** turn ratios  
- **Transitions** (e.g., clinician → participant, participant → participant)  
- **Speaker dominance indices**  
- **Engagement rates** between participants  
- **Distribution metrics** (e.g., Gini index, entropy)  
- **Transition matrices & dyadic graphs**  
- **Temporal trends** (with optional bins)  
- **Reliability**: inter-coder sequence comparisons (e.g., Levenshtein distance)
- **Correlation** with treatment outcome measures (e.g., ACOM, WAB) for longitudinal studies 

---

## Limitations
- **Turn Overlap**: current system assumes sequentialization - not uncommonly violated in group settings.  
- **Subjectivity**: coder judgment needed for speaker boundaries and substantiality. Calibration recommended.  
- **Binary turn length**: `mark1` vs. `mark2` is coarse; future versions may refine scale.  
- **Scalability**: beyond 9+ participants, codes like `P1`, `P2`, `C` may be adopted.
 
---

# Profile of Word Errors and Retrieval in Speech (POWERS) coding

## Measures

The POWERS coding system addresses the need to assess language abilities in conversation for people with aphasia. DIAAD facilitates quantification of the following subset of POWERS variables for both the clinician and client (see the [POWERS](https://doi.org/10.3233/ACS-2013-20107) manual for full details): 

   - **filled pauses** - disfluencies like "um", "uh", "er", etc.
   - **speech units** - these more or less map onto tokens excluding filled pauses
   - **content words** - nouns (including proper nouns), non-auxiliary verbs, adjectives, -ly-terminal adverbs, and numerals
   - **nouns** - a subset of content words
   - **number of turns** - a verbal contribution to the conversation with three types:
      - *substantial turn* - contains at least one content word
      - *minimal turn* - hands the turn back to the other conversation partner
      - *subminimal turn (a nonce, non-canonical term)* - not classifiable as either type above
   - **collaborative repair** - sequences of turns devoted to overcoming communicative error/difficulty

## Automation (reliability details pending)

DIAAD automates as much as possible. Below are descriptions of automatability and ICC2 utterance-level reliability metrics on a stratified (by study site, mild/severe aphasia profile, and pre-/post-tx test) random selection of XX samples (XX utterances).
   - **fully automated** with regex and spaCy (`en_core_web_trf`):
      - *filled pauses:*
      - *speech units:*
      - *content words:*
      - *noun count:*
   - **semi-automated** with a computational first pass followed by manual checks:
      - *turn type:*
   - **fully manual** given the rich contextual dependencies:
      - *collaborative repair*

## Typical Workflow

1. **Tabularize utterances (if needed)**  
   If `*Utterances*.xlsx` files aren’t present, DIAAD will call RASCAL to read `.cha` files and tabularize utterances, Assigning samples unique identifiers at the utterance and transcript levels.

2. **Prepare POWERS coding files**  
   `diaad powers make` creates full dataset plus reliability coding workbooks, with most coding automated.

3. **Human coding**  
   Coders complete POWERS annotations in the generated spreadsheets.

4. **Analyze**  
   `diaad powers analyze` aggregates and reports POWERS metrics at the turn, speaker, and dialog levels.

5. **Reliability evaluation**  
   `diaad powers evaulate` matches reliability files and runs ICC2 evaluation.

6. **Reliability subset (optional)**  
   `diaad powers reselect` Reselects reliability coding subset if ICC2 measures fail to meet threshold (0.7 a typical minimum).

---

## 🧪 Testing

This project uses [pytest](https://docs.pytest.org/) for its testing suite.  
All tests are located under the `tests/` directory, organized by module/function.

### Running Tests
To run the full suite:

```bash
pytest
```
Run "quietly":
```bash
pytest -q
```
Run a specific test file:
```bash
pytest tests/test_samples/test_digital_convo_turns_analyzer.py
```
---

## Status and Contact

I warmly welcome feedback, feature suggestions, or bug reports. Feel free to reach out by:

- Submitting an issue through the GitHub Issues tab

- Emailing me directly at: nsm [at] temple.edu

Thanks for your interest and collaboration!

---

## Citation & Acknowledgments

Full details of the POWERS coding system can be found in the manual:

> Herbert, R., Best, W., Hickin, J., Howard, D., & Osborne, F. (2013). Powers: Profile of word errors and retrieval in speech: An assessment tool for use with people with communication impairment. CQUniversity.

If DIAAD supports your work, please cite the repo:

> McCloskey N. (2025). DIAAD: Digital Interface for Aggregate Analysis of Dialog. GitHub repository. https://github.com/nmccloskey/diaad

---
