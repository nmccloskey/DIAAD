# DIAAD — Database-oriented, Integrative Architecture for Analyzing Discourse

![PyPI version](https://img.shields.io/pypi/v/diaad)
![Python](https://img.shields.io/pypi/pyversions/diaad)
![License](https://img.shields.io/github/license/nmccloskey/DIAAD)
<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diaad.streamlit.app/) -->

> Open-source Python infrastructure for reproducible discourse analysis in clinical aphasiology.

---

⚠️ **Development Notice**

**DIAAD** is the actively developed successor to the earlier **RASCAL** software architecture.

DIAAD generalizes and expands the original laboratory-specific pipeline into a modular,
database-native framework designed to support reproducible discourse analysis workflows
that integrate manual and automated coding.

The original **RASCAL** repository is retained for historical continuity and may later
be redeveloped as a laboratory-specific wrapper built on top of DIAAD.

---

## Installation

We recommend installing DIAAD into a dedicated virtual environment using Anaconda:

### 1. Create and activate your environment:

```bash
conda create --name diaad python=3.12
conda activate diaad
```

### 2. Download DIAAD:
(PyPI distribution pending)
```bash
# install the latest development version
pip install git+https://github.com/nmccloskey/diaad.git@main
```
---

## Program Overview (many components pending)

DIAAD is a **relational database-native discourse analysis architecture** designed to scaffold reproducible measurement workflows that integrate manual and automated coding.

### Novelty and Niche

Discourse research benefits from powerful standalone tools (e.g., automated speech recognition systems and linguistic analysis engines). However, large-scale discourse research also requires **robust management and integration of transcripts, metadata, and measurements** across multiple stages of human and computational analysis.

DIAAD addresses this need by providing **revision-tolerant, database-centered infrastructure** that unifies:

- management of transcripts and associated metadata and clinical measures  
- preset and customizable **human-in-the-loop coding workflows**  
- integration of external data sources, including:
  - automated linguistic analysis
  - macrostructural coding systems
- **language- and metric-agnostic reproducible measurement pipelines**

Within this architecture, DIAAD formalizes discourse measurement workflows within a stable relational framework.

DIAAD also implements several automated procedures that illustrate the affordances of this infrastructure, including:

- algorithmic **character-level transcription reliability evaluation**
- **randomized, blinded sampling** for manual annotation tasks
- computational extraction of selected **POWERS features** (e.g., speech units, content words)

These automated procedures are secondary to the architectural contribution and serve primarily as **proof-of-concept implementations within a broader database-native workflow**.

---

## Core Architectural Features

### Relational Database Model

- Transcriptions are **tabularized with stable unique identifiers** at both document and utterance levels.
- Transcript metadata are automatically extracted from filenames based on user-defined configuration, enabling hierarchical dataset organization.
- Transcript content, metadata, and derived measurements are stored within a **unified relational database structure**, enabling:

  - filtering and aggregation
  - reproducibility
  - transparent audit trails
  - integration with external datasets

This structure allows researchers to leverage relational database logic for **scalable discourse analysis workflows**.

---

### Human-in-the-Loop Design

DIAAD is designed to support workflows where manual coding and computational analysis coexist.

Key features include:

- **Position indices** that allow transcript revision without breaking unique identifiers.
- Stable identifiers enabling **randomized and blinded sampling** for manual coding.
- Built-in scaffolding for manual annotation workflows, including:

  - generation of coding tables and reliability subsets
  - Complete Utterance coding
  - manual word counting workflows
  - POWERS coding tables

- automated **reliability evaluation** across transcription and coding procedures.

---

### Software Integration

DIAAD integrates with widely used discourse analysis tools while maintaining a unified relational data structure.

Supported integrations include:

- **Batchalign** (Li et al., 2023) for automated CHAT-formatted transcription upstream of DIAAD workflows.
- **CLAN** (MacWhinney & Fromm, 2022) via CLI wrappers, allowing established computational analyses to be incorporated into the same relational framework.

The relational data architecture can be extended to incorporate any external data tables that share compatible sample identifiers.

---

## DIAAD Modules

### Current modules include:

- Transcript table generation 
  - from `.cha` files
  - Metadata extraction
  - Relational database construction
- Alignment-based string similarity for **character-level transcription reliability**
- Complete Utterance (CU) coding workflow
- Manual word count workflow


### Developing modules include:
- **Customizable blinded sampling** for arbitrary manual annotation tasks
- POWERS coding workflow, with computational extraction of selected **POWERS features**
  - speech units
  - disfluencies
  - nouns
  - content words
- **Digital Conversation Turns (DCT)** analysis
- CLAN wrappers for command-line workflows
- Target vocabulary coverage analysis with built-in CoreLex-style resources and optional normative data from Cavanaugh et al. (2021)

---

## Project Lineage

DIAAD evolves from earlier laboratory software developed under the **RASCAL** project.

The transition reflects a shift from a laboratory-specific semi-automated coding pipeline toward a **generalized relational architecture for discourse measurement workflows**.

See `lineage.md` for a detailed description of the project's development history.

---

## Setup

To prepare for running DIAAD, complete the following steps:

### 1. Create your working directory:

We recommend creating a fresh project directory where you'll run your analysis.

Example structure:

```plaintext
your_project/
├── config.yaml           # Configuration file (see below)
└── diaad_data/
    └── input/            # Place your CHAT (.cha) files and/or Excel data here
                          # (DIAAD will make an output directory)
```

### 2. Provide a `config.yaml` file

This file specifies the directories, coders, reliability settings, and tier structure.

You can download the example config file from the repo or create your own like this:

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
random_seed: 8
reliability_fraction: 0.2
coders:
- '1'
- '2'
- '3'
CU_paradigms:
- SAE
- AAE
exclude_participants:
- INV
strip_clan: true
prefer_correction: true
lowercase: true
tiers:
  site:
    values:
    - AC
    - BU
    - TU
    partition: true
    blind: true
  test:
    values:
    - Pre
    - Post
    - Maint
    blind: true
  study_id:
    values: (AC|BU|TU)\d+
  narrative:
    values:
    - CATGrandpa
    - BrokenWindow
    - RefusedUmbrella
    - CatRescue
    - BirthdayScene
```

### Explanation:

- General

  - `random_seed` - ensures deterministic selections for replicability

  - `reliability_fraction` - the proportion of data to subset for reliability (default 20%).

  - `coders` - alphanumeric coder identifiers.

  - `CU_paradigms` - allows users to accommodate multiple dialects if desired. If at least two paradigms are entered, parallel coding columns will be prepared and processed in all CU functions.

  - `exclude_participants` - speakers appearing in .cha files to exclude from transcription reliability and CU coding (neutral utterances).

- Transcription Reliability

  - `strip_clan` - removes CLAN markup but preserve speech-like content, including filled pauses (e.g., '&um' -> 'um') and partial words.

  - `prefer_correction` - toggles policy for accepted corrections '[: x] [*]': True keeps x, False keeps original.

  - `lowercase` - toggles case regularization.

**Specifying tiers:**
The tier system facilitates tabularization by associating a unit of analysis with its possible values and extracting this information from the file name of individual transcripts.

- **Multiple values**: enter as a comma- or newline-separated list. These are treated as **literal choices** and combined into a regex internally. See below examples.
  - *narrative*: `BrokenWindow, RefusedUmbrella, CatRescue`
  - *test*: `PreTx, PostTx`
  
- **Single value**: treated as a **regular expression** and validated immediately. Examples include:
  - Digits only: `\\d+`
  - Lab site + digits: `(AC|BU|TU)\\d+`
  - Three uppercase letters + three digits: `[A-Z]{3}\\d{3}`

- **Tier attributes**
  - **Partition**: creates separate coding files and **separate reliability** subsets by that tier. In this example, separate CU coding files will be generated for each site (AC, BU, TU), but not for each narrative or test value.
  - **Blind**: generates blind codes for CU summaries.

***Example: Tier-Based Tabularization from Filenames (according to the above config).***

Source files:
- `TU88PreTxBrokenWindow.cha`
- `BU77Maintenance_CatRescue.cha`

Tabularization:

| Site | Test  | ParticipantID | Narrative     |
|------|-------|---------------|---------------|
| TU   | Pre   | TU88          | BrokenWindow  |
| BU   | Maint | BU77          | CatRescue     |
---

## Running the Program

Once installed, DIAAD can be run from any directory using the command-line interface.

Commands follow the pattern:

```
diaad <module> <action>
```

Multiple commands may be executed sequentially by separating them with commas.

```
diaad [-h] [--config CONFIG] command [command ...]
```

Examples:

```bash
# Reselect transcription reliability samples
diaad transcripts reselect

# Prepare transcript tables from CHAT transcripts
diaad transcripts tabularize

# Generate CU coding and reliability spreadsheets
diaad cus files

# Run multiple steps in sequence
diaad transcripts tabularize, cus files, words files

# Use a custom configuration directory
diaad cus files --config other_config
```

---

## DIAAD Commands by Module

DIAAD commands are grouped by **module**. Each command corresponds to a specific operation within a module.

---

### Transcripts Module

| Command                  | Description                                     | Input                                       | Output                                                  | Function                                     |
| ------------------------ | ----------------------------------------------- | ------------------------------------------- | ------------------------------------------------------- | -------------------------------------------- |
| `transcripts select`     | Select transcription reliability samples        | Raw `.cha` files                            | Reliability & full sample lists + template `.cha` files | `select_transcription_reliability_samples`   |
| `transcripts evaluate`   | Evaluate transcription reliability              | Reliability `.cha` pairs                    | Agreement metrics + alignment text reports              | `evaluate_transcription_reliability`         |
| `transcripts reselect`   | Reselect transcription reliability samples      | Original + reliability transcription tables | New reliability subset(s)                               | `reselect_transcription_reliability_samples` |
| `transcripts tabularize` | Convert CHAT transcripts into structured tables | Raw `.cha` files                            | Sample- and utterance-level spreadsheets                | `tabularize_transcripts`                     |

---

### Complete Utterances (CU) Module

| Command         | Description                                     | Input                            | Output                                         | Function                     |
| --------------- | ----------------------------------------------- | -------------------------------- | ---------------------------------------------- | ---------------------------- |
| `cus files`     | Generate CU coding and reliability spreadsheets | Utterance tables                 | CU coding + reliability spreadsheets           | `make_cu_coding_files`       |
| `cus evaluate`  | Evaluate CU coding reliability                  | Completed CU coding spreadsheets | Reliability summaries + reports                | `evaluate_cu_reliability`    |
| `cus reselect`  | Reselect CU reliability samples                 | Completed CU coding spreadsheets | New reliability subset(s)                      | `reselect_cu_wc_reliability` |
| `cus analyze`   | Analyze completed CU coding                     | Completed CU coding spreadsheets | Sample- and utterance-level CU analyses        | `analyze_cu_coding`          |
### Word Count Module

| Command          | Description                                             | Input                             | Output                                    | Function                          |
| ---------------- | ------------------------------------------------------- | --------------------------------- | ----------------------------------------- | --------------------------------- |
| `words files`    | Generate word-count coding and reliability spreadsheets | CU coding tables                  | Word count + reliability spreadsheets     | `make_word_count_files`           |
| `words evaluate` | Evaluate word-count reliability                         | Completed word-count spreadsheets | Reliability summaries + agreement reports | `evaluate_word_count_reliability` |
| `words reselect` | Reselect word-count reliability samples                 | Completed word-count spreadsheets | New reliability subset(s)                 | `reselect_cu_wc_reliability`      |

---

### Target Vocabulary Coverage

| Command         | Description                                                         | Input                    | Output                                                     | Function                 |
| --------------- | ------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------- | ------------------------ |
| `vocab file`    | Generate a blank target vocabulary resource JSON template           | None                     | Blank JSON template for custom target vocabulary resources | `make_target_vocab_file` |
| `vocab analyze` | Run target vocabulary coverage analysis with built-in/CoreLex-style tasks | CU and WC summary tables | Summary and long-format detail tables with percentile data | `run_target_vocab`       |

---

### Digital Conversation Turns

| Command         | Description                                          | Input            | Output                                      | Function                        |
| --------------- | ---------------------------------------------------- | ---------------- | ------------------------------------------- | ------------------------------- |
| `turns files`   | Generate digital conversation turn coding templates  | Transcript tables | Primary and reliability turn-coding files   | `make_digital_convo_turn_files` |
| `turns analyze` | Analyze completed digital conversation turn files    | Turn coding files | Speaker, group, session, and transition summaries | `analyze_digital_convo_turns`   |

---

## Typical Workflow Example

A typical workflow might look like:

```bash
# Convert transcripts to tables
diaad transcripts tabularize

# Create CU coding spreadsheets
diaad cus files

# After coding is complete, evaluate reliability
diaad cus evaluate

# Analyze completed CU coding
diaad cus analyze

# Run target vocabulary coverage analysis
diaad vocab analyze

# Generate a blank target vocabulary resource template
diaad vocab file

# Generate digital conversation turn coding templates
diaad turns files
```

Commands can also be chained:

```bash
diaad transcripts tabularize, cus files
```

---

## Configuration Files

By default, DIAAD reads configuration settings from:

```
config.yaml
```

A different configuration file may be specified using:

```bash
diaad transcripts tabularize --config other_config.yaml
```

---

## Notes

### Input Transcriptions

- `.cha` files must be formatted correctly according to CHAT conventions.
- Ensure filenames match tier values as specified in `config.yaml`.
- DIAAD searches tier values using exact spelling and capitalization.

### Transcript Tables (command `transcripts tabularize`)

This function prepares both utterance- and sample-level tabulations of CHAT-formatted transcripts in Excel files, assigning unique alphanumeric identifiers encoding level of analysis – ‘S’ for sample and ‘U’ for utterance – for example, `S008` and `U0246`.

The `transcript_tables.xlsx` output contains two sheets:
 - `samples` for transcript metadata, including file name and tier values
 - `utterances` for transcript content, i.e., (CHAT-coded) utterances & comments (from `%com` lines)

This encoded tabularization: 
- establishes unique, human-readable identifiers that satisfy database logic
- facilitates data management across DIAAD inputs and outputs, including joins between tables
- promotes transparency and consistency in text processing
- minimizes potential bias during manual coding

If not provided when running either `cus files`, `vocab analyze`, or `turns files`, these tables are automatically generated from `.cha` inputs.

### Transcription Reliability Input (command `transcripts evaluate`)

In both the CLI and webapp versions, this DIAAD function matches original with reliability transcripts based on common tiers plus a `reliability` tag in the file name, e.g., `TU88_PreTxBrokenWindow.cha` & `TU88PreTxBrokenWindow_reliability.cha`. The command `transcripts select` generates empty `.cha` file templates with the `reliabiilty` tag for the randomly selected samples. In the CLI version, reliability samples can be collected into a `/reliability` subdirectory in the input folder. The tier values must match the originals, but this provides an alternative to tagging filenames.

### Logs & Metadata

The `logs` subdirectory in the output folder contains two files describing the program run:
 - `diaad_YYMMDD_HHMM.log` - contains log messages (e.g, runtime, detected files, errors, etc.)
 - `diaad_YYMMDD_HHMM_metadata.json` - takes a snapshot of input & output directory content just before program terminates

## 🧪 Testing

This project uses [pytest](https://docs.pytest.org/) for its testing suite.  
All tests are located under the `tests/` directory, organized by module/function.

### Running Tests
To run the full suite:

```bash
pytest
```
Run with verbose output:
```bash
pytest -v
```
Run a specific test file:
```bash
pytest tests/test_coding/test_corelex_analyze.py
```

## Status and Contact

I warmly welcome feedback, feature suggestions, or bug reports. Feel free to reach out by:

- Submitting an issue through the GitHub Issues tab

- Emailing me directly at: nsm [at] temple.edu

Thanks for your interest and collaboration!


## Acknowledgments

DIAAD builds on and integrates functionality from two excellent open-source tools which I highly recommend to researchers and clinicians working with language data:

- [**batchalign2**](https://github.com/TalkBank/batchalign2) – Developed by the TalkBank team, Batchalign provides a robust backend for automatic speech recognition (ASR). DIAAD was designed to function downstream of this system, leveraging its debulletized `.cha` files as input. This integration allows researchers to significantly expedite batch transcription, which without an ASR springboard might bottleneck discourse analysis.

> Liu H, MacWhinney B, Fromm D, Lanzi A. *Automation of Language Sample Analysis*. J Speech Lang Hear Res. 2023 Jul 12;66(7):2421-2433. doi: 10.1044/2023_JSLHR-22-00642. Epub 2023 Jun 22. PMID: 37348510; PMCID: PMC10555460.

- [**coreLexicon**](https://github.com/rbcavanaugh/coreLexicon) – A web-based interface for Core Lexicon analysis developed by Rob Cavanaugh, et al (2021). DIAAD's target vocabulary coverage workflow uses built-in CoreLex-style resources and accesses associated normative data using resource-declared Google Sheet URLs to calculate optional percentiles.

  - **Inspiration & overlap:** DIAAD retains the accuracy, efficiency, and percentile concepts from the original web app while exporting batch-friendly summary and long-format detail tables.
  - **Enhancements:** DIAAD supports batch analysis of uploaded/input tabular data and reports normalized target vocabulary coverage for aggregate comparisons across tasks.
  - **Recommended use cases:** The original web app remains an excellent choice for small numbers of individualized reports, while DIAAD's workflow focuses on higher-throughput downstream statistical workflows.

> Cavanaugh, R., Dalton, S. G., & Richardson, J. (2021). coreLexicon: *An open-source web-app for scoring core lexicon analysis*. R package version 0.0.1.0000. https://github.com/aphasia-apps/coreLexicon
