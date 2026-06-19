# DIAAD - Database-oriented, Integrative Architecture for Analyzing Discourse

![PyPI version](https://img.shields.io/pypi/v/diaad)
![Python](https://img.shields.io/pypi/pyversions/diaad)
![License](https://img.shields.io/github/license/nmccloskey/DIAAD)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diaad-web.streamlit.app/)

DIAAD is open-source Python infrastructure for reproducible discourse analysis workflows. It helps organize CHAT transcripts, transcript-derived tables, metadata, manual coding files, reliability checks, blinding, rate calculations, target vocabulary analysis, generated examples, and run artifacts across repeated analyses.

DIAAD is under active development. Output formats, command behavior, and documentation may continue to change before a stable 1.0 release.

## Core Capabilities

- Convert CHAT `.cha` files into sample- and utterance-level transcript tables.
- Generate coding and reliability workbooks for Complete Utterances, word counts, POWERS, Digital Conversation Turns, and custom templates.
- Select, evaluate, and reselect reliability samples.
- Analyze completed coding files and calculate per-minute rates from speaking-time tables.
- Analyze target vocabulary coverage using built-in or custom resources.
- Encode and decode configured identifier columns for blinding workflows.
- Generate runnable synthetic example projects and generated Example I/O documentation.
- Record resolved configuration, logs, manifests, and other run artifacts for reproducibility.

## Installation

DIAAD requires Python `>=3.12,<3.13`.

### Recommended: conda

```bash
conda create -n diaad python=3.12
conda activate diaad
python -m pip install diaad
```

### Alternative: venv

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install DIAAD:

```bash
python -m pip install diaad
```

### Optional dependency groups

```bash
python -m pip install "diaad[web]"      # local Streamlit web app
python -m pip install "diaad[nlp]"      # NLP-backed workflows (spaCy)
python -m pip install "diaad[web,nlp]"  # both
```

If you install the `nlp` extra and plan to use the default POWERS automation model, also install the required spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

### Development installation

```bash
git clone https://github.com/nmccloskey/DIAAD.git
cd DIAAD
python -m pip
```

## Quick Start

Check the command-line interface:

```bash
diaad --help
```

Check the installed version:

```bash
diaad --version
```

Generate a full synthetic example dataset:

```bash
diaad examples
```

Generate examples for one command:

```bash
diaad examples --for-command "transcripts tabularize"
```

Run a basic transcript tabularization command:

```bash
diaad transcripts tabularize
```

Run multiple commands in sequence:

```bash
diaad "transcripts tabularize, cus files, words files"
```

Launch the local web app after installing `diaad[web]`:

```bash
diaad streamlit
```

The legacy `streamlit_diaad` launcher is also available.

## Project Configuration

For real projects, use a project folder with split configuration files:

```text
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
    output/
```

When run from `your_project/`, the CLI uses `./config` automatically. You can also pass a config source explicitly:

```bash
diaad transcripts tabularize --config config
```

Use a dry run to inspect the resolved configuration before processing data:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

See the manual for the current configuration model and settings.

## Documentation

The authored manual is in [`docs/manual`](docs/manual/00_outline.md).

Useful entry points:

- [Installation](docs/manual/02_operation/01_installation.md)
- [Command-line operation](docs/manual/02_operation/02_command_line.md)
- [Web app operation](docs/manual/02_operation/03_webapp.md)
- [Configuration](docs/manual/02_operation/04_configuration.md)
- [Functional overview](docs/manual/01_overview/03_functional_overview.md)
- [Generated Example I/O view](docs/manual/03_features/03_generated_example_io.md)

Generated Example I/O pages are built from packaged example specs and should be regenerated rather than edited by hand.

## Development and Tests

Install development dependencies:

```bash
pip install -e ".[dev,web,nlp]"
```

Run the test suite:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_examples/test_examples.py
```

## Data Privacy

DIAAD can support blinding and organized handling of discourse data, but it is not a de-identification guarantee. For identifiable, sensitive, or difficult-to-deidentify data, prefer local CLI processing and follow the privacy requirements of the project, institution, and data source.

## Acknowledgments

DIAAD builds on the broader open-source language-analysis ecosystem and is designed to work downstream of CHAT/TalkBank-oriented workflows. Fuller methodological context and acknowledgments are maintained in the manual.

## License

DIAAD is distributed under the MIT License. License metadata is declared in [`pyproject.toml`](pyproject.toml).

## Contact

Please use GitHub Issues for bug reports, feature requests, and documentation problems.
