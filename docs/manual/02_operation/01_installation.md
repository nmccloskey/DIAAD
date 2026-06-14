# Installation

DIAAD can be used in three practical ways:

- a hosted web app, if a deployment is available for your project;
- a local web app launched from an installed Python environment;
- the local command-line interface, which gives the most control over files, configuration, and repeated runs.

For identifiable, sensitive, or difficult-to-deidentify discourse data, local command-line use is usually the safest default. The web app is useful for learning DIAAD, trying generated examples, and running appropriately deidentified projects.

## Python Version

DIAAD currently requires Python `>=3.12,<3.13`. Use Python 3.12 in a fresh environment.

With Conda:

```bash
conda create --name diaad python=3.12
conda activate diaad
```

With `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux, the activation command is:

```bash
source .venv/bin/activate
```

## Install Options

The base installation provides DIAAD's command-line interface and core analysis dependencies:

```bash
pip install diaad
```

If you are installing from the current development branch instead of a release:

```bash
pip install "diaad @ git+https://github.com/nmccloskey/diaad.git@main"
```

If you are working from a local checkout of the repository:

```bash
git clone https://github.com/nmccloskey/diaad.git
cd diaad
pip install -e .
```

## Dependency Splits

DIAAD separates core processing dependencies from optional web, NLP, and development dependencies.

| Install target | Command | What it adds |
|---|---|---|
| Core CLI | `pip install diaad` | The `diaad` command and core processing libraries. |
| Web app | `pip install "diaad[web]"` | Streamlit support for launching the local web app with `streamlit_diaad`. |
| NLP support | `pip install "diaad[nlp]"` | spaCy support for workflows that use automated NLP processing. |
| Web plus NLP | `pip install "diaad[web,nlp]"` | Both the local web app and spaCy support. |
| Development | `pip install -e ".[dev,web,nlp]"` | Editable install with tests, web support, and NLP support. |

The core installation includes the libraries DIAAD needs for transcript processing, spreadsheet I/O, reliability calculations, vocabulary analysis, and related file handling. These include `psair`, `pandas`, `openpyxl`, `xlsxwriter`, `numpy`, `scipy`, `scikit-learn`, `pingouin`, `pylangacq`, `biopython`, `python-Levenshtein`, `PyYAML`, `tqdm`, `nltk`, `contractions`, `num2words`, and `pyenchant`.

The `nlp` extra installs spaCy, but it does not by itself install a spaCy language model. DIAAD's default advanced configuration names `en_core_web_sm`, so install that model if you plan to use NLP-backed automation:

```bash
python -m spacy download en_core_web_sm
```

Some systems may also require system-level dictionary support for `pyenchant`. If spelling or vocabulary-related functionality fails with an enchant dictionary error, install the appropriate dictionary package for your operating system and retry in the same environment.

## Verify Installation

Check the command-line interface:

```bash
diaad --help
```

If you installed the web extra, check the local web app launcher:

```bash
streamlit_diaad
```

This should start a Streamlit server and print a local URL.

To confirm the installed package version:

```bash
python -c "import diaad; print(diaad.__version__)"
```

## Project Setup

For real analyses, use a separate project folder for each dataset or study.

```text
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
    output/
```

The `config/` directory is optional for quick tests because DIAAD has packaged defaults. For real work, keeping project-specific configuration files is recommended. Configuration is described in more detail in [Configuration](04_configuration.md).

## Generated Examples

After installation, generate example files before running DIAAD on real data:

```bash
diaad examples
```

To generate only the files needed for a particular command:

```bash
diaad examples --for-command "transcripts tabularize"
```

The examples command writes into the configured output location, using DIAAD's normal timestamped run structure. These generated files are useful for checking installation, learning expected input layouts, and comparing your project structure against a known runnable example.

## Updating and Removing DIAAD

To update a release installation:

```bash
pip install --upgrade diaad
```

To remove DIAAD from the active environment:

```bash
pip uninstall diaad
```

If you created a dedicated Conda environment and want to remove it entirely:

```bash
conda remove --name diaad --all
```

## Recommendations

Use a fresh Python 3.12 environment, keep project configuration files with the project, and run command-specific generated examples before processing study data. Install `diaad[web]` only when you want the local web app, and install `diaad[nlp]` only when you need workflows that depend on spaCy. For sensitive discourse data, prefer local CLI runs over hosted web processing.
