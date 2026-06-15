# Configuration

DIAAD configuration controls where files are read and written, how samples are selected, how identifiers are interpreted, which file names commands expect, and which optional processing behaviors are enabled. Most users should keep a project-specific configuration with each analysis project, even though DIAAD can run from packaged defaults for quick tests and generated examples.

This page explains the shared configuration model. Command-specific pages later in the manual explain which settings matter for individual commands.

## Configuration Sources

DIAAD starts from packaged defaults and then applies user configuration and command-line overrides.

For CLI runs, the loading order is:

1. DIAAD packaged defaults.
2. A user configuration source, if one is provided.
3. Command-line overrides such as `--input-dir`, `--output-dir`, and `--set KEY=VALUE`.

If the CLI command does not include `--config`, DIAAD checks for a `config/` directory in the current working directory. If that directory exists, DIAAD uses it. If it does not exist, DIAAD uses only packaged defaults.

The web app uses a temporary workspace for each run. It accepts split configuration files or a configuration built in the app, then sets input and output paths to the app's temporary folders for that session.

## Recommended Project Layout

For real projects, use a folder like this:

```text
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
    output/
```

Run CLI commands from `your_project/`:

```bash
diaad transcripts tabularize
```

With that layout, DIAAD finds `config/` automatically and resolves relative input and output paths from the project folder.

You can also point to a configuration source explicitly:

```bash
diaad transcripts tabularize --config config
diaad transcripts tabularize --config diaad_config.yaml
```

## Split and Nested Forms

The recommended form is a split configuration directory:

```text
config/
  project.yaml
  advanced.yaml
```

`project.yaml` contains common study-level settings:

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
random_seed: 99
reliability_fraction: 0.2
shuffle_samples: true
auto_tabularize: false
metadata_fields: {}
```

`advanced.yaml` contains file names, identifier columns, blinding settings, and lower-level processing settings:

```yaml
transcript_table_filename: transcript_tables.xlsx
sample_id_column: sample_id
utterance_id_column: utterance_id
auto_blind: false
blind_columns:
  - sample_id
```

The CLI also accepts a single nested YAML file:

```yaml
project:
  input_dir: diaad_data/input
  output_dir: diaad_data/output
advanced:
  transcript_table_filename: transcript_tables.xlsx
  sample_id_column: sample_id
  utterance_id_column: utterance_id
```

The CLI permits missing sections and fills them from defaults. The web app expects the split form when configuration files are uploaded.

## Project Settings

Current `project` settings are:

| Setting | Default | Purpose |
|---|---|---|
| `input_dir` | `diaad_data/input` | Project input directory for CLI runs. |
| `output_dir` | `diaad_data/output` | Base output directory for CLI runs. |
| `random_seed` | `99` | Seed used for reproducible sample selection and shuffling. |
| `reliability_fraction` | `0.2` | Fraction of samples selected for reliability workflows. |
| `shuffle_samples` | `true` | Whether eligible samples are shuffled before selection or tabularization steps that use ordering. |
| `strip_clan` | `true` | Whether CLAN formatting is stripped in relevant transcript comparison workflows. |
| `prefer_correction` | `true` | Whether corrected forms are preferred in relevant transcript reliability processing. |
| `lowercase` | `true` | Whether text is lowercased in relevant comparison workflows. |
| `exclude_speakers` | `[]` | Speaker codes to exclude in transcript-derived outputs where supported. |
| `auto_tabularize` | `false` | Whether later commands may create transcript tables automatically when needed. |
| `num_bins` | `4` | Number of bins for selected sample-level template workflows. |
| `num_coders` | `0` | Number of coders for coding-file generation workflows that create coder-specific materials. |
| `stimulus_column` | `''` | Optional metadata column used by workflows that organize outputs by stimulus. |
| `automate_powers` | `true` | Whether POWERS file generation should use automated first-pass support when available. |
| `metadata_fields` | `{}` | Project-specific metadata field definitions. |

The most common settings to adjust early are `input_dir`, `output_dir`, `metadata_fields`, `reliability_fraction`, `num_coders`, `exclude_speakers`, and `stimulus_column`.

## Advanced Settings

Current `advanced` settings are:

| Setting | Default | Purpose |
|---|---|---|
| `transcript_table_filename` | `transcript_tables.xlsx` | Expected transcript table workbook name. |
| `sample_id_column` | `sample_id` | Canonical sample identifier column. |
| `utterance_id_column` | `utterance_id` | Canonical utterance identifier column. |
| `reliability_tag` | `_reliability` | Suffix/tag used for transcription reliability files. |
| `reliability_dirname` | `reliability` | Directory name used for reliability materials. |
| `cu_paradigms` | `[]` | Complete Utterance paradigms expected in CU workflows. |
| `cu_samples_filename` | `cu_coding_by_sample_long.xlsx` | Complete Utterance sample-level coding file name. |
| `cu_utts_filename` | `cu_coding_by_utterance.xlsx` | Complete Utterance utterance-level coding file name. |
| `word_count_filename` | `word_counting.xlsx` | Manual word-count coding file name. |
| `word_count_column` | `word_count` | Column containing manually entered word counts. |
| `wc_samples_filename` | `word_counting_by_sample.xlsx` | Sample-level word-count analysis file name. |
| `speaking_time_filename` | `speaking_times.xlsx` | Speaking-time workbook name for rate calculations. |
| `speaking_time_column` | `speaking_time` | Column containing speaking time values. |
| `powers_coding_filename` | `powers_coding.xlsx` | POWERS coding workbook name. |
| `powers_reliability_filename` | `powers_reliability_coding.xlsx` | POWERS reliability coding workbook name. |
| `spacy_model_name` | `en_core_web_sm` | spaCy language model name for NLP-backed workflows. |
| `dct_coding_filename` | `conversation_turns.xlsx` | Digital Conversational Turns primary coding workbook name. |
| `dct_coding_reliability` | `conversation_turns_reliability.xlsx` | Digital Conversational Turns reliability coding workbook name. |
| `target_vocabulary_resource_path` | `''` | Optional custom target-vocabulary resource path. |
| `auto_blind` | `false` | Whether supported workflows should automatically use configured blinding columns. |
| `blind_columns` | `[sample_id]` | Columns to encode for blinding. |
| `metadata_source` | `transcript_tables.xlsx` | Metadata source file for functionality that reads metadata independently. |
| `id_columns` | `[sample_id, utterance_id]` | Identifier columns preserved across selected workflows. |
| `codebook_filename` | `''` | Optional codebook filename for workflows that use one. |

The most common settings to adjust early are `transcript_table_filename`, identifier column names, blinding settings, `cu_paradigms`, `target_vocabulary_resource_path`, and filenames for coding workbooks your project already maintains.

## Path Behavior

In CLI runs, relative paths in configuration are resolved from the current project root, which is normally the directory where you run the `diaad` command.

For example:

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

run from:

```text
your_project/
```

resolve to:

```text
your_project/diaad_data/input
your_project/diaad_data/output
```

Each normal CLI run creates a timestamped run directory under the configured output directory:

```text
diaad_data/output/diaad_YYMMDD_HHMM/
```

The web app does not use local paths from uploaded configuration as browser-accessible paths. It copies uploaded files into a temporary workspace and returns outputs in a ZIP file.

## Command-Line Overrides

The CLI can override selected configuration values for a single invocation.

Use direct path flags for input and output directories:

```bash
diaad transcripts tabularize --input-dir study_input --output-dir study_output
```

Use `--set KEY=VALUE` for other values:

```bash
diaad transcripts tabularize --set project.random_seed=123
diaad transcripts tabularize --set reliability_fraction=0.25
```

Sectioned keys such as `project.random_seed` and `advanced.transcript_table_filename` are clearest. Unsectioned keys are accepted only when DIAAD can match them to a known configuration field.

List values are usually easier and safer to edit in YAML than on the command line. For project-level changes that should be repeated, update the configuration files instead of relying on long `--set` commands.

## Checking Effective Configuration

Use a dry run before processing data when configuration has changed:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

To save the resolved configuration:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml --dry-run-config-out resolved_config.yaml
```

Dry-run output reflects packaged defaults, user configuration, and CLI overrides together. It is a good way to confirm paths, filenames, blinding settings, and reliability settings before a run creates outputs.

## Metadata Fields

`metadata_fields` defines project-specific metadata extracted from transcript files. Values may be strings or lists, depending on the metadata field being represented.

Example:

```yaml
metadata_fields:
  group:
    - control
    - treatment
  session: "Session:\\s*(\\w+)"
```

Metadata field definitions are project-specific and should be checked against the project's transcript conventions. Later workflow and transcript documentation should explain how metadata fields affect particular outputs.

## Blinding Settings

Blinding settings live in `advanced.yaml`:

```yaml
auto_blind: false
blind_columns:
  - sample_id
```

`auto_blind` controls whether supported workflows automatically apply configured blinding behavior. `blind_columns` names the columns to encode. Standalone blinding commands can also be used explicitly.

Software blinding is not the same as full de-identification. A workflow may hide sample identifiers while transcript content, contextual details, staff memory, or file handling practices still affect practical privacy or coder masking.

## Validation

DIAAD normalizes configuration values before running. Important validation rules include:

- `reliability_fraction` must be greater than `0` and less than or equal to `1`.
- `num_bins` must be at least `1`.
- `sample_id_column`, `utterance_id_column`, and `spacy_model_name` must be non-empty strings.
- `id_columns` must contain non-empty strings.
- Boolean strings such as `true`, `false`, `yes`, `no`, `1`, and `0` are normalized when possible.

If configuration fails validation, fix the YAML or CLI override and rerun the command.

## Recommendations

Use split configuration files for real projects. Keep `project.yaml` readable and study-facing, and edit `advanced.yaml` more cautiously because it defines filenames and identifier conventions that downstream outputs depend on. Run `--dry-run-config` after changing settings, especially before reliability selection, blinding, or any workflow that reads manually completed coding files.
