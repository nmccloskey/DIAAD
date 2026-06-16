# Configuration Sources, Defaults, and Overrides Usage Guide

Configuration is the main way to make DIAAD runs reproducible. It controls input and output directories, identifiers, file names, reliability settings, blinding settings, transcript processing options, and module-specific defaults.

## Packaged Defaults

DIAAD can run without user configuration. In that case it uses packaged defaults such as:

```text
input_dir: diaad_data/input
output_dir: diaad_data/output
transcript_table_filename: transcript_tables.xlsx
sample_id_column: sample_id
utterance_id_column: utterance_id
auto_tabularize: false
auto_blind: false
```

This is useful for quick tests and generated examples. For study data, use project configuration files so the project carries its assumptions with it.

## Project Configuration

The preferred project form is:

```text
config/
  project.yaml
  advanced.yaml
```

Use `project.yaml` for study-facing settings that are likely to change between projects:

- `input_dir`
- `output_dir`
- `random_seed`
- `reliability_fraction`
- `shuffle_samples`
- `exclude_speakers`
- `auto_tabularize`
- `num_coders`
- `stimulus_column`
- `metadata_fields`

Use `advanced.yaml` for file and identifier conventions that downstream outputs depend on:

- `transcript_table_filename`
- `sample_id_column`
- `utterance_id_column`
- module coding workbook filenames
- `speaking_time_filename`
- `target_vocabulary_resource_path`
- `auto_blind`
- `blind_columns`
- `metadata_source`
- `id_columns`
- `codebook_filename`

The full setting table is in Configuration (`docs/manual/02_operation/04_configuration.md`).

## Nested Config Files

The CLI also accepts a single nested YAML file with top-level `project:` and `advanced:` sections:

```yaml
project:
  input_dir: diaad_data/input
  output_dir: diaad_data/output
advanced:
  transcript_table_filename: transcript_tables.xlsx
```

A directory may also contain a single nested `config.yaml`. Do not mix split files and nested config files in the same config directory; that is ambiguous.

## Command-Line Overrides

Use direct flags for one-off path changes:

```bash
diaad transcripts tabularize --input-dir site_a/input --output-dir site_a/output
```

Use `--set` for other one-off settings:

```bash
diaad transcripts tabularize --set project.random_seed=123
diaad powers files --set advanced.powers_coding_filename=site_a_powers.xlsx
```

Unsectioned keys are accepted only when DIAAD can map them to a known setting:

```bash
diaad transcripts tabularize --set reliability_fraction=0.25
```

For repeated work, update YAML rather than accumulating long command lines.

## Web App Configuration

The web app accepts split `project.yaml` and `advanced.yaml` files or lets users build configuration in the app. During a web run, DIAAD rewrites `input_dir` and `output_dir` to temporary web-session folders. Other settings still affect processing.

Inspect the web app's Config Preview before running. The web builder is a convenience interface, and some starter values are more workflow-friendly than the packaged CLI defaults. For example, the builder may prefill common speaker, coder, stimulus, CU paradigm, and metadata-field values so users can see expected shapes.

TODO: Confirm whether the web config-builder starter values should be documented as intentional presets, or whether they should be aligned more strictly with packaged defaults before publication.

## When To Use Dry Run

Use `--dry-run-config` when:

- starting a new project;
- switching between CLI and web-generated config files;
- changing input or output directories;
- changing identifier columns;
- changing blinding settings;
- changing transcript table or coding workbook filenames;
- using `--set` overrides;
- diagnosing a command that cannot find expected inputs.

Dry-run output includes packaged defaults, user configuration, and CLI overrides together.

## Practical Recommendations

Keep project configuration under version control when policy permits. If the data themselves cannot be stored in the same repository, store the config files and run artifacts with the analysis record.

Use packaged defaults for exploration, not as an invisible project method. When settings affect sampling, blinding, metadata extraction, or downstream file matching, write them into `project.yaml` or `advanced.yaml` so the project can be rerun and reviewed.

## Read Next

- Configuration operation page: `docs/manual/02_operation/04_configuration.md`
- CLI and web execution: `docs/manual/05_functionalities/02_cli_web_execution/01_quickstart.md`
- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
