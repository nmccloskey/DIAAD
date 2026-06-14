# Command-Line Operation

The DIAAD command-line interface is the most flexible way to run DIAAD. It is recommended for sensitive data, large or repeated workflows, custom configuration, scripted analysis, and any project where outputs need to be audited across repeated runs.

## Basic Command Form

Most commands use the form:

```bash
diaad <module> <action>
```

For example:

```bash
diaad transcripts tabularize
diaad cus analyze
diaad powers rates
```

Multiple commands can be chained in one invocation by separating command names with commas. Quoting the full command string is the safest form across shells:

```bash
diaad "transcripts tabularize, cus files, words files"
```

## Configuration

DIAAD can run with packaged defaults, a split configuration directory, or a single nested configuration file.

If no `--config` path is supplied, DIAAD checks for a `config/` directory in the current working directory. If that directory is not present, DIAAD uses packaged defaults.

```bash
diaad transcripts tabularize
```

To use a split configuration directory:

```bash
diaad transcripts tabularize --config config
```

The split directory normally contains:

```text
config/
  project.yaml
  advanced.yaml
```

To use a single nested YAML file:

```bash
diaad transcripts tabularize --config diaad_config.yaml
```

The nested file should use top-level `project:` and `advanced:` sections. See [Configuration](04_configuration.md) for the shared configuration model, defaults, path behavior, and validation rules.

## Common Options

The most commonly used CLI options are:

| Option | Purpose |
|---|---|
| `--config PATH` | Load configuration from a split config directory or nested YAML file. |
| `--input-dir PATH` | Override `project.input_dir` for the current run. |
| `--output-dir PATH` | Override `project.output_dir` for the current run. |
| `--set KEY=VALUE` | Override a supported configuration value from the command line. Repeat as needed. |
| `--dry-run-config` | Resolve configuration and exit without running a DIAAD command. |
| `--dry-run-config-out PATH` | Write the resolved dry-run configuration to a file. |
| `--dry-run-config-format json|yaml` | Choose the dry-run output format. |

Sectioned override keys are the clearest form:

```bash
diaad transcripts tabularize --set project.random_seed=123
diaad transcripts tabularize --set advanced.transcript_table_filename=my_transcripts.xlsx
```

Use a dry run to check the effective configuration before processing data:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

To save the resolved configuration:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml --dry-run-config-out resolved_config.yaml
```

## Input and Output Locations

By default, DIAAD reads from:

```text
diaad_data/input
```

and writes to:

```text
diaad_data/output
```

Each normal run creates a timestamped output folder under the configured output directory. Run artifacts include the resolved configuration, command-line arguments, logs, and file manifests so the run can be inspected later.

Many transcript-based commands expect a transcript table workbook named by `advanced.transcript_table_filename`, which defaults to:

```text
transcript_tables.xlsx
```

By default, DIAAD does not automatically create transcript tables when a later command needs them. Run `transcripts tabularize` first, or intentionally enable `project.auto_tabularize` in configuration.

## Available Commands

Current command modules and actions are:

| Module | Commands |
|---|---|
| `examples` | `examples` |
| `blinding` | `blinding encode`, `blinding decode` |
| `transcripts` | `transcripts select`, `transcripts evaluate`, `transcripts reselect`, `transcripts tabularize`, `transcripts chats` |
| `templates` | `templates utterances`, `templates samples`, `templates times`, `templates subset` |
| `cus` | `cus files`, `cus evaluate`, `cus reselect`, `cus analyze`, `cus rates` |
| `words` | `words files`, `words evaluate`, `words reselect`, `words analyze`, `words rates` |
| `powers` | `powers files`, `powers analyze`, `powers rates`, `powers evaluate`, `powers reselect` |
| `vocab` | `vocab file`, `vocab check`, `vocab analyze`, `vocab rates` |
| `turns` | `turns files`, `turns evaluate`, `turns reselect`, `turns analyze` |

## Generated Examples

The examples command is both a command and a workflow for creating runnable synthetic DIAAD inputs.

Generate the full example dataset:

```bash
diaad examples
```

Generate only the files relevant to one command:

```bash
diaad examples --for-command "vocab check"
```

Generate files for multiple commands:

```bash
diaad examples --for-command "transcripts tabularize" --for-command "cus files"
```

For documentation maintenance, generated Example I/O pages can be refreshed with:

```bash
diaad examples --render-docs
```

The examples-specific flags apply only to `diaad examples`.

## Typical First Runs

For a transcript-based project, begin by tabularizing transcripts:

```bash
diaad transcripts tabularize
```

Then generate coding files or analysis files from the transcript table:

```bash
diaad cus files
diaad words files
diaad vocab check
```

For a chained first pass:

```bash
diaad "transcripts tabularize, cus files, words files"
```

For workflows that require manual coding, inspect and complete the generated coding workbooks before running evaluation, analysis, or rate commands.

## Practical Recommendations

Run `diaad examples --for-command "<command>"` before using an unfamiliar command. Run `--dry-run-config` when changing configuration. Keep configuration files with the project, but handle transcript files and outputs according to the privacy requirements of the dataset. Avoid changing sample identifiers, transcript table filenames, or core ID columns in the middle of a project unless the change is intentional and documented.

## Common Problems

If a command cannot find transcript tables, run `diaad transcripts tabularize` first or check `advanced.transcript_table_filename`.

If a command name is rejected, compare it with the command table above. DIAAD uses canonical module-action names such as `transcripts tabularize`, not descriptive aliases.

If an NLP-backed command fails because spaCy or a language model is missing, install the NLP extra and the configured model:

```bash
pip install "diaad[nlp]"
python -m spacy download en_core_web_sm
```

If a run used unexpected directories, check the resolved configuration with `--dry-run-config`. Command-line `--input-dir`, `--output-dir`, and `--set` values override configuration files for the current invocation.
