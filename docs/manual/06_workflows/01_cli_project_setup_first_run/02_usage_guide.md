# CLI Project Setup and First Run Usage Guide

The CLI workflow is organized around a project root. The project root is the folder where DIAAD resolves relative paths, discovers `config/`, and writes timestamped outputs.

## Choose The CLI When

The CLI is a good fit when:

- data are sensitive or hard to deidentify;
- repeated processing is expected across batches, sites, sessions, or timepoints;
- a Python, PowerShell, shell, or subprocess-based workflow will call DIAAD commands;
- outputs need to stay in a controlled local directory;
- staged manual review is part of the workflow;
- complete run logs and audit artifacts matter.

The CLI can still be used for exploration. Generated examples are the safest first run.

## Build The Project Skeleton

A typical project starts as:

```text
your_project/
  config/
  diaad_data/
    input/
    output/
```

For real work, use split configuration files:

```text
config/
  project.yaml
  advanced.yaml
```

Keep study-facing settings such as input paths, output paths, reliability fraction, and metadata fields readable in `project.yaml`. Treat `advanced.yaml` more cautiously because it defines filenames, identifier columns, and workflow-specific defaults used downstream.

## Learn With Examples First

Before processing study data, generate a command-specific example for the command you plan to run:

```bash
diaad examples --for-command "transcripts tabularize"
```

Inspect the example README, `example_config/`, `example_input/`, and `example_output/`. Then compare your project folder to the example package.

The full example dataset is useful when you want an atlas of DIAAD behavior:

```bash
diaad examples
```

It is comprehensive, but it does not look exactly like one normal project output tree.

## Run A Configuration Dry Run

Before running a command on real files, inspect the resolved configuration:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

Save the resolved configuration when it will be part of an audit trail:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml --dry-run-config-out resolved_config.yaml
```

## Stage Commands Around Human Work

Some DIAAD commands can be safely chained. For example, transcript tabularization can feed immediate coding-file generation:

```bash
diaad "transcripts tabularize, cus files"
```

Do not chain through manual review points. If a command generates a coding workbook, complete and review that workbook before running evaluation, analysis, or rates.

## Manage Identifiers And Blinding

For workflows with coder-facing files, decide early whether to encode configured identifiers before manual coding. Blinding may be useful for reducing bias, but it is not the same as full de-identification and may not be practically effective when coders already know the samples.

When a project does use encoded identifiers, the clean pattern is:

1. encode before manual coding;
2. decode back to original sample identifiers before DIAAD analysis;
3. re-encode downstream exports if blinded statistical workflows are preferred.

## Preserve The Run Directory

Each normal CLI run writes a timestamped output directory. Preserve it with:

- `logs/`;
- substantive output files;
- the project configuration used for the run;
- any manually edited workbooks that become inputs to later runs.

The log artifacts make it easier to reconstruct a run after command-line overrides, reliability sampling, or blinding decisions.

## Read Next

- CLI and web execution: `docs/manual/05_functionalities/02_cli_web_execution/02_usage_guide.md`
- Configuration sources and overrides: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/02_usage_guide.md`
- Blinding functionality: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
