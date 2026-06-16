# CLI Project Setup and First Run Quickstart

Use the command-line workflow when you want local control over files, repeatable runs, scripting, audit artifacts, or sensitive data handling. It is the best starting point for research projects that will eventually become a repeated pipeline.

## Start A Project Folder

Create a project folder with configuration, input, and output areas:

```text
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
    output/
```

Run DIAAD from `your_project/`. If `config/` is present, the CLI uses it automatically.

## First Check

Inspect the effective configuration before processing files:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

For a hands-on first run with synthetic files:

```bash
diaad examples
```

For a smaller package focused on one command:

```bash
diaad examples --for-command "transcripts tabularize"
```

## First Real Run

For a transcript-based project, put CHAT files under the configured input directory and start with:

```bash
diaad transcripts tabularize
```

Later commands can use the generated transcript table:

```bash
diaad cus files
diaad words files
diaad powers files
diaad vocab check
```

Use chained commands only when no human review is needed between them:

```bash
diaad "transcripts tabularize, cus files"
```

## Blinding Checkpoint

For manual coding workflows, decide whether identifiers should be encoded before coder-facing workbooks are distributed. The ideal pattern is often encode before manual coding, decode back to original sample identifiers before DIAAD analysis, and optionally re-encode exported analysis tables for blinded statistical workflows. This is a project decision, not an absolute requirement.

## Read Next

- Installation: `docs/manual/02_operation/01_installation.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Run provenance: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/01_quickstart.md`
- Example workflow: `docs/manual/06_workflows/03_example_dataset_command_specific_packages/01_quickstart.md`
