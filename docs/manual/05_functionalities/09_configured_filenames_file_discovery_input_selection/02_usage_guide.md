# Configured Filenames, File Discovery, and Input Selection Usage Guide

This page summarizes the cross-cutting behavior behind DIAAD's file selection. The main conceptual explanation lives in the Exact File Name Matching feature page.

## How DIAAD Selects Files

Many commands start from a configured input filename. For example, transcript-table commands look for `advanced.transcript_table_filename`, which defaults to:

```text
transcript_tables.xlsx
```

When DIAAD needs exactly one file, it recursively searches the relevant directories and requires a single match. If no match exists, it raises an error. If multiple matches exist, it raises an error and lists the matched paths.

This behavior is stricter than "use the newest file" or "use the first file found." It protects users from accidentally analyzing the wrong workbook.

## Input Directory Habits

Use the input directory as a staging area for the next run:

- include the exact files that command needs;
- keep older outputs elsewhere unless they are the intended inputs;
- avoid multiple copies of the same configured workbook name;
- preserve filenames when moving reviewed outputs forward;
- use command-specific subdirectories only when the command documentation indicates they are expected.

For transcript-table workflows, a reviewed table is often placed under:

```text
diaad_data/input/transcript_tables/transcript_tables.xlsx
```

## Configuring Filenames

Change configured filenames only when your project has a durable reason, such as an institutional naming convention or an existing dataset with stable workbook names.

After changing filenames, run a dry-run config check:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

Then confirm that downstream command pages name the same configured setting.

## Known Variations

Not every command searches in exactly the same way:

- CHAT discovery recursively reads `.cha` files rather than requiring one file.
- Some workbook-based tools accept exactly one file by extension.
- Target Vocabulary Coverage has fallback logic for unblinded utterance data versus transcript tables.
- Custom Target Vocabulary Coverage resources use the configured resource path.

The shared principle remains the same: DIAAD should not silently choose among ambiguous candidate files.

## Read Next

- Exact file name matching feature: `docs/manual/03_features/03_exact_file_name_matching.md`
- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
- Transcript preprocessing: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/02_usage_guide.md`
