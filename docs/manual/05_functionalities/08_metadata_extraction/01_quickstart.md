# Metadata Extraction Quickstart

DIAAD can extract project-specific metadata from transcript paths and file names during transcript tabularization. Extracted metadata becomes columns in the transcript table and can be reused by later coding, blinding, unblinding, and analysis steps.

## Configure Metadata Fields

Metadata fields live under `project.metadata_fields`:

```yaml
project:
  metadata_fields:
    group:
      - control
      - treatment
    session: "Session:\\s*(\\w+)"
```

Run transcript tabularization after configuring the fields:

```bash
diaad transcripts tabularize
```

Then inspect:

- the metadata columns in the `samples` sheet;
- the `metadata_mismatch` column;
- the `metadata_mismatches` sheet.

## Safest Default

Start with no metadata fields, or with a small set of high-confidence fields. Add fields gradually and rerun tabularization until the mismatch diagnostics match your project expectations.

## Read Next

- Configuration: `docs/manual/02_operation/04_configuration.md`
- Transcript preprocessing: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/01_quickstart.md`
- Blinding module: `docs/manual/04_modules/08_blinding/`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
