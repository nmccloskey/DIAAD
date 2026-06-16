# Metadata Extraction Usage Guide

Metadata extraction is a project-facing convenience. It helps DIAAD carry study variables from file organization into transcript tables, but it does not remove the need to inspect those values.

## When Metadata Is Extracted

Metadata extraction happens during transcript tabularization. DIAAD reads configured fields, attempts to resolve each field for each transcript, and writes the results into the `samples` sheet.

If a field cannot be resolved cleanly, DIAAD leaves the value blank, sets `metadata_mismatch` for that sample, and writes a diagnostic row to `metadata_mismatches`.

## What To Put In Metadata

Good metadata fields usually describe sample-level context, such as:

- study group;
- site;
- task or stimulus;
- session or visit;
- condition;
- cohort;
- speaker role when it is encoded in file organization.

Avoid metadata fields that duplicate unstable coding decisions. Metadata should be descriptive context, not an analysis result that will be revised during coding.

## Inspecting Results

After tabularization, check:

1. Do the expected metadata columns appear in `samples`?
2. Are blank values expected?
3. Does `metadata_mismatch` equal `1` only for samples that need review?
4. Does `metadata_mismatches` identify the source path and field that failed?
5. Are metadata values represented consistently enough for later grouping, blinding, or analysis?

Treat mismatches as a data-quality task. Fix the file naming convention, folder structure, or metadata field configuration before creating downstream coding files when possible.

## Downstream Use

Later DIAAD steps may read joined transcript-table data to recover identifiers and metadata. Metadata is especially important when:

- blinding or unblinding needs identifier context;
- a workflow needs to preserve sample groupings;
- outputs need to remain joinable to external study metadata;
- custom analysis scripts will merge DIAAD outputs with other project tables.

For Target Vocabulary Coverage, the `stimulus_column` setting is separate from `metadata_fields`. It names the column used to connect transcript samples with target-vocabulary resources. That column may be extracted as metadata, but the settings serve different purposes.

## Read Next

- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Configurable identifiers: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md`
- Blinding functionality: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
