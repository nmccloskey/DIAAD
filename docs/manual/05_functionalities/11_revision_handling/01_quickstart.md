# Revision Handling Quickstart

Transcript and coding revisions are normal, but they can invalidate downstream files if they change the rows, identifiers, or text that earlier coding work depended on.

## Safest Practice

Revise transcript tables before generating manual coding files.

Recommended order:

1. Run `diaad transcripts tabularize`.
2. Review and correct transcript tables.
3. Resolve metadata mismatches.
4. Generate coding files.
5. Complete manual coding.
6. Run reliability and analysis commands.

If transcript tables change after coding files exist, decide whether affected coding files must be regenerated, recoded, or reanalyzed.

## Key Rule

Preserve identifiers when the analytic unit is the same. Create or update identifiers carefully when a new sample or utterance is truly added.

## Read Next

- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Transcript preprocessing: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/02_usage_guide.md`
- Configurable identifiers: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md`
