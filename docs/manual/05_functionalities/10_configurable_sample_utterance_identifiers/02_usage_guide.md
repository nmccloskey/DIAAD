# Configurable Sample and Utterance Identifiers Usage Guide

Identifier columns are a cross-workflow contract. Once transcript tables and coding files exist, changing the identifier convention can break joins and invalidate already-created manual files.

## Recommended Practice

For new projects:

```yaml
advanced:
  sample_id_column: sample_id
  utterance_id_column: utterance_id
```

Keep these defaults through tabularization, coding-file generation, reliability selection, analysis, blinding, and export unless there is a strong reason to do otherwise.

## Understanding The Pair

In generated transcript tables, `sample_id` identifies the transcript or sample. `utterance_id` identifies an utterance within that sample. Utterance IDs may repeat across samples because numbering restarts within each transcript.

For utterance-level work, use the pair:

```text
sample_id + utterance_id
```

This is especially important for Complete Utterances, Word Counting, POWERS, Target Vocabulary Coverage, blinding, and reliability evaluation outputs that need to rejoin utterance-level rows.

## Changing Identifier Column Names

If you need custom names, configure them before creating data artifacts:

```yaml
advanced:
  sample_id_column: expanded_sample_id
  utterance_id_column: expanded_utterance_id
```

Then run a dry-run config check and make sure all input workbooks use the same column names.

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

Do not change identifier column names midway through a coding project unless you are prepared to regenerate or update downstream files.

## Relationship To Blinding

Blinding settings use their own identifier-related configuration, including `advanced.id_columns` and `advanced.blind_columns`. These should align with the sample and utterance identifier convention in the files being encoded or decoded.

For most projects, keep `id_columns` consistent with the identifiers that define a record in the files being blinded.

## Read Next

- Blinding functionality: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
- Metadata extraction: `docs/manual/05_functionalities/08_metadata_extraction/02_usage_guide.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
