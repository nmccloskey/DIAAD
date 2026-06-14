# `transcripts tabularize` Implementation Notes

`transcripts tabularize` is registered as the canonical command `transcripts tabularize` and dispatches through `run_tabularize_transcripts()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `transcripts tabularize`.
2. `src/diaad/cli/dispatch.py` marks it as a CHAT-required command and dispatches it.
3. `src/diaad/core/run_context.py` loads CHAT files with `read_cha_files()`.
4. `src/diaad/core/run_wrappers.py` calls `tabularize_transcripts()`.
5. `src/diaad/transcripts/transcript_tables.py` writes the workbook.

## CHAT Discovery

CHAT files are read recursively from `project.input_dir` by `src/diaad/transcripts/cha_files.py`. The dictionary keys are input-relative paths, which preserve enough path context for metadata extraction and later tracing.

When the run context loads ordinary CHAT inputs, it excludes directories named by `advanced.reliability_dirname`. The default is `reliability`.

## Workbook Structure

The implementation writes the configured transcript table filename under:

```text
transcript_tables/
```

The default workbook is:

```text
transcript_tables/transcript_tables.xlsx
```

It contains `samples`, `utterances`, and `metadata_mismatches`. The reserved output column `metadata_mismatch` cannot also be used as a configured metadata field name.

## Identifier Behavior

Sample IDs use the configured sample identifier column, defaulting to `sample_id`. Utterance IDs use the configured utterance identifier column, defaulting to `utterance_id`.

Sample IDs are assigned from sorted file order unless `project.shuffle_samples` is enabled. When shuffling is enabled, the run context seeds Python and NumPy random generators from `project.random_seed` before dispatch.

Utterance IDs reset within each sample. Identifier padding expands for larger projects.

## Relevant Sources

- `src/diaad/transcripts/transcript_tables.py`
- `src/diaad/transcripts/cha_files.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `src/diaad/cli/dispatch.py`
- `tests/test_transcripts/test_transcript_tables.py`
- `tests/test_transcripts/test_cha_files.py`
