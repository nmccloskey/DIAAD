# `powers files` Implementation Notes

`powers files` dispatches to `make_powers_coding_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `powers files`.
2. `src/diaad/cli/dispatch.py` marks it as requiring transcript tables.
3. `src/diaad/core/run_context.py` threads metadata fields, reliability settings, excluded speakers, automation settings, blinding config, filenames, and identifier columns.
4. `src/diaad/core/run_wrappers.py` calls `make_powers_coding_files()`.
5. `src/diaad/coding/powers/files.py` writes the workbooks.

## Transcript Input

The command discovers the configured transcript table filename in the input directory or current run output directory. It extracts utterance-level transcript data and validates the configured sample and utterance identifier columns.

Transcript-table administrative columns are dropped before export. Metadata columns are preserved when they correspond to communication-oriented metadata fields such as `communication`, `topic`, `subject`, `dialogue`, or `conversation`.

## Output Files

The command writes under:

```text
powers_coding/
```

Current filenames are:

```text
powers_coding.xlsx
powers_reliability_coding.xlsx
powers_blind_codebook.xlsx
```

The blind codebook is only written when blinding is active and produces codebook rows.

## Automation Boundary

Automation is implemented in `src/diaad/coding/powers/automation.py`. It can populate `speech_units`, `filled_pauses`, `content_words`, `num_nouns`, and `tagged_utterance`.

The remaining POWERS fields, including Section E fields, are not automatically operationalized by the current code.

## Section E Boundary

The primary workbook writer creates `section_e` as a separate sheet with one row per sample. Reliability workbooks are written as a single utterance-level sheet. The current analysis and reliability paths read utterance-level coding data and do not consume the `section_e` sheet.

## Relevant Sources

- `src/diaad/coding/powers/files.py`
- `src/diaad/coding/powers/automation.py`
- `src/diaad/coding/utils/coders.py`
- `src/diaad/coding/utils/sampling.py`
- `tests/test_coding/test_powers/test_automation.py`
- `tests/test_coding/test_powers/test_identifiers.py`
