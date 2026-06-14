# `templates utterances` Implementation Notes

`templates utterances` dispatches to `make_utterance_template_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `templates utterances`.
2. `src/diaad/cli/dispatch.py` marks it as requiring transcript tables.
3. `src/diaad/core/run_context.py` threads template settings, identifier columns, blinding config, and transcript table filename.
4. `src/diaad/core/run_wrappers.py` calls `make_utterance_template_files()`.
5. `src/diaad/coding/templates/utterances.py` builds the primary and reliability workbooks.

## Data Preparation

The command reads both `samples` and `utterances` from the transcript table. It requires the configured sample identifier, configured utterance identifier, and `utterance` columns in the utterance-level data.

If a configured stimulus column is available on the sample sheet, it is joined into the output and renamed to `stimulus`.

## Output Files

The implementation writes under:

```text
coding_templates/
```

Output filenames are currently fixed by the Templates module:

```text
utterance_coding_template.xlsx
utterance_reliability_template.xlsx
utterance_template_codebook.xlsx
```

The workbook sheet name is `coding_template`.

## Reliability And Blinding

Reliability subset size uses the shared `calc_subset_size()` helper. Coder assignment and reliability sampling use Python's random generator, which is seeded from `project.random_seed` by the run context.

Blinding is applied only when the blinding config reports that coding-mode blinding should run. In current configuration terms, that means `advanced.auto_blind` is true and at least one configured blind column is available.

## Relevant Sources

- `src/diaad/coding/templates/utterances.py`
- `src/diaad/coding/templates/utils.py`
- `src/diaad/coding/utils/coders.py`
- `src/diaad/coding/utils/sampling.py`
- `tests/test_coding/test_templates/test_utils.py`
- `tests/test_coding/test_templates/test_identifiers.py`
