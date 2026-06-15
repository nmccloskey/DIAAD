# `turns reselect` Implementation Notes

`turns reselect` dispatches to `run_reselect_digital_convo_turns()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `turns reselect`.
2. `src/diaad/cli/dispatch.py` dispatches the command without a transcript-table prerequisite.
3. `src/diaad/core/run_context.py` passes metadata fields, input/output paths, reliability fraction, random seed, and sample identifier.
4. `src/diaad/core/run_wrappers.py` calls `reselect_digital_convo_turns_rel()`.
5. `src/diaad/coding/convo_turns/rel_reselection.py` discovers pairs and writes reselected reliability workbooks.

## Discovery And Fallback

The command calls shared reliability-pair discovery with:

```text
coding_glob = *conversation_turns_template.xlsx
rel_glob = *conversation_turns_reliability*.xlsx
rel_label = TURNS
```

If discovery finds no reliability mates, the DCT wrapper falls back to filename-based pairing. In fallback mode, it requires one primary `conversation_turns_template.xlsx` match and pairs it with all matching reliability files.

## Reselection Logic

Shared helpers load the primary and reliability workbooks, collect used sample identifiers from prior reliability files, select unused identifiers according to the configured reliability fraction, and write the reselected workbook.

The DCT-specific frame builder selects all primary rows for the newly selected sample identifiers and clears `turns`.

## Output Path

Outputs are written under:

```text
reselected_turns_reliability/
```

The default output file for the standard primary filename is:

```text
reselected_conversation_turns_reliability_template.xlsx
```

## Relevant Sources

- `src/diaad/coding/convo_turns/rel_reselection.py`
- `src/diaad/coding/utils/reselection_utils.py`
- `src/diaad/core/run_context.py`
- `tests/test_coding/test_convo_turns/test_identifiers.py`
