# Templates Implementation Notes

The Templates module is implemented under `src/diaad/coding/templates/`.

## Shared Utilities

Template commands share utilities for:

- finding transcript tables;
- validating required columns;
- assigning coder IDs;
- expanding sample rows by bins or coders;
- selecting reliability subsets;
- optionally applying configured blinding;
- writing workbook exports.

Generic template outputs are written under `coding_templates/`.

## Utterance And Sample Templates

`templates utterances` builds from transcript-table utterance rows. `templates samples` builds from sample rows and can expand samples into bins using `project.num_bins`.

Both commands can create primary and reliability workbooks. They can also write a codebook when blinding or coder assignment makes that useful for later interpretation.

## Speaking-Time Templates

`templates times` writes `speaking_times.xlsx`. The workbook is a place for users to enter time values; rate commands later read the configured speaking-time filename and column.

## Sample Subsets

`templates subset` finds exactly one eligible Excel workbook by extension in the input directory. The workbook must contain a `samples` sheet and the configured sample identifier column. The optional `exclude` column marks rows that should not be selected.

The output workbook, `sample_subset.xlsx`, contains `samples` and `subset` sheets with selected/excluded status columns.

## Template Combination

`templates combine` finds all eligible Excel workbooks recursively under the configured input directory. It requires all workbooks to have the same sheet names and matching columns within each sheet.

The output workbook, `combined.xlsx`, mirrors the input sheet names, adds `combined_id` and `source_file` to each combined sheet, and writes a `metadata` sheet with row counts by source workbook and sheet.

## Boundaries

Template files are intended as starting points for human work. They do not validate a custom coding scheme and do not automatically analyze arbitrary codes unless a later command or external workflow is designed for that purpose.
