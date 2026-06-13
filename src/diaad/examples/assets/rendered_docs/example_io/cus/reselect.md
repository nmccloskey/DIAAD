---
object_type: command
object_types:
- command
object_id: cus.reselect
command_id: cus.reselect
canonical_command: cus reselect
module_id: cus
title: CU Reliability Reselection Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# CU Reliability Reselection Example

This example demonstrates how `diaad cus reselect` selects replacement CU reliability rows after an earlier reliability workbook has already been used.

## Command

```bash
diaad cus reselect --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      cu_coding/
        cu_coding.xlsx
        cu_reliability_coding.xlsx
    output/
      diaad_YYMMDD_HHMM/
        reselected_cu_coding_reliability/
          reselected_cu_reliability_coding.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.34
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
```

## Input Snippet

The command reads prior CU coding and reliability workbooks from `diaad_data/input/cu_coding/`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/reselected_cu_coding_reliability/reselected_cu_reliability_coding.xlsx`

| sample_id | input_order | shuffled_order | stimulus | utterance_id | position | position_sub | speaker | utterance | comment | id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 3 |  | picnic | U0001 | 1 | 0 | INV | What do you notice first? |  | 1 |  |  |
| 2 | 3 |  | picnic | U0002 | 2 | 0 | PAR | A picnic. |  | 1 | 1.0 | 1.0 |
| 2 | 3 |  | picnic | U0003 | 3 | 0 | PAR | The dad is opening the basket. |  | 1 | 1.0 | 0.0 |
| 2 | 3 |  | picnic | U0004 | 4 | 0 | PAR | The dog wants food! |  | 1 | 0.0 | 0.0 |
| 2 | 3 |  | picnic | U0005 | 5 | 0 | INV | What might happen next? |  | 1 |  |  |
| 2 | 3 |  | picnic | U0006 | 6 | 0 | PAR | They will eat lunch. |  | 1 | 1.0 | 1.0 |
| 2 | 3 |  | picnic | U0007 | 7 | 0 | PAR | Maybe the dog gets a cracker. |  | 1 | 1.0 | 1.0 |

## Notes

The synthetic example has only three samples, so the reselected workbook is intentionally tiny.
