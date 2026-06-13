---
object_type: command
object_types:
- command
object_id: words.reselect
command_id: words.reselect
canonical_command: words reselect
module_id: words
title: Word Count Reliability Reselection Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Word Count Reliability Reselection Example

This example demonstrates how `diaad words reselect` selects replacement word-count reliability rows after an earlier reliability workbook has already been used.

## Command

```bash
diaad words reselect --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      word_counts/
        word_counting.xlsx
        word_count_reliability.xlsx
    output/
      diaad_YYMMDD_HHMM/
        reselected_word_count_reliability/
          reselected_word_count_reliability.xlsx
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

The command reads prior word-count coding and reliability workbooks from `diaad_data/input/word_counts/`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/reselected_word_count_reliability/reselected_word_count_reliability.xlsx`

| sample_id | utterance_id | speaker | utterance | comment | id | word_count | wc_comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | U0001 | INV | What do you notice first? |  | 1 | 5 |  |
| 2 | U0002 | PAR | A picnic. |  | 1 | 2 |  |
| 2 | U0003 | PAR | The dad is opening the basket. |  | 1 | 6 |  |
| 2 | U0004 | PAR | The dog wants food! |  | 1 | 4 |  |
| 2 | U0005 | INV | What might happen next? |  | 1 | 4 |  |
| 2 | U0006 | PAR | They will eat lunch. |  | 1 | 4 |  |
| 2 | U0007 | PAR | Maybe the dog gets a cracker. |  | 1 | 6 |  |

## Notes

The synthetic example has only three samples, so the reselected workbook is intentionally tiny.
