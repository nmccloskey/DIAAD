---
object_type: command
object_types:
- command
object_id: words.evaluate
command_id: words.evaluate
canonical_command: words evaluate
module_id: words
title: Word Count Reliability Evaluation Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Word Count Reliability Evaluation Example

This example demonstrates how `diaad words evaluate` compares primary word counts with a synthetic reliability workbook.

## Command

```bash
diaad words evaluate --config config
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
        word_count_reliability/
          word_count_reliability_results.xlsx
          word_count_reliability_report.txt
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

## Advanced Config

```yaml
word_count_filename: word_counting.xlsx
word_count_column: word_count
```

## Input Snippet

The command reads `diaad_data/input/word_counts/word_counting.xlsx` and `diaad_data/input/word_counts/word_count_reliability.xlsx`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/word_count_reliability/word_count_reliability_results.xlsx`

| sample_id | utterance_id | speaker | utterance | comment | id | word_count_org | wc_comment | word_count_rel | abs_diff | perc_diff | perc_sim | agmt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | U0002 | PAR | The family brought food to the park. |  | 1 | 7 |  | 6 | 1 | 15.38 | 84.62 | 1 |
| 1 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 | 8 |  | 8 | 0 | 0.0 | 100.0 | 1 |
| 1 | U0004 | PAR | Then they share sandwiches. |  | 1 | 4 |  | 4 | 0 | 0.0 | 100.0 | 1 |
| 1 | U0006 | PAR | Yes, the dog waits beside them. |  | 1 | 6 |  | 6 | 0 | 0.0 | 100.0 | 1 |
| 1 | U0007 | PAR | The day is quiet. |  | 1 | 4 |  | 3 | 1 | 28.57 | 71.43 | 1 |
| 3 | U0002 | PAR | The family is sitting on a blanket. |  | 2 | 7 |  | 8 | 1 | 13.33 | 86.67 | 1 |
| 3 | U0003 | PAR | They have sandwiches, apples, and juice. |  | 2 | 6 |  | 6 | 0 | 0.0 | 100.0 | 1 |
| 3 | U0005 | PAR | She is pouring a drink. |  | 2 | 5 |  | 5 | 0 | 0.0 | 100.0 | 1 |

`diaad_data/output/diaad_YYMMDD_HHMM/word_count_reliability/word_count_reliability_report.txt`

```text
Word Count Reliability Report

Source reliability file: word_count_reliability.xlsx

Coverage in primary coding file
--------------------------------
Samples represented: 2/3 (66.7%)
Utterances represented: 10/21 (47.6%)

Paired utterances: 10
```

## Notes

Reliability word counts are synthetic, with small deterministic differences from the primary counts.
