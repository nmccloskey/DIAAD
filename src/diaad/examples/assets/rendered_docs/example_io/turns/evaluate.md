---
object_type: command
object_types:
- command
command_id: turns.evaluate
canonical_command: turns evaluate
module_id: turns
view: example_io
title: Conversation Turns Reliability Evaluation Example
slot: examples
---

# Conversation Turns Reliability Evaluation Example

This example demonstrates how `diaad turns evaluate` compares primary and reliability-coded digital conversation turns.

## Command

```bash
diaad turns evaluate --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      conversation_turns/
        conversation_turns_template.xlsx
        conversation_turns_reliability_template.xlsx
    output/
      diaad_YYMMDD_HHMM/
        turns_reliability/
          conversation_turns_reliability_results.xlsx
          conversation_turns_reliability_report.txt
          global_alignments/
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
```

## Input Snippet

`diaad_data/input/conversation_turns/conversation_turns_template.xlsx`

| sample_id | session | bin | turns |
| --- | --- | --- | --- |
| S001 | visit_1 | bin_1 | 0.1..23.0.12 |
| S001 | visit_1 | bin_2 | 1.0..32.10. |
| S001 | visit_2 | bin_1 | 0.12.3..01 |
| S001 | visit_2 | bin_2 | 2.30.1..20 |
| S002 | visit_1 | bin_1 | 0.13..20.1 |
| S002 | visit_1 | bin_2 | 3.0..12.30 |

`diaad_data/input/conversation_turns/conversation_turns_reliability_template.xlsx`

| sample_id | session | bin | turns |
| --- | --- | --- | --- |
| S001 | visit_1 | bin_1 | 0.1..23.01 |
| S001 | visit_1 | bin_2 | 1.0..32.10. |
| S001 | visit_2 | bin_1 | 0.12.3..01 |
| S001 | visit_2 | bin_2 | 2.30.1..2 |

## Output Preview

`expected_outputs/turns_module/turns_evaluate/conversation_turns_reliability_results.xlsx`

### Sheet: counts

| sample_id | session | bin | participant | count_main | count_rel | perc_agmt |
| --- | --- | --- | --- | --- | --- | --- |
| S001 | visit_1 | bin_1 | 0 | 2 | 2 | 100 |
| S001 | visit_1 | bin_1 | 1 | 2 | 2 | 100 |
| S001 | visit_1 | bin_1 | 2 | 2 | 1 | 50 |
| S001 | visit_1 | bin_1 | 3 | 1 | 1 | 100 |
| S001 | visit_1 | bin_2 | 0 | 2 | 2 | 100 |
| S001 | visit_1 | bin_2 | 1 | 2 | 2 | 100 |
| S001 | visit_1 | bin_2 | 2 | 1 | 1 | 100 |
| S001 | visit_1 | bin_2 | 3 | 1 | 1 | 100 |

### Sheet: sequences

| sample_id | session | bin | levenshtein_distance | levenshtein_similarity |
| --- | --- | --- | --- | --- |
| S001 | visit_1 | bin_1 | 2 | 0.8333333333333334 |
| S001 | visit_1 | bin_2 | 0 | 1.0 |
| S001 | visit_2 | bin_1 | 0 | 1.0 |
| S001 | visit_2 | bin_2 | 1 | 0.9 |
| S002 | visit_1 | bin_1 | 10 | 0.0 |
| S002 | visit_1 | bin_2 | 10 | 0.0 |

### Sheet: samples

| sample_id | avg_perc_agmt | avg_dist | avg_sim |
| --- | --- | --- | --- |
| S001 | 93.75 | 0.75 | 0.9333333333333333 |
| S002 | 0.0 | 10.0 | 0.0 |

`expected_outputs/turns_module/turns_evaluate/conversation_turns_reliability_report.txt`

```text
Digital Conversation Turns Reliability Report

Source reliability file: conversation_turns_reliability_template.xlsx

Coverage in primary coding file
--------------------------------
Samples represented: 2/2 (100.0%)
Session-bin rows represented: 6/6 (100.0%)

Primary reliability metrics
---------------------------

Count totals by participant within sample/session/bin
Paired participant targets: 24
```

## Notes

The synthetic turn strings use four speakers (`0`, `1`, `2`, and `3`) and include both mark1 (`.`) and mark2 (`..`) examples.
