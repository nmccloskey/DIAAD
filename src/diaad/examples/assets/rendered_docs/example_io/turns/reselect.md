---
object_type: command
object_types:
- command
command_id: turns.reselect
canonical_command: turns reselect
module_id: turns
view: example_io
title: Conversation Turns Reliability Reselection Example
slot: examples
---

# Conversation Turns Reliability Reselection Example

This example demonstrates how `diaad turns reselect` selects replacement samples for digital conversation-turn reliability coding.

## Command

```bash
diaad turns reselect --config config
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
        reselected_turns_reliability/
          reselected_conversation_turns_reliability_template.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.34
random_seed: 99
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
```

## Input Snippet

The primary turns workbook has two synthetic sample IDs. The prior reliability workbook contains only `S001`, so reselection can choose a replacement sample.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/reselected_turns_reliability/reselected_conversation_turns_reliability_template.xlsx`

| sample_id | session | bin | turns |
| --- | --- | --- | --- |
| S002 | visit_1 | bin_1 |  |
| S002 | visit_1 | bin_2 |  |

## Notes

Reselected rows keep the session and bin structure while clearing the `turns` cells for fresh reliability coding.
