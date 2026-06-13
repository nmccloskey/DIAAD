---
object_type: command
object_types:
- command
object_id: templates.times
command_id: templates.times
canonical_command: templates times
module_id: templates
title: Speaking-Time Template Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Speaking-Time Template Example

This example demonstrates how `diaad templates times` creates a blank sample-level speaking-time workbook.

## Command

```bash
diaad templates times --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      transcript_tables/
        transcript_tables.xlsx
    output/
      diaad_YYMMDD_HHMM/
        coding_templates/
          speaking_times.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/coding_templates/speaking_times.xlsx`

### Sheet: coding_template

| sample_id | speaking_time |
| --- | --- |
| S001 |  |
| S002 |  |
| S003 |  |

## Notes

The `speaking_time` column is intentionally blank. It is a template for project-specific duration values used later by rate calculations.
