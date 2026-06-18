---
object_type: command
object_types:
- command
object_id: templates.combine
command_id: templates.combine
canonical_command: templates combine
module_id: templates
title: Template Combination Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Template Combination Example

This example demonstrates how `diaad templates combine` stacks same-schema Excel workbooks and records their source files.

## Command

```bash
diaad templates combine --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      template_combination/
        site_a/
          template_batch_a.xlsx
        site_b/
          template_batch_b.xlsx
    output/
      diaad_YYMMDD_HHMM/
        coding_templates/
          combined.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input/template_combination
output_dir: diaad_data/output
```

## Input Snippet

`diaad_data/input/template_combination/site_a/template_batch_a.xlsx`

### Sheet: ratings

| sample_id | coder | engagement_code |
| --- | --- | --- |
| P1_picnic_pre | coder_a | 2 |
| P2_picnic_pre | coder_a | 1 |

### Sheet: notes

| sample_id | coder | note |
| --- | --- | --- |
| P1_picnic_pre | coder_a | clear topic shift |
| P2_picnic_pre | coder_a | brief response |

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/coding_templates/combined.xlsx`

### Sheet: ratings

| combined_id | source_file | sample_id | coder | engagement_code |
| --- | --- | --- | --- | --- |
| 1 | site_a/template_batch_a.xlsx | P1_picnic_pre | coder_a | 2 |
| 2 | site_a/template_batch_a.xlsx | P2_picnic_pre | coder_a | 1 |
| 3 | site_b/template_batch_b.xlsx | P1_picnic_post | coder_b | 3 |

### Sheet: notes

| combined_id | source_file | sample_id | coder | note |
| --- | --- | --- | --- | --- |
| 1 | site_a/template_batch_a.xlsx | P1_picnic_pre | coder_a | clear topic shift |
| 2 | site_a/template_batch_a.xlsx | P2_picnic_pre | coder_a | brief response |
| 3 | site_b/template_batch_b.xlsx | P1_picnic_post | coder_b | expanded detail |

### Sheet: metadata

| source_file | order | sheet | num_rows |
| --- | --- | --- | --- |
| site_a/template_batch_a.xlsx | 1 | ratings | 2 |
| site_a/template_batch_a.xlsx | 1 | notes | 2 |
| site_b/template_batch_b.xlsx | 2 | ratings | 1 |
| site_b/template_batch_b.xlsx | 2 | notes | 1 |

## Notes

All discovered `.xlsx` workbooks must have the same sheet names and matching columns within each sheet. The output adds `combined_id` and `source_file` columns to each combined data sheet, then writes a `metadata` sheet with source row counts.
