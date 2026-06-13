---
object_type: command
object_types:
- command
command_id: templates.subset
canonical_command: templates subset
module_id: templates
view: example_io
title: Sample Subset Example
slot: examples
---

# Sample Subset Example

This example demonstrates how `diaad templates subset` creates a randomized sample subset workbook from a general sample list.

The command has one user-facing command name. If the input `samples` sheet has only the configured sample identifier column, DIAAD runs plain subset mode. If the sheet also has a binary `exclude` column, DIAAD runs re-subset mode: it calculates the target size from all samples but selects only from samples marked `exclude == 0`.

## Plain Subset Mode

### Command

```bash
diaad templates subset --config config
```

### Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      sample_subset/
        sample_subset_input.xlsx
    output/
      diaad_YYMMDD_HHMM/
        coding_templates/
          sample_subset.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

### Basic Config

```yaml
input_dir: diaad_data/input/sample_subset
output_dir: diaad_data/output
random_seed: 99
reliability_fraction: 0.34
```

### Advanced Config

```yaml
sample_id_column: sample_id
```

### Input Snippet

`diaad_data/input/sample_subset/sample_subset_input.xlsx`

| sample_id | bin |
| --- | --- |
| P1_picnic_pre | 1 |
| P1_picnic_pre | 2 |
| P1_picnic_post | 1 |
| P2_picnic_pre | 1 |
| P3_picnic_pre | 1 |
| P3_picnic_post | 1 |

### Output Preview

`expected_outputs/templates_module/templates_subset/sample_subset.xlsx`

### Sheet: samples

| sample_id | selected | excluded |
| --- | --- | --- |
| P1_picnic_pre | 0 | 0 |
| P1_picnic_post | 0 | 0 |
| P2_picnic_pre | 0 | 0 |
| P3_picnic_pre | 1 | 0 |
| P3_picnic_post | 1 | 0 |

### Sheet: subset

| sample_id | selected | excluded |
| --- | --- | --- |
| P3_picnic_pre | 1 | 0 |
| P3_picnic_post | 1 | 0 |

## Re-Subset Mode

### Command

```bash
diaad templates subset --config config
```

### Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      sample_resubset/
        sample_resubset_input.xlsx
    output/
      diaad_YYMMDD_HHMM/
        coding_templates/
          sample_subset.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

### Basic Config

```yaml
input_dir: diaad_data/input/sample_resubset
output_dir: diaad_data/output
random_seed: 99
reliability_fraction: 0.34
```

### Input Snippet

`diaad_data/input/sample_resubset/sample_resubset_input.xlsx`

| sample_id | bin | exclude |
| --- | --- | --- |
| P1_picnic_pre | 1 | 1 |
| P1_picnic_pre | 2 | 1 |
| P1_picnic_post | 1 | 0 |
| P2_picnic_pre | 1 | 0 |
| P3_picnic_pre | 1 | 1 |
| P3_picnic_post | 1 | 0 |

### Output Preview

`expected_outputs/templates_module/templates_resubset/sample_subset.xlsx`

### Sheet: samples

| sample_id | selected | excluded |
| --- | --- | --- |
| P1_picnic_pre | 0 | 1 |
| P1_picnic_post | 0 | 0 |
| P2_picnic_pre | 1 | 0 |
| P3_picnic_pre | 0 | 1 |
| P3_picnic_post | 1 | 0 |

### Sheet: subset

| sample_id | selected | excluded |
| --- | --- | --- |
| P2_picnic_pre | 1 | 0 |
| P3_picnic_post | 1 | 0 |

## Notes

Each `templates subset` run should point `input_dir` at a folder containing exactly one `.xlsx` workbook. The output workbook always contains `samples` and `subset` sheets. The `samples` sheet records all unique sample IDs with program-generated `selected` and `excluded` columns; the `subset` sheet contains only rows where `selected == 1`.
