# Sample Template Example

This example demonstrates how `diaad templates samples` creates blank sample-level coding workbooks with bins, coder assignment, and reliability rows.

## Command

```bash
diaad templates samples --config config
```

## Project Files

```
example_files/
  synthetic_project/
    README.md
    config/
      project.yaml
      advanced.yaml
    input/
      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
        reliability/
          P1_picnic_pre.cha
          P2_picnic_pre.cha
      transcription_reliability_selection/
        transcription_reliability_samples.xlsx
    expected_outputs/
      templates_module/
        templates_samples/
          sample_coding_template.xlsx
          sample_reliability_template.xlsx
          sample_template_codebook.xlsx
```

## Basic Config

```yaml
input_dir: input
output_dir: output
reliability_fraction: 0.34
num_bins: 2
num_coders: 2
stimulus_field: stimulus
```

## Output Preview

`expected_outputs/templates_module/templates_samples/sample_coding_template.xlsx`

### Sheet: coding_template

| sample_id | coder_id | stimulus | bin |
| --- | --- | --- | --- |
| 1 | 1 | picnic | 1 |
| 1 | 1 | picnic | 2 |
| 2 | 2 | picnic | 1 |
| 2 | 2 | picnic | 2 |
| 3 | 1 | picnic | 1 |
| 3 | 1 | picnic | 2 |

`expected_outputs/templates_module/templates_samples/sample_reliability_template.xlsx`

### Sheet: coding_template

| sample_id | coder_id | stimulus | bin |
| --- | --- | --- | --- |
| 1 | 2 | picnic | 1 |
| 1 | 2 | picnic | 2 |
| 2 | 1 | picnic | 1 |
| 2 | 1 | picnic | 2 |

`expected_outputs/templates_module/templates_samples/sample_template_codebook.xlsx`

### Sheet: Sheet1

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Notes

The example uses two bins and two coders so the assignment and reliability-subset behavior is visible in a tiny workbook.
