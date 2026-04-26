# Speaking-Time Template Example

This example demonstrates how `diaad templates times` creates a blank sample-level speaking-time workbook.

## Command

```bash
diaad templates times --config config
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
        templates_times/
          speaking_times.xlsx
```

## Basic Config

```yaml
input_dir: input
output_dir: output
```

## Output Preview

`expected_outputs/templates_module/templates_times/speaking_times.xlsx`

### Sheet: coding_template

| sample_id | speaking_time |
| --- | --- |
| S001 |  |
| S002 |  |
| S003 |  |

## Notes

The `speaking_time` column is intentionally blank. It is a template for project-specific duration values used later by rate calculations.
