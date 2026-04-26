# CU Rate Calculation Example

This example demonstrates how `diaad cus rates` combines CU sample summaries with speaking times to calculate rates per minute.

## Command

```bash
diaad cus rates --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      cu_coding_analysis/
        cu_coding_by_sample_long.xlsx
      speaking_times/
        speaking_times.xlsx
    output/
      diaad_YYMMDD_HHMM/
        cu_coding_analysis/
          cu_coding_rates.xlsx
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
cu_samples_file: cu_coding_by_sample_long.xlsx
speaking_time_file: speaking_times.xlsx
speaking_time_field: speaking_time
```

## Input Snippet

The command reads `diaad_data/input/cu_coding_analysis/cu_coding_by_sample_long.xlsx` and `diaad_data/input/speaking_times/speaking_times.xlsx`.

## Output Preview

`expected_outputs/cus_module/cus_rates/cu_coding_rates.xlsx`

| sample_id | coder | paradigm | sv_col | rel_col | cu_col | speaking_time | speaking_minutes | cu_per_min | p_sv_per_min | p_rel_per_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | primary | base | sv | rel | cu | 95 | 1.583333333333333 | 0.632 | 1.895 | 1.895 |
| S003 | primary | base | sv | rel | cu | 102 | 1.7 | 1.765 | 2.353 | 1.765 |
| S002 | primary | base | sv | rel | cu | 88 | 1.466666666666667 | 0.682 | 2.045 | 1.364 |

## Notes

Speaking times are synthetic seconds added to the generated speaking-time template for this example.
