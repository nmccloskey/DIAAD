---
object_type: command
object_types:
- command
command_id: powers.rates
canonical_command: powers rates
module_id: powers
view: example_io
title: POWERS Rate Calculation Example
slot: examples
---

# POWERS Rate Calculation Example

This example demonstrates how `diaad powers rates` combines POWERS dialog summaries with speaking times to calculate rates per minute.

## Command

```bash
diaad powers rates --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      powers_coding_analysis/
        powers_analysis.xlsx
      speaking_times/
        speaking_times.xlsx
    output/
      diaad_YYMMDD_HHMM/
        powers_coding_analysis/
          powers_coding_rates.xlsx
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
speaking_time_filename: speaking_times.xlsx
speaking_time_column: speaking_time
```

## Input Snippet

The command reads `diaad_data/input/powers_coding_analysis/powers_analysis.xlsx` and `diaad_data/input/speaking_times/speaking_times.xlsx`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/powers_coding_analysis/powers_coding_rates.xlsx`

| sample_id | source_file | speech_units_sum | content_words_sum | num_nouns_sum | filled_pauses_sum | circumlocutions_sum | sem_paras_sum | phon_errs_sum | neologisms_sum | lg_pauses_sum | num_repairs | speaking_time | speaking_minutes | speech_units_sum_per_min | content_words_sum_per_min | num_nouns_sum_per_min | filled_pauses_sum_per_min | circumlocutions_sum_per_min | sem_paras_sum_per_min | phon_errs_sum_per_min | neologisms_sum_per_min | lg_pauses_sum_per_min | num_repairs_per_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | powers_analysis.xlsx | 10 | 36 | 15 | 10 | 13 | 16 | 10 | 18 | 13 | 1 | 95 | 1.583333333333333 | 6.316 | 22.737 | 9.474 | 6.316 | 8.211 | 10.105 | 6.316 | 11.368 | 8.211 | 0.632 |
| S002 | powers_analysis.xlsx | 10 | 36 | 15 | 10 | 15 | 18 | 10 | 21 | 15 | 1 | 88 | 1.466666666666667 | 6.818 | 24.545 | 10.227 | 6.818 | 10.227 | 12.273 | 6.818 | 14.318 | 10.227 | 0.682 |
| S003 | powers_analysis.xlsx | 11 | 29 | 14 | 11 | 14 | 17 | 11 | 22 | 14 | 1 | 102 | 1.7 | 6.471 | 17.059 | 8.235 | 6.471 | 8.235 | 10.0 | 6.471 | 12.941 | 8.235 | 0.588 |

## Notes

Speaking times are synthetic seconds added to the generated speaking-time template for this example.
