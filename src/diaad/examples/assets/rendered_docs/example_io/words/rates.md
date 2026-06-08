# Word Count Rate Calculation Example

This example demonstrates how `diaad words rates` combines word-count sample summaries with speaking times to calculate rates per minute.

## Command

```bash
diaad words rates --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      word_count_analysis/
        word_counting_by_sample.xlsx
      speaking_times/
        speaking_times.xlsx
    output/
      diaad_YYMMDD_HHMM/
        word_count_analysis/
          word_counting_rates.xlsx
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
wc_samples_filename: word_counting_by_sample.xlsx
speaking_time_filename: speaking_times.xlsx
speaking_time_column: speaking_time
```

## Input Snippet

The command reads `diaad_data/input/word_count_analysis/word_counting_by_sample.xlsx` and `diaad_data/input/speaking_times/speaking_times.xlsx`.

## Output Preview

`expected_outputs/words_module/words_rates/word_counting_rates.xlsx`

| sample_id | no_utt_coded | no_utt_missing | total_words | mean_words_per_utt | sd_words_per_utt | min_words_per_utt | max_words_per_utt | speaking_time | speaking_minutes | total_words_per_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | 5 | 0 | 29 | 5.8 | 1.789 | 4 | 8 | 95 | 1.583333333333333 | 18.316 |
| S003 | 5 | 0 | 22 | 4.4 | 1.673 | 2 | 6 | 102 | 1.7 | 12.941 |
| S002 | 5 | 0 | 24 | 4.8 | 1.789 | 3 | 7 | 88 | 1.466666666666667 | 16.364 |

## Notes

Speaking times are synthetic seconds added to the generated speaking-time template for this example.
