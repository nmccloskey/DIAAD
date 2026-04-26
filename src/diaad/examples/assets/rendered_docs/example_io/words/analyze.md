# Word Count Analysis Example

This example demonstrates how `diaad words analyze` summarizes filled word-count coding by utterance and by sample.

## Command

```bash
diaad words analyze --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      word_counts/
        word_counting.xlsx
        word_count_blind_codebook.xlsx
    output/
      diaad_YYMMDD_HHMM/
        word_count_analysis/
          word_counting_by_utterance.xlsx
          word_counting_by_sample.xlsx
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
word_count_file: word_counting.xlsx
word_count_field: word_count
metadata_source: transcript_tables
coding_blind_cols:
- sample_id
id_cols:
- sample_id
- utterance_id
```

## Input Snippet

The command reads `diaad_data/input/word_counts/word_counting.xlsx`. The blind codebook is included so analysis outputs can recover sample identifiers.

## Output Preview

`expected_outputs/words_module/words_analyze/word_counting_by_utterance.xlsx`

| sample_id | utterance_id | speaker | utterance | word_count |
| --- | --- | --- | --- | --- |
| S001 | U0001 | INV | Please tell the picnic story again. |  |
| S001 | U0002 | PAR | The family brought food to the park. | 7.0 |
| S001 | U0003 | PAR | The little girl [/] the little girl pours juice. | 8.0 |
| S001 | U0004 | PAR | Then they share sandwiches. | 4.0 |
| S001 | U0005 | INV | Anything else? |  |
| S001 | U0006 | PAR | Yes, the dog waits beside them. | 6.0 |
| S001 | U0007 | PAR | The day is quiet. | 4.0 |
| S003 | U0001 | INV | What do you notice first? |  |

`expected_outputs/words_module/words_analyze/word_counting_by_sample.xlsx`

| sample_id | no_utt_coded | no_utt_missing | total_words | mean_words_per_utt | sd_words_per_utt | min_words_per_utt | max_words_per_utt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | 5 | 2 | 29 | 5.8 | 1.789 | 4 | 8 |
| S003 | 5 | 2 | 22 | 4.4 | 1.673 | 2 | 6 |
| S002 | 5 | 2 | 24 | 4.8 | 1.789 | 3 | 7 |

## Notes

The preview uses synthetic filled word counts generated from the packaged example specs.
