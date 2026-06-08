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
word_count_filename: word_counting.xlsx
word_count_column: word_count
auto_blind: true
blind_columns:
- sample_id
metadata_source: transcript_tables
codebook_filename: ''
```

## Input Snippet

The command reads `diaad_data/input/word_counts/word_counting.xlsx`. The blind codebook is included so analysis outputs can recover sample identifiers.

## Output Preview

`expected_outputs/words_module/words_analyze/word_counting_by_utterance.xlsx`

| utterance_id | speaker | utterance | word_count | sample_id_blinded |
| --- | --- | --- | --- | --- |
| U0002 | PAR | The family brought food to the park. | 7 | 1 |
| U0003 | PAR | The little girl [/] the little girl pours juice. | 8 | 1 |
| U0004 | PAR | Then they share sandwiches. | 4 | 1 |
| U0006 | PAR | Yes, the dog waits beside them. | 6 | 1 |
| U0007 | PAR | The day is quiet. | 4 | 1 |
| U0002 | PAR | A picnic. | 2 | 2 |
| U0003 | PAR | The dad is opening the basket. | 6 | 2 |
| U0004 | PAR | The dog wants food! | 4 | 2 |

`expected_outputs/words_module/words_analyze/word_counting_by_sample.xlsx`

| no_utt_coded | no_utt_missing | total_words | mean_words_per_utt | sd_words_per_utt | min_words_per_utt | max_words_per_utt | sample_id_blinded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.0 | 0.0 | 29.0 | 5.8 | 1.789 | 4.0 | 8.0 | 1.0 |
| 5.0 | 0.0 | 22.0 | 4.4 | 1.673 | 2.0 | 6.0 | 2.0 |
| 5.0 | 0.0 | 24.0 | 4.8 | 1.789 | 3.0 | 7.0 | 3.0 |

## Notes

The preview uses synthetic filled word counts generated from the packaged example specs.
