# Word Count File Example

This example demonstrates how `diaad words files` creates word-count coding and reliability workbooks from transcript tables.

## Command

```bash
diaad words files --config config
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
        word_counts/
          word_counting.xlsx
          word_count_reliability.xlsx
          word_count_blind_codebook.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.34
num_coders: 2
stimulus_column: stimulus
exclude_participants:
- INV
```

## Advanced Config

```yaml
word_count_filename: word_counting.xlsx
word_count_column: word_count
metadata_source: transcript_tables
auto_blind: true
blind_columns:
- sample_id
```

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`expected_outputs/words_module/words_files/word_counting.xlsx`

| sample_id | utterance_id | speaker | utterance | comment | id | word_count | wc_comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | U0001 | INV | Please tell the picnic story again. |  | 1 |  |  |
| 1 | U0002 | PAR | The family brought food to the park. |  | 1 | 7.0 |  |
| 1 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 | 8.0 |  |
| 1 | U0004 | PAR | Then they share sandwiches. |  | 1 | 4.0 |  |
| 1 | U0005 | INV | Anything else? |  | 1 |  |  |
| 1 | U0006 | PAR | Yes, the dog waits beside them. |  | 1 | 6.0 |  |
| 1 | U0007 | PAR | The day is quiet. |  | 1 | 4.0 |  |
| 2 | U0001 | INV | What do you notice first? |  | 1 |  |  |

`expected_outputs/words_module/words_files/word_count_reliability.xlsx`

| sample_id | utterance_id | speaker | utterance | comment | id | word_count | wc_comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | U0001 | INV | Please tell the picnic story again. |  | 2 |  |  |
| 1 | U0002 | PAR | The family brought food to the park. |  | 2 | 6.0 |  |
| 1 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 2 | 8.0 |  |
| 1 | U0004 | PAR | Then they share sandwiches. |  | 2 | 4.0 |  |
| 1 | U0005 | INV | Anything else? |  | 2 |  |  |
| 1 | U0006 | PAR | Yes, the dog waits beside them. |  | 2 | 6.0 |  |
| 1 | U0007 | PAR | The day is quiet. |  | 2 | 3.0 |  |
| 3 | U0001 | INV | Tell me what is happening in the picnic picture. |  | 1 |  |  |

`expected_outputs/words_module/words_files/word_count_blind_codebook.xlsx`

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Notes

The generated local example fills synthetic word counts into the blank coding workbooks so downstream word-count examples can be demonstrated. Real `words files` output starts as coding material for human review.
