# POWERS Coding Analysis Example

This example demonstrates how `diaad powers analyze` summarizes filled POWERS coding by utterance, turn, speaker, and dialog.

## Command

```bash
diaad powers analyze --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      powers_coding/
        powers_coding.xlsx
    output/
      diaad_YYMMDD_HHMM/
        powers_coding_analysis/
          powers_analysis.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

## Input Snippet

The command reads `diaad_data/input/powers_coding/powers_coding.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_analyze/powers_analysis.xlsx`

### Sheet: Utterances

| sample_id | utterance_id | speaker | utterance | comment | coder_id | POWERS_comment | speech_units | turn_type | turn_label | content_words | num_nouns | circumlocutions | sem_paras | phon_errs | neologisms | comments | lg_pauses | filled_pauses | collab_repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | U0001 | INV | Please tell the picnic story again. |  | 1 |  | 1 | T | T1 | 5 | 2 | 1 | 1 | 1 | 1 |  | 1 | 1 |  |
| S001 | U0002 | PAR | The family brought food to the park. |  | 1 |  | 2 | T | T2 | 7 | 3 | 2 | 2 | 2 | 2 |  | 2 | 2 |  |
| S001 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 |  | 1 | ST | ST1 | 9 | 3 | 3 | 3 | 1 | 3 |  | 3 | 1 | repair_1 |
| S001 | U0004 | PAR | Then they share sandwiches. |  | 1 |  | 2 | MT | MT1 | 4 | 2 | 1 | 4 | 2 | 4 |  | 1 | 2 |  |
| S001 | U0005 | INV | Anything else? |  | 1 |  | 1 | T | T3 | 1 | 0 | 2 | 1 | 1 | 5 |  | 2 | 1 |  |
| S001 | U0006 | PAR | Yes, the dog waits beside them. |  | 1 |  | 2 | NV | NV1 | 6 | 3 | 3 | 2 | 2 | 1 |  | 3 | 2 |  |
| S001 | U0007 | PAR | The day is quiet. |  | 1 |  | 1 | T | T4 | 4 | 2 | 1 | 3 | 1 | 2 |  | 1 | 1 |  |
| S003 | U0001 | INV | What do you notice first? |  | 1 |  | 2 | T | T5 | 4 | 2 | 2 | 4 | 2 | 3 |  | 2 | 2 |  |

### Sheet: Turns

| sample_id | speaker | turn_label | speech_units_sum | content_words_sum | num_nouns_sum | filled_pauses_sum | circumlocutions_sum | sem_paras_sum | phon_errs_sum | neologisms_sum | lg_pauses_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | INV | T1 | 1 | 5 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| S001 | INV | T3 | 1 | 1 | 0 | 1 | 2 | 1 | 1 | 5 | 2 |
| S001 | PAR | MT1 | 2 | 4 | 2 | 2 | 1 | 4 | 2 | 4 | 1 |
| S001 | PAR | NV1 | 2 | 6 | 3 | 2 | 3 | 2 | 2 | 1 | 3 |
| S001 | PAR | ST1 | 1 | 9 | 3 | 1 | 3 | 3 | 1 | 3 | 3 |
| S001 | PAR | T2 | 2 | 7 | 3 | 2 | 2 | 2 | 2 | 2 | 2 |
| S001 | PAR | T4 | 1 | 4 | 2 | 1 | 1 | 3 | 1 | 2 | 1 |
| S002 | INV | NV3 | 2 | 4 | 2 | 2 | 3 | 2 | 2 | 3 | 3 |

### Sheet: Speakers

| sample_id | speaker | speech_units_sum | content_words_sum | num_nouns_sum | filled_pauses_sum | circumlocutions_sum | sem_paras_sum | phon_errs_sum | neologisms_sum | lg_pauses_sum | total_turns | num_T | num_MT | num_ST | num_NV | mean_turn_length | ratio_num_nouns_to_speech_units | ratio_num_nouns_to_total_turns | ratio_num_nouns_to_num_ST | ratio_content_words_to_speech_units | ratio_content_words_to_total_turns | ratio_content_words_to_num_ST | ratio_STs_to_turns | ratio_MTs_to_turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | INV | 2 | 6 | 2 | 2 | 3 | 2 | 2 | 6 | 3 | 2 | 2 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |  | 3.0 | 3.0 |  | 0.0 | 0.0 |
| S001 | PAR | 8 | 30 | 13 | 8 | 10 | 14 | 8 | 12 | 10 | 5 | 2 | 1 | 1 | 1 | 1.6 | 1.625 | 2.6 | 13.0 | 3.75 | 6.0 | 30.0 | 0.2 | 0.2 |
| S002 | INV | 3 | 12 | 5 | 3 | 6 | 5 | 3 | 8 | 6 | 2 | 0 | 0 | 1 | 1 | 1.5 | 1.666666666666667 | 2.5 | 5.0 | 4.0 | 6.0 | 12.0 | 0.5 | 0.0 |
| S002 | PAR | 7 | 24 | 10 | 7 | 9 | 13 | 7 | 13 | 9 | 5 | 3 | 1 | 1 | 0 | 1.4 | 1.428571428571429 | 2.0 | 10.0 | 3.428571428571428 | 4.8 | 24.0 | 0.2 | 0.2 |
| S003 | INV | 4 | 7 | 3 | 4 | 5 | 8 | 4 | 5 | 5 | 2 | 1 | 0 | 0 | 1 | 2.0 | 0.75 | 1.5 |  | 1.75 | 3.5 |  | 0.0 | 0.0 |
| S003 | PAR | 7 | 22 | 11 | 7 | 9 | 9 | 7 | 17 | 9 | 5 | 3 | 1 | 1 | 0 | 1.4 | 1.571428571428571 | 2.2 | 11.0 | 3.142857142857143 | 4.4 | 22.0 | 0.2 | 0.2 |

### Sheet: Dialogs

| sample_id | speech_units_sum | content_words_sum | num_nouns_sum | filled_pauses_sum | circumlocutions_sum | sem_paras_sum | phon_errs_sum | neologisms_sum | lg_pauses_sum | num_repairs | prop_repairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | 10 | 36 | 15 | 10 | 13 | 16 | 10 | 18 | 13 | 1 | 0.1428571428571428 |
| S002 | 10 | 36 | 15 | 10 | 15 | 18 | 10 | 21 | 15 | 1 | 0.2857142857142857 |
| S003 | 11 | 29 | 14 | 11 | 14 | 17 | 11 | 22 | 14 | 1 | 0.1428571428571428 |

## Notes

The preview uses synthetic filled POWERS values generated from the packaged example specs.
