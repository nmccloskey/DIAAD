---
object_type: command
object_types:
- command
command_id: powers.files
canonical_command: powers files
module_id: powers
view: example_io
title: POWERS Coding File Example
slot: examples
---

# POWERS Coding File Example

This example demonstrates how `diaad powers files` creates POWERS coding and reliability workbooks from transcript tables.

## Command

```bash
diaad powers files --config config
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
        powers_coding/
          powers_coding.xlsx
          powers_reliability_coding.xlsx
          powers_blind_codebook.xlsx
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
exclude_speakers:
- INV
automate_powers: false
```

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_files/powers_coding.xlsx`

### Sheet: utterance_coding

| sample_id | utterance_id | speaker | utterance | comment | coder_id | POWERS_comment | speech_units | turn_type | content_words | num_nouns | circumlocutions | sem_paras | phon_errs | neologisms | comments | lg_pauses | filled_pauses | collab_repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | U0001 | INV | Please tell the picnic story again. |  | 1 |  | 1 | T | 5 | 2 | 1 | 1 | 1 | 1 |  | 1 | 1 |  |
| 1 | U0002 | PAR | The family brought food to the park. |  | 1 |  | 2 | T | 7 | 3 | 2 | 2 | 2 | 2 |  | 2 | 2 |  |
| 1 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 |  | 1 | ST | 9 | 3 | 3 | 3 | 1 | 3 |  | 3 | 1 | repair_1 |
| 1 | U0004 | PAR | Then they share sandwiches. |  | 1 |  | 2 | MT | 4 | 2 | 1 | 4 | 2 | 4 |  | 1 | 2 |  |
| 1 | U0005 | INV | Anything else? |  | 1 |  | 1 | T | 1 | 0 | 2 | 1 | 1 | 5 |  | 2 | 1 |  |
| 1 | U0006 | PAR | Yes, the dog waits beside them. |  | 1 |  | 2 | NV | 6 | 3 | 3 | 2 | 2 | 1 |  | 3 | 2 |  |
| 1 | U0007 | PAR | The day is quiet. |  | 1 |  | 1 | T | 4 | 2 | 1 | 3 | 1 | 2 |  | 1 | 1 |  |
| 2 | U0001 | INV | What do you notice first? |  | 1 |  | 2 | T | 4 | 2 | 2 | 4 | 2 | 3 |  | 2 | 2 |  |

### Sheet: section_e

| sample_id | type_of_day | amount_of_enjoyment | degree_of_difficulty | other_notes |
| --- | --- | --- | --- | --- |
| 1 | weekday | 4 | 2 |  |
| 2 | weekday | 3 | 2 |  |
| 3 | weekend | 4 | 1 |  |

`expected_outputs/powers_module/powers_files/powers_reliability_coding.xlsx`

| sample_id | utterance_id | speaker | utterance | comment | coder_id | POWERS_comment | speech_units | turn_type | content_words | num_nouns | circumlocutions | sem_paras | phon_errs | neologisms | comments | lg_pauses | filled_pauses | collab_repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | U0001 | INV | Please tell the picnic story again. |  | 2 |  | 1 | T | 4 | 2 | 1 | 1 | 1 | 1 |  | 1 | 2 |  |
| 1 | U0002 | PAR | The family brought food to the park. |  | 2 |  | 2 | T | 7 | 3 | 2 | 2 | 2 | 2 |  | 2 | 2 |  |
| 1 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 2 |  | 1 | ST | 9 | 3 | 3 | 3 | 1 | 3 |  | 3 | 1 | repair_1 |
| 1 | U0004 | PAR | Then they share sandwiches. |  | 2 |  | 2 | MT | 4 | 2 | 1 | 4 | 2 | 4 |  | 1 | 2 |  |
| 1 | U0005 | INV | Anything else? |  | 2 |  | 1 | T | 1 | 0 | 2 | 1 | 1 | 5 |  | 2 | 1 |  |
| 1 | U0006 | PAR | Yes, the dog waits beside them. |  | 2 |  | 2 | NV | 5 | 3 | 3 | 2 | 2 | 1 |  | 3 | 2 |  |
| 1 | U0007 | PAR | The day is quiet. |  | 2 |  | 1 | T | 4 | 2 | 1 | 3 | 1 | 2 |  | 1 | 1 |  |
| 3 | U0001 | INV | Tell me what is happening in the picnic picture. |  | 1 |  | 2 | T | 8 | 3 | 2 | 4 | 2 | 3 |  | 2 | 2 |  |

`expected_outputs/powers_module/powers_files/powers_blind_codebook.xlsx`

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Notes

The generated local example fills synthetic POWERS values into the blank coding workbooks so downstream POWERS examples can be demonstrated. The blind codebook lets later POWERS analysis reconnect blinded sample IDs to speaking times. Real `powers files` output starts as coding material for human review.
