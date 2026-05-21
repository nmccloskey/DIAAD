# Blinding Encode Example

This example demonstrates how `diaad blinding encode` blinds `sample_id` in a standalone workbook and writes a reusable codebook.

## Command

```bash
diaad blinding encode --config config
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
        blinding/
          powers_coding_blinded.xlsx
          powers_coding_blinding_diagnostics.xlsx
          blind_codebook.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
random_seed: 99
```

## Advanced Config

```yaml
auto_blind: true
blind_columns:
- sample_id
```

## Input Snippet

`diaad_data/input/powers_coding/powers_coding.xlsx`

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

## Output Preview

`expected_outputs/blinding_module/blinding_encode/powers_coding_blinded.xlsx`

| utterance_id | speaker | utterance | comment | coder_id | POWERS_comment | speech_units | turn_type | content_words | num_nouns | circumlocutions | sem_paras | phon_errs | neologisms | comments | lg_pauses | filled_pauses | collab_repair | sample_id_blinded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0001 | INV | Please tell the picnic story again. |  | 1 |  | 1 | T | 5 | 2 | 1 | 1 | 1 | 1 |  | 1 | 1 |  | 1 |
| U0002 | PAR | The family brought food to the park. |  | 1 |  | 2 | T | 7 | 3 | 2 | 2 | 2 | 2 |  | 2 | 2 |  | 1 |
| U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 |  | 1 | ST | 9 | 3 | 3 | 3 | 1 | 3 |  | 3 | 1 | repair_1 | 1 |
| U0004 | PAR | Then they share sandwiches. |  | 1 |  | 2 | MT | 4 | 2 | 1 | 4 | 2 | 4 |  | 1 | 2 |  | 1 |
| U0005 | INV | Anything else? |  | 1 |  | 1 | T | 1 | 0 | 2 | 1 | 1 | 5 |  | 2 | 1 |  | 1 |
| U0006 | PAR | Yes, the dog waits beside them. |  | 1 |  | 2 | NV | 6 | 3 | 3 | 2 | 2 | 1 |  | 3 | 2 |  | 1 |
| U0007 | PAR | The day is quiet. |  | 1 |  | 1 | T | 4 | 2 | 1 | 3 | 1 | 2 |  | 1 | 1 |  | 1 |
| U0001 | INV | What do you notice first? |  | 1 |  | 2 | T | 4 | 2 | 2 | 4 | 2 | 3 |  | 2 | 2 |  | 3 |

`expected_outputs/blinding_module/blinding_encode/blind_codebook.xlsx`

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | 1 | 1 |
| sample_id | 3 | 2 |
| sample_id | 2 | 3 |

`expected_outputs/blinding_module/blinding_encode/powers_coding_blinding_diagnostics.xlsx`

| sample_id | utterance_id | sample_id_blinded |
| --- | --- | --- |
| 1 | U0001 | 1 |
| 1 | U0002 | 1 |
| 1 | U0003 | 1 |
| 1 | U0004 | 1 |
| 1 | U0005 | 1 |
| 1 | U0006 | 1 |
| 1 | U0007 | 1 |
| 2 | U0001 | 3 |

## Notes

The input is the synthetic POWERS coding workbook from the generated example project. The command discovers the first non-codebook `.xlsx` in the input folder and generates a new blind codebook because no codebook is supplied.
