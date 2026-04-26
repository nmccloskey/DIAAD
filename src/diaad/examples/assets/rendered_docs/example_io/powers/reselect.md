# POWERS Reliability Reselection Example

This example demonstrates how `diaad powers reselect` selects replacement POWERS reliability rows after an earlier reliability workbook has already been used.

## Command

```bash
diaad powers reselect --config config
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
        powers_reliability_coding.xlsx
    output/
      diaad_YYMMDD_HHMM/
        reselected_powers_reliability/
          reselected_powers_reliability_coding.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.34
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
automate_powers: false
```

## Input Snippet

The command reads prior POWERS coding and reliability workbooks from `diaad_data/input/powers_coding/`.

## Output Preview

`expected_outputs/powers_module/powers_reselect/reselected_powers_reliability_coding.xlsx`

| sample_id | utterance_id | speaker | utterance | comment | coder_id | POWERS_comment | speech_units | turn_type | content_words | num_nouns | circumlocutions | sem_paras | phon_errs | neologisms | comments | lg_pauses | filled_pauses | collab_repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S003 | U0001 | INV | What do you notice first? |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S003 | U0002 | PAR | A picnic. |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S003 | U0003 | PAR | The dad is opening the basket. |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S003 | U0004 | PAR | The dog wants food! |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S003 | U0005 | INV | What might happen next? |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S003 | U0006 | PAR | They will eat lunch. |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S003 | U0007 | PAR | Maybe the dog gets a cracker. |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Notes

The synthetic example has only three samples, so the reselected workbook is intentionally tiny.
