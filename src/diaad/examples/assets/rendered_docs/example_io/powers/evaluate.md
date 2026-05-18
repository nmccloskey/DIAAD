# POWERS Reliability Evaluation Example

This example demonstrates how `diaad powers evaluate` compares primary POWERS coding with a synthetic reliability workbook.

## Command

```bash
diaad powers evaluate --config config
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
        powers_reliability/
          powers_reliability_results.xlsx
          powers_reliability_report.txt
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

The command reads `diaad_data/input/powers_coding/powers_coding.xlsx` and `diaad_data/input/powers_coding/powers_reliability_coding.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_evaluate/powers_reliability_results.xlsx`

### Sheet: merged

| reliability_id | sample_id | utterance_id | speaker | utterance | comment | coder_id | POWERS_comment | speech_units_org | turn_type_org | content_words_org | num_nouns_org | circumlocutions_org | sem_paras_org | phon_errs_org | neologisms_org | comments | lg_pauses_org | filled_pauses_org | collab_repair_org | speech_units_rel | content_words_rel | num_nouns_rel | filled_pauses_rel | circumlocutions_rel | sem_paras_rel | phon_errs_rel | neologisms_rel | lg_pauses_rel | turn_type_rel | collab_repair_rel | speech_units_abs_diff | speech_units_perc_diff | speech_units_perc_sim | content_words_abs_diff | content_words_perc_diff | content_words_perc_sim | num_nouns_abs_diff | num_nouns_perc_diff | num_nouns_perc_sim | filled_pauses_abs_diff | filled_pauses_perc_diff | filled_pauses_perc_sim | circumlocutions_abs_diff | circumlocutions_perc_diff | circumlocutions_perc_sim | sem_paras_abs_diff | sem_paras_perc_diff | sem_paras_perc_sim | phon_errs_abs_diff | phon_errs_perc_diff | phon_errs_perc_sim | neologisms_abs_diff | neologisms_perc_diff | neologisms_perc_sim | lg_pauses_abs_diff | lg_pauses_perc_diff | lg_pauses_perc_sim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | S001 | U0001 | INV | Please tell the picnic story again. |  | 1 |  | 1 | T | 5 | 2 | 1 | 1 | 1 | 1 |  | 1 | 1 |  | 1 | 4 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | T |  | 0 | 0.0 | 100.0 | 1 | 22.22 | 77.78 | 0 | 0 | 100 | 1 | 66.67 | 33.33 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 2 | S001 | U0002 | PAR | The family brought food to the park. |  | 1 |  | 2 | T | 7 | 3 | 2 | 2 | 2 | 2 |  | 2 | 2 |  | 2 | 7 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | T |  | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0 | 100 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 3 | S001 | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 |  | 1 | ST | 9 | 3 | 3 | 3 | 1 | 3 |  | 3 | 1 | repair_1 | 1 | 9 | 3 | 1 | 3 | 3 | 1 | 3 | 3 | ST | repair_1 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0 | 100 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 4 | S001 | U0004 | PAR | Then they share sandwiches. |  | 1 |  | 2 | MT | 4 | 2 | 1 | 4 | 2 | 4 |  | 1 | 2 |  | 2 | 4 | 2 | 2 | 1 | 4 | 2 | 4 | 1 | MT |  | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0 | 100 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 5 | S001 | U0005 | INV | Anything else? |  | 1 |  | 1 | T | 1 | 0 | 2 | 1 | 1 | 5 |  | 2 | 1 |  | 1 | 1 | 0 | 1 | 2 | 1 | 1 | 5 | 2 | T |  | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0 | 100 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 6 | S001 | U0006 | PAR | Yes, the dog waits beside them. |  | 1 |  | 2 | NV | 6 | 3 | 3 | 2 | 2 | 1 |  | 3 | 2 |  | 2 | 5 | 3 | 2 | 3 | 2 | 2 | 1 | 3 | NV |  | 0 | 0.0 | 100.0 | 1 | 18.18 | 81.82 | 0 | 0 | 100 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 7 | S001 | U0007 | PAR | The day is quiet. |  | 1 |  | 1 | T | 4 | 2 | 1 | 3 | 1 | 2 |  | 1 | 1 |  | 1 | 4 | 2 | 1 | 1 | 3 | 1 | 2 | 1 | T |  | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0 | 100 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 | 0 | 0.0 | 100.0 |
| 8 | S002 | U0001 | INV | Tell me what is happening in the picnic picture. |  | 2 |  | 1 | ST | 8 | 3 | 3 | 3 | 1 | 5 |  | 3 | 1 | repair_1 | 2 | 8 | 3 | 2 | 2 | 4 | 2 | 3 | 2 | T |  | 1 | 66.67 | 33.33 | 0 | 0.0 | 100.0 | 0 | 0 | 100 | 1 | 66.67 | 33.33 | 1 | 40.0 | 60.0 | 1 | 28.57 | 71.43 | 1 | 66.67 | 33.33 | 2 | 50.0 | 50.0 | 1 | 40.0 | 60.0 |

### Sheet: continuous_summary

| metric | paired_utterances | mean_abs_diff | mean_perc_diff | mean_perc_sim | exact_agreement_pct | within_1_count_pct | ICC2 | ICC_warning | total_merged_rows | org_missing | rel_missing | rows_dropped_missing | rows_dropped_missing_pct | org_mean | rel_mean | org_sd | rel_sd | org_var | rel_var | pooled_sd | pooled_var | org_unique_values | rel_unique_values | max_value_prop_pct | both_zero_pct | nonzero_either_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speech_units | 14 | 0.5 | 33.335 | 66.665 | 50.0 | 100.0 | 0.0 |  | 14 | 0 | 0 | 0 | 0 | 1.4286 | 1.5 | 0.5136 | 0.5189 | 0.2637 | 0.2692 | 0.5162 | 0.2665 | 2 | 2 | 57.143 | 0.0 | 100.0 |
| content_words | 14 | 0.214 | 4.926 | 95.074 | 78.571 | 100.0 | 0.9781 |  | 14 | 0 | 0 | 0 | 0 | 5.1429 | 4.9286 | 2.1788 | 2.2348 | 4.7473 | 4.9945 | 2.207 | 4.8709 | 8 | 8 | 21.429 | 0.0 | 100.0 |
| num_nouns | 14 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 |  | 14 | 0 | 0 | 0 | 0 | 2.1429 | 2.1429 | 0.9493 | 0.9493 | 0.9011 | 0.9011 | 0.9493 | 0.9011 | 4 | 4 | 42.857 | 7.143 | 92.857 |
| filled_pauses | 14 | 0.5 | 33.335 | 66.665 | 50.0 | 100.0 | 0.0421 |  | 14 | 0 | 0 | 0 | 0 | 1.4286 | 1.6429 | 0.5136 | 0.4972 | 0.2637 | 0.2473 | 0.5055 | 0.2555 | 2 | 2 | 64.286 | 0.0 | 100.0 |
| circumlocutions | 14 | 0.643 | 32.381 | 67.619 | 50.0 | 85.714 | 0.332 |  | 14 | 0 | 0 | 0 | 0 | 2.0 | 1.9286 | 0.8771 | 0.8287 | 0.7692 | 0.6868 | 0.8532 | 0.728 | 3 | 3 | 35.714 | 0.0 | 100.0 |
| sem_paras | 14 | 0.786 | 33.606 | 66.394 | 50.0 | 85.714 | 0.3542 |  | 14 | 0 | 0 | 0 | 0 | 2.4286 | 2.3571 | 1.1579 | 1.1507 | 1.3407 | 1.3242 | 1.1543 | 1.3324 | 4 | 4 | 28.571 | 0.0 | 100.0 |
| phon_errs | 14 | 0.5 | 33.335 | 66.665 | 50.0 | 100.0 | 0.0 |  | 14 | 0 | 0 | 0 | 0 | 1.4286 | 1.5 | 0.5136 | 0.5189 | 0.2637 | 0.2692 | 0.5162 | 0.2665 | 2 | 2 | 57.143 | 0.0 | 100.0 |
| neologisms | 14 | 1.214 | 42.313 | 57.687 | 50.0 | 50.0 | 0.275 |  | 14 | 0 | 0 | 0 | 0 | 2.7857 | 2.8571 | 1.5777 | 1.4064 | 2.489 | 1.978 | 1.4945 | 2.2335 | 5 | 5 | 28.571 | 0.0 | 100.0 |

### Sheet: categorical_summary

| metric | paired_utterances | percent_agreement | kappa |
| --- | --- | --- | --- |
| turn_type | 14 | 57.1 | 0.3333 |
| collab_repair | 14 | 78.6 | 0.2759 |

`expected_outputs/powers_module/powers_evaluate/powers_reliability_report.txt`

```text
POWERS Reliability Report

Source reliability file: powers_reliability_coding.xlsx

Paired utterances: 14

Note: ICC(2,1) is variance-sensitive. Sparse or low-variance count metrics may show low ICC values despite high exact agreement or small absolute differences. For flagged metrics, interpret ICC alongside agreement and distribution diagnostics.

Continuous metrics
------------------
speech_units: n=14, mean_abs_diff=0.5, mean_perc_diff=33.335%, mean_perc_sim=66.665%, exact_agreement=50.0%, within_1_count=100.0%, ICC(2,1)=-0.0
content_words: n=14, mean_abs_diff=0.214, mean_perc_diff=4.926%, mean_perc_sim=95.074%, exact_agreement=78.571%, within_1_count=100.0%, ICC(2,1)=0.9781
```

## Notes

Reliability POWERS values are synthetic, with small deterministic differences from the primary coding.
