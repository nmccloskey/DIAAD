---
object_type: command
object_types:
- command
command_id: vocab.rates
canonical_command: vocab rates
module_id: vocab
view: example_io
title: Target Vocabulary Rate Calculation Example
slot: examples
---

# Target Vocabulary Rate Calculation Example

This example demonstrates how `diaad vocab rates` converts target-vocabulary analysis counts into per-minute rates.

## Command

```bash
diaad vocab rates --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      target_vocab_analysis/
        target_vocab_data_YYMMDD_HHMM.xlsx
    output/
      diaad_YYMMDD_HHMM/
        target_vocab/
          target_vocab_rates.xlsx
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

The command reads `diaad_data/input/target_vocab_analysis/target_vocab_data_YYMMDD_HHMM.xlsx`.

## Output Preview

`expected_outputs/vocab_module/vocab_rates/target_vocab_rates.xlsx`

| sample_id | narrative | source_file | num_tokens | num_base_forms_produced | num_core_token_matches | speaking_time | speaking_minutes | core_tokens_per_min | num_tokens_per_min | num_base_forms_produced_per_min | num_core_token_matches_per_min | accuracy_pwa_percentile | accuracy_control_percentile | efficiency_pwa_percentile | efficiency_control_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | picnic | target_vocab_data_260101_0000.xlsx | 29 | 9 | 10 | 95 | 1.583333333333333 | 6.315789473684211 | 18.316 | 5.684 | 6.316 |  |  |  |  |
| S002 | picnic | target_vocab_data_260101_0000.xlsx | 24 | 9 | 9 | 88 | 1.466666666666667 | 6.136363636363637 | 16.364 | 6.136 | 6.136 |  |  |  |  |
| S003 | picnic | target_vocab_data_260101_0000.xlsx | 22 | 5 | 6 | 102 | 1.7 | 3.529411764705882 | 12.941 | 2.941 | 3.529 |  |  |  |  |

## Notes

DIAAD includes five built-in narrative resources: `BrokenWindow`, `CatRescue`, `Cinderella`, `RefusedUmbrella`, and `Sandwich`. Those built-ins require no user JSON. This synthetic picnic example uses a small custom JSON resource so the vocabulary targets match the synthetic transcripts. Rates are based on the `speaking_time` values stored in the target-vocabulary analysis summary sheet.
