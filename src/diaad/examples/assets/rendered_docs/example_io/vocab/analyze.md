---
object_type: command
object_types:
- command
object_id: vocab.analyze
command_id: vocab.analyze
canonical_command: vocab analyze
module_id: vocab
title: Target Vocabulary Analysis Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Target Vocabulary Analysis Example

This example demonstrates how `diaad vocab analyze` calculates target-vocabulary coverage for synthetic picnic samples.

## Command

```bash
diaad vocab analyze --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      target_vocab/
        unblind_utterance_data.xlsx
        resources/
          picnic_target_vocab.json
    output/
      diaad_YYMMDD_HHMM/
        target_vocab/
          target_vocab_data_YYMMDD_HHMM.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
stimulus_column: stimulus
exclude_speakers:
- INV
```

## Advanced Config

```yaml
target_vocabulary_resource_path: diaad_data/input/target_vocab/resources/picnic_target_vocab.json
```

## Input Snippet

`diaad_data/input/target_vocab/resources/picnic_target_vocab.json`

```json
{
  "id": "picnic",
  "display_name": "Synthetic Picnic",
  "base_forms": [
    "apple",
    "basket",
    "blanket",
    "child",
    "day",
    "dog",
    "drink",
    "family"
  ],
  "variant_map": {
    "apple": [
      "apples"
    ],
    "child": [
      "children"
    ],
    "drink": [
      "drinks"
    ]
  }
}
```

`diaad_data/input/target_vocab/unblind_utterance_data.xlsx`

| sample_id | stimulus | timepoint | utterance_id | speaker | utterance | speaking_time | word_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | picnic | post | U0002 | PAR | The family brought food to the park. | 95 | 7 |
| S001 | picnic | post | U0003 | PAR | The little girl [/] the little girl pours juice. | 95 | 8 |
| S001 | picnic | post | U0004 | PAR | Then they share sandwiches. | 95 | 4 |
| S001 | picnic | post | U0006 | PAR | Yes, the dog waits beside them. | 95 | 6 |
| S001 | picnic | post | U0007 | PAR | The day is quiet. | 95 | 4 |
| S002 | picnic | pre | U0002 | PAR | The family is sitting on a blanket. | 88 | 7 |
| S002 | picnic | pre | U0003 | PAR | They have sandwiches, apples, and juice. | 88 | 6 |
| S002 | picnic | pre | U0005 | PAR | She is pouring a drink. | 88 | 5 |

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/target_vocab/target_vocab_data_260101_0000.xlsx`

### Sheet: summary

| sample_id | narrative | speaking_time | num_tokens | num_base_forms_produced | num_core_token_matches | lexicon_coverage | core_tokens_per_min | accuracy_pwa_percentile | accuracy_control_percentile | efficiency_pwa_percentile | efficiency_control_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | picnic | 95 | 29 | 9 | 10 | 0.5 | 6.315789473684211 |  |  |  |  |
| S002 | picnic | 88 | 24 | 9 | 9 | 0.5 | 6.136363636363637 |  |  |  |  |
| S003 | picnic | 102 | 22 | 5 | 6 | 0.2777777777777778 | 3.529411764705882 |  |  |  |  |

### Sheet: details

| sample_id | narrative | base_form | num_tokens_matched | score |
| --- | --- | --- | --- | --- |
| S001 | picnic | apple | 0 | 0 |
| S001 | picnic | basket | 0 | 0 |
| S001 | picnic | blanket | 0 | 0 |
| S001 | picnic | child | 0 | 0 |
| S001 | picnic | day | 1 | 1 |
| S001 | picnic | dog | 1 | 1 |
| S001 | picnic | drink | 0 | 0 |
| S001 | picnic | family | 1 | 1 |

## Notes

DIAAD includes five built-in narrative resources: `BrokenWindow`, `CatRescue`, `Cinderella`, `RefusedUmbrella`, and `Sandwich`. Those built-ins require no user JSON. This synthetic picnic example uses a small custom JSON resource so the vocabulary targets match the synthetic transcripts. The custom picnic resource intentionally has no norm tables, so percentile columns are blank while coverage counts and rates are still calculated.
