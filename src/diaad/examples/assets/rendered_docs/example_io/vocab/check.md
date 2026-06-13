---
object_type: command
object_types:
- command
command_id: vocab.check
canonical_command: vocab check
module_id: vocab
view: example_io
title: Target Vocabulary Resource Check Example
slot: examples
---

# Target Vocabulary Resource Check Example

This example demonstrates how `diaad vocab check` validates the active built-in and custom target-vocabulary resources.

## Command

```bash
diaad vocab check --config config
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
        resources/
          picnic_target_vocab.json
    output/
      diaad_YYMMDD_HHMM/
        target_vocab/
          target_vocab_resource_check.txt
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

## Output Preview

`expected_outputs/vocab_module/vocab_check/target_vocab_resource_check.txt`

```text
Target vocabulary resource check

Custom resource path: input/target_vocab/resources/picnic_target_vocab.json
Custom resource id: picnic
Active resource count: 6
Active resource ids:
- BrokenWindow
- CatRescue
- Cinderella
- RefusedUmbrella
- Sandwich
- picnic

Built-in narrative resources remain available when a custom JSON path is configured.
```

## Notes

DIAAD includes five built-in narrative resources: `BrokenWindow`, `CatRescue`, `Cinderella`, `RefusedUmbrella`, and `Sandwich`. Those built-ins require no user JSON. This synthetic picnic example uses a small custom JSON resource so the vocabulary targets match the synthetic transcripts. The command reports validation details through the DIAAD run log; the generated text file shown here is a compact documentation preview of the same resource set.
