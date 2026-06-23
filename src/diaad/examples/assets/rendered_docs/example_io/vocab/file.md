---
object_type: command
object_types:
- command
object_id: vocab.file
command_id: vocab.file
canonical_command: vocab file
module_id: vocab
title: Target Vocabulary Resource Template Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Target Vocabulary Resource Template Example

This example demonstrates how `diaad vocab file` creates a blank JSON template for a custom target-vocabulary resource.

## Command

```bash
diaad vocab file --config config
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
    output/
      diaad_YYMMDD_HHMM/
        target_vocab/
          target_vocabulary_resource_template.json
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/target_vocab/target_vocabulary_resource_template.json`

```json
{
  "resource_id": "",
  "display_name": "",
  "language": "",
  "task_type": "",
  "base_forms": [],
  "variant_map": {},
  "norms": {
    "accuracy": {
      "url": "",
      "format": "",
      "columns": {
        "raw_score": "",
        "group": "",
        "pwa_percentile": "",
        "control_percentile": ""
      }
    },
    "efficiency": {
      "url": "",
      "format": "",
      "columns": {
        "raw_score": "",
        "group": "",
        "pwa_percentile": "",
        "control_percentile": ""
      }
    }
  }
}
```

## Notes

DIAAD includes five built-in narrative resources: `BrokenWindow`, `CatRescue`, `Cinderella`, `RefusedUmbrella`, and `Sandwich`. Those built-ins require no user JSON. This synthetic picnic example uses a small custom JSON resource so the vocabulary targets match the synthetic transcripts. Use `diaad vocab file` when a project needs to start authoring a custom resource.
