---
object_type: command
object_types:
- command
object_id: blinding.decode
command_id: blinding.decode
canonical_command: blinding decode
module_id: blinding
title: Blinding Decode Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Blinding Decode Example

This example demonstrates how `diaad blinding decode` restores blinded identifiers in a standalone workbook using a blind codebook.

## Command

```bash
diaad blinding decode --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      cu_coding/
        cu_coding.xlsx
        cu_blind_codebook.xlsx
    output/
      diaad_YYMMDD_HHMM/
        blinding/
          cu_coding_decoded.xlsx
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

`diaad_data/input/cu_coding/cu_coding.xlsx`

| sample_id | stimulus | utterance_id | speaker | utterance | comment | coder_id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | picnic | U0001 | INV | Please tell the picnic story again. |  | 1 |  |  |
| 1 | picnic | U0002 | PAR | The family brought food to the park. |  | 1 | 1.0 | 1.0 |
| 1 | picnic | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 | 0.0 | 1.0 |
| 1 | picnic | U0004 | PAR | Then they share sandwiches. |  | 1 | 1.0 | 0.0 |
| 1 | picnic | U0005 | INV | Anything else? |  | 1 |  |  |
| 1 | picnic | U0006 | PAR | Yes, the dog waits beside them. |  | 1 | 1.0 | 0.0 |
| 1 | picnic | U0007 | PAR | The day is quiet. |  | 1 | 0.0 | 1.0 |
| 2 | picnic | U0001 | INV | What do you notice first? |  | 1 |  |  |

`diaad_data/input/cu_coding/cu_blind_codebook.xlsx`

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/blinding/cu_coding_decoded.xlsx`

| sample_id | stimulus | utterance_id | speaker | utterance | comment | coder_id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | picnic | U0001 | INV | Please tell the picnic story again. |  | 1 |  |  |
| S001 | picnic | U0002 | PAR | The family brought food to the park. |  | 1 | 1.0 | 1.0 |
| S001 | picnic | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 | 0.0 | 1.0 |
| S001 | picnic | U0004 | PAR | Then they share sandwiches. |  | 1 | 1.0 | 0.0 |
| S001 | picnic | U0005 | INV | Anything else? |  | 1 |  |  |
| S001 | picnic | U0006 | PAR | Yes, the dog waits beside them. |  | 1 | 1.0 | 0.0 |
| S001 | picnic | U0007 | PAR | The day is quiet. |  | 1 | 0.0 | 1.0 |
| S003 | picnic | U0001 | INV | What do you notice first? |  | 1 |  |  |

## Notes

The input is the synthetic CU coding workbook and codebook created by the CU examples. The decode command discovers the codebook, restores `sample_id`, and removes the suffixed blinded identifier column.
