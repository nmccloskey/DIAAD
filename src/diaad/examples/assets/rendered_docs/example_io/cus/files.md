---
object_type: command
object_types:
- command
object_id: cus.files
command_id: cus.files
canonical_command: cus files
module_id: cus
title: CU Coding File Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# CU Coding File Example

This example demonstrates how `diaad cus files` creates complete-utterance coding and reliability workbooks from transcript tables.

## Command

```bash
diaad cus files --config config
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
        cu_coding/
          cu_coding.xlsx
          cu_reliability_coding.xlsx
          cu_blind_codebook.xlsx
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
```

## Advanced Config

```yaml
cu_paradigms: []
auto_blind: true
blind_columns:
- sample_id
metadata_source: transcript_tables.xlsx
codebook_filename: ''
```

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/cu_coding/cu_coding.xlsx`

| sample_id | stimulus | utterance_id | speaker | utterance | comment | id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | picnic | U0001 | INV | Please tell the picnic story again. |  | 1 |  |  |
| 1 | picnic | U0002 | PAR | The family brought food to the park. |  | 1 | 1.0 | 1.0 |
| 1 | picnic | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 1 | 0.0 | 1.0 |
| 1 | picnic | U0004 | PAR | Then they share sandwiches. |  | 1 | 1.0 | 0.0 |
| 1 | picnic | U0005 | INV | Anything else? |  | 1 |  |  |
| 1 | picnic | U0006 | PAR | Yes, the dog waits beside them. |  | 1 | 1.0 | 0.0 |
| 1 | picnic | U0007 | PAR | The day is quiet. |  | 1 | 0.0 | 1.0 |
| 2 | picnic | U0001 | INV | What do you notice first? |  | 1 |  |  |

`diaad_data/output/diaad_YYMMDD_HHMM/cu_coding/cu_reliability_coding.xlsx`

| sample_id | stimulus | utterance_id | speaker | utterance | comment | id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | picnic | U0001 | INV | Please tell the picnic story again. |  | 2 |  |  |
| 1 | picnic | U0002 | PAR | The family brought food to the park. |  | 2 | 1.0 | 0.0 |
| 1 | picnic | U0003 | PAR | The little girl [/] the little girl pours juice. |  | 2 | 0.0 | 1.0 |
| 1 | picnic | U0004 | PAR | Then they share sandwiches. |  | 2 | 1.0 | 0.0 |
| 1 | picnic | U0005 | INV | Anything else? |  | 2 |  |  |
| 1 | picnic | U0006 | PAR | Yes, the dog waits beside them. |  | 2 | 1.0 | 0.0 |
| 1 | picnic | U0007 | PAR | The day is quiet. |  | 2 | 0.0 | 1.0 |
| 3 | picnic | U0001 | INV | Tell me what is happening in the picnic picture. |  | 1 |  |  |

`diaad_data/output/diaad_YYMMDD_HHMM/cu_coding/cu_blind_codebook.xlsx`

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Notes

The generated local example fills the blank CU fields with synthetic coding values so downstream CU examples can be demonstrated. Real `cus files` output starts as coding material for human review.
