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
stimulus_field: stimulus
exclude_participants:
- INV
```

## Advanced Config

```yaml
cu_paradigms: []
auto_blind: true
blind_cols:
- sample_id
metadata_source: transcript_tables
codebook_filename: ''
```

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`expected_outputs/cus_module/cus_files/cu_coding.xlsx`

| sample_id | input_order | shuffled_order | stimulus | utterance_id | position | position_sub | speaker | utterance | comment | id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 |  | picnic | U0001 | 1 | 0 | INV | Please tell the picnic story again. |  | 1 |  |  |
| 1 | 1 |  | picnic | U0002 | 2 | 0 | PAR | The family brought food to the park. |  | 1 | 1.0 | 1.0 |
| 1 | 1 |  | picnic | U0003 | 3 | 0 | PAR | The little girl [/] the little girl pours juice. |  | 1 | 0.0 | 1.0 |
| 1 | 1 |  | picnic | U0004 | 4 | 0 | PAR | Then they share sandwiches. |  | 1 | 1.0 | 0.0 |
| 1 | 1 |  | picnic | U0005 | 5 | 0 | INV | Anything else? |  | 1 |  |  |
| 1 | 1 |  | picnic | U0006 | 6 | 0 | PAR | Yes, the dog waits beside them. |  | 1 | 1.0 | 0.0 |
| 1 | 1 |  | picnic | U0007 | 7 | 0 | PAR | The day is quiet. |  | 1 | 0.0 | 1.0 |
| 2 | 3 |  | picnic | U0001 | 1 | 0 | INV | What do you notice first? |  | 1 |  |  |

`expected_outputs/cus_module/cus_files/cu_reliability_coding.xlsx`

| sample_id | input_order | shuffled_order | stimulus | utterance_id | position | position_sub | speaker | utterance | comment | id | sv | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 |  | picnic | U0001 | 1 | 0 | INV | Please tell the picnic story again. |  | 2 |  |  |
| 1 | 1 |  | picnic | U0002 | 2 | 0 | PAR | The family brought food to the park. |  | 2 | 1.0 | 0.0 |
| 1 | 1 |  | picnic | U0003 | 3 | 0 | PAR | The little girl [/] the little girl pours juice. |  | 2 | 0.0 | 1.0 |
| 1 | 1 |  | picnic | U0004 | 4 | 0 | PAR | Then they share sandwiches. |  | 2 | 1.0 | 0.0 |
| 1 | 1 |  | picnic | U0005 | 5 | 0 | INV | Anything else? |  | 2 |  |  |
| 1 | 1 |  | picnic | U0006 | 6 | 0 | PAR | Yes, the dog waits beside them. |  | 2 | 1.0 | 0.0 |
| 1 | 1 |  | picnic | U0007 | 7 | 0 | PAR | The day is quiet. |  | 2 | 0.0 | 1.0 |
| 3 | 2 |  | picnic | U0001 | 1 | 0 | INV | Tell me what is happening in the picnic picture. |  | 1 |  |  |

`expected_outputs/cus_module/cus_files/cu_blind_codebook.xlsx`

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Notes

The generated local example fills the blank CU fields with synthetic coding values so downstream CU examples can be demonstrated. Real `cus files` output starts as coding material for human review.
